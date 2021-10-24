from typing import Union, Callable, Optional
import gym, warnings
import numpy as np
from cvxopt import matrix, solvers
from sb3_contrib.common.safety.safe_region import SafeRegion
from stable_baselines3.common.type_aliases import GymStepReturn

class SafetyCBF(gym.Wrapper):

    """
    Safety wrapper that uses discrete-time control barrier functions to stay inside a safe region
    Pass either the unactuated dynamics or the dynamics
    Based on https://arxiv.org/pdf/1903.08792.pdf

    @param env: Gym environment to be wrapped
    @param safe_region: c
    @param actuated_dynamics_fn: Actuated dynamics function of the variables that need to be rendered safe
    @param unactuated_dynamics_fn: Unactuated dynamics function of the variables that need to be rendered safe
    @param dynamics_fn: Dynamics function of the variables that need to be rendered safe
    @param punishment_fn: Optional reward punishment function
    @param alter_action_space: Alternative gym action space
    @param transform_action_space_fn: Action space transformation function
    @param gamma
    """

    def __init__(self,
                 env: gym.Env,
                 safe_region: SafeRegion,
                 actuated_dynamics_fn: Union[Callable[[gym.Env, Union[int, float, np.ndarray]], np.ndarray], str],
                 unactuated_dynamics_fn: Optional[Union[Callable[[gym.Env], np.ndarray], str]] = None,
                 dynamics_fn: Optional[Union[Callable[[gym.Env, Union[int, float, np.ndarray]], np.ndarray], str]] = None,
                 punishment_fn: Optional[Union[Callable[[gym.Env, SafeRegion, Union[int, float, np.ndarray],
                                                         Union[int, float, np.ndarray]], float], str]] = None,
                 alter_action_space: Optional[gym.Space] = None,
                 transform_action_space_fn: Optional[Union[Callable, str]] = None,
                 gamma: float = .5):


        super().__init__(env)

        if unactuated_dynamics_fn is None and dynamics_fn is None:
            raise ValueError("Either dynamics_fn or unactuated_dynamics have to be specified")

        self._gamma = gamma
        self._safe_region = safe_region

        if not hasattr(self.env, "action_space"):
            warnings.warn("Environment does not have attribute ``action_space``")
        if not hasattr(self.env, "state"):
            warnings.warn("Environment does not have attribute ``state``")

        if isinstance(self.action_space, gym.spaces.Discrete):
            _num_actions = self.action_space.N
        elif isinstance(self.action_space, gym.spaces.Box):
            _num_actions = sum(self.action_space.shape)
        else:
            raise RuntimeError(f"Action space {self.action_space} not supported")

        # Alter action space if necessary
        if alter_action_space is not None:
            self.action_space = alter_action_space

        # Half-space representation of safe region
        self._A, self._b = safe_region.halfspaces

        # Setup quadratic program (objective) https://cvxopt.org/userguide/coneprog.html
        self._P = matrix(np.identity(_num_actions), tc='d')
        self._q = matrix(np.zeros(_num_actions), tc='d')

        # Fetch unbounded functions
        if isinstance(actuated_dynamics_fn, str):
            fn = getattr(self.env, actuated_dynamics_fn)
            if not callable(fn):
                raise ValueError(f"Attribute {fn} is not a method")
            self._actuated_dynamics_fn = fn
        else:
            self._actuated_dynamics_fn = actuated_dynamics_fn

        if unactuated_dynamics_fn is not None:
            if isinstance(unactuated_dynamics_fn, str):
                fn = getattr(self.env, unactuated_dynamics_fn)
                if not callable(fn):
                    raise ValueError(f"Attribute {fn} is not a method")
                self._unactuated_dynamics_fn = fn
            else:
                self._unactuated_dynamics_fn = unactuated_dynamics_fn
        else:
            if isinstance(dynamics_fn, str):
                fn = getattr(self.env, dynamics_fn)
                if not callable(fn):
                    raise ValueError(f"Attribute {fn} is not a method")
                self._dynamics_fn = fn
            else:
                self._dynamics_fn = dynamics_fn
            self._unactuated_dynamics_fn = None

        if punishment_fn is not None:
            if isinstance(punishment_fn, str):
                fn = getattr(self.env, punishment_fn)
                if not callable(fn):
                    raise ValueError(f"Attribute {fn} is not a method")
                self._punishment_fn = fn
            else:
                self._punishment_fn = punishment_fn
        else:
            self._punishment_fn = None

        if transform_action_space_fn is not None:
            if isinstance(transform_action_space_fn, str):
                fn = getattr(self.env, transform_action_space_fn)
                if not callable(fn):
                    raise ValueError(f"Attribute {fn} is not a method")
                self._transform_action_space_fn = fn
            else:
                self._transform_action_space_fn = transform_action_space_fn
        else:
            self._transform_action_space_fn = None

        if alter_action_space is not None:
            if self._transform_action_space_fn is None:
                raise ValueError("transform_action_space_fn is None")

    def step(self, action) -> GymStepReturn:

        # Transform action if necessary
        if self._transform_action_space_fn is not None:
            action = self._transform_action_space_fn(action)

        # Setup quadratic program (CBF constraint) https://cvxopt.org/userguide/coneprog.html
        G = matrix(np.asarray([[np.dot(p, self._actuated_dynamics_fn(self.env))] for p in self._A],
                              dtype=np.double),tc='d')

        # Check whether the unactuated_dynamics_fn or the dynamics_fn have been provided
        if self._unactuated_dynamics_fn is not None:
            h = matrix(np.asarray([-np.dot(self._A[i], self._unactuated_dynamics_fn(self.env))
                                   - np.dot(self._A[i], self._actuated_dynamics_fn(self.env)) * action
                                   + (1 - self._gamma) * np.dot(self._A[i], self.env.state) +
                                   self._gamma * self._b[i] for i in range(len(self._A))],
                                  dtype=np.double), tc='d')
        else:
            h = matrix(np.asarray([- np.dot(self._A[i], self._dynamics_fn(self.env, action))
                                   + (1 - self._gamma) * np.dot(self._A[i], self.env.state) +
                                   self._gamma * self._b[i] for i in range(len(self._A))],
                                  dtype=np.double), tc='d')

        # Solve QP
        sol = solvers.qp(self._P, self._q, G, h)
        compensation = sol['x'][0]

        # Forward safe action action + action_cbf
        obs, reward, done, info = self.env.step(action + compensation)

        # Optional reward punishment
        if self._punishment_fn is not None:
            punishment = self._punishment_fn(self.env, self._safe_region, action, compensation)
            info["cbf"] = {"action_rl": action,
                           "compensation": compensation,
                           "reward_rl": reward,
                           "punishment": punishment}
            reward += punishment
        else:
            info["cbf"] = {"action_rl": action,
                           "compensation": compensation,
                           "reward_rl": reward,
                           "punishment": None}

        return obs, reward, done, info