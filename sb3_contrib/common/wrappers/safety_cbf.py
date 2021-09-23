from typing import Union, Callable, Optional

import gym, warnings
import numpy as np
from cvxopt import matrix, solvers
from sb3_contrib.common.safety.safe_region import SafeRegion
from stable_baselines3.common.type_aliases import GymStepReturn


# State from Observation
# State requiered?
# Safe function not needed?

class SafetyCBF(gym.Wrapper):
    """

    :param env: Gym environment to be wrapped
    :param safe_region: Safe region object
    :param unactuated_dynamics: ...
    :param actuated_dynamics: ...
    (:param dynamics_fn: Unbounded function ...)
    :param punishment_fn: Unbounded function ...
    :param alter_action_space: ...
    :param transform_action_space_fn ...
    :param gamma: ...
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
            raise ValueError("Either dynamics_fn or unactuated_dynamics have to be specified.")

        self._A, self._b = safe_region.halfspaces
        #self._P = matrix([[0.]],tc='d')
        #self._q = matrix([0.], tc='d')
        self._P = matrix([[1., 0], [0, 1e55]], tc='d')
        self._q = matrix([0, 0], tc='d')

        self._gamma = gamma
        self._safe_region = safe_region

        if not hasattr(self.env, "action_space"):
            warnings.warn("Environment does not have attribute ``action_space``")
        if alter_action_space is not None:
            self.action_space = alter_action_space

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

    def step(self, action) -> GymStepReturn:

        if self._transform_action_space_fn is not None:
            action = self._transform_action_space_fn(action)

        # If discrete, needs safety backup
        # Could also try only constraint

        G = matrix(np.asarray([[np.dot(p, self._actuated_dynamics_fn(self.env)), -1] for p in self._A], dtype=np.double), tc='d')

        if self._unactuated_dynamics_fn is not None:
            h = matrix(np.asarray([-np.dot(self._A[i], self._unactuated_dynamics_fn(self.env))
                                   - np.dot(self._A[i], self._actuated_dynamics_fn(self.env)) * action
                                   + (1 - self._gamma) * np.dot(self._A[i], self.env.state) +  # TODO
                                   self._gamma * self._b[i] for i in range(len(self._A))],
                                  dtype=np.double), tc='d')
        else:
            h = matrix(np.asarray([- np.dot(self._A[i], self._dynamics_fn(self.env, action))
                                   + (1 - self._gamma) * np.dot(self._A[i], self.env.state) +  # TODO
                                   self._gamma * self._b[i] for i in range(len(self._A))],
                                  dtype=np.double), tc='d')

        # G = matrix(
        #     np.asarray([[np.dot(p, self._actuated_dynamics_fn(self.env))] for p in self._A], dtype=np.double),
        #     tc='d')
        # if self._unactuated_dynamics_fn is not None:
        #     h = matrix(np.asarray([-np.dot(self._A[i], self._unactuated_dynamics_fn(self.env))
        #                            - np.dot(self._A[i], self._actuated_dynamics_fn(self.env)) * action
        #                            + (1 - self._gamma) * np.dot(self._A[i], self.env.state) +  # TODO
        #                            self._gamma * self._b[i] for i in range(len(self._A))],
        #                           dtype=np.double), tc='d')
        # else:
        #     h = matrix(np.asarray([- np.dot(self._A[i], self._dynamics_fn(self.env, action))
        #                            + (1 - self._gamma) * np.dot(self._A[i], self.env.state) +  # TODO
        #                            self._gamma * self._b[i] for i in range(len(self._A))],
        #                           dtype=np.double), tc='d')


        sol = solvers.qp(self._P, self._q, G, h)

        #TODO: actions if not just one action
        #action_bar = sol['x'][:-1]
        action_bar = sol['x'][0]
        #print(action, sol['x'][0], sol['x'][1])


        obs, reward, done, info = self.env.step(action + action_bar)

        if self._punishment_fn is not None:
            punishment = self._punishment_fn(self.env, self._safe_region, action, action_bar)
            info["cbf"] = {"action": action,
                           "action_bar": action_bar,
                           "reward": reward,
                           "punishment": punishment}
            reward += punishment
        else:
            info["cbf"] = {"action": action,
                           "action_bar": action_bar,
                           "reward": reward,
                           "punishment": None}

        return obs, reward, done, info
