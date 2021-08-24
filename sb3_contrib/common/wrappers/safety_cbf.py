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
    (:param safe_action_fn: Unbounded function ...)
    :param punishment_fn: Unbounded function ...
    :param alter_action_space: ...
    :param transform_action_space_fn ...
    :param gamma: ...
    """

    def __init__(self,
                 env: gym.Env,
                 safe_region: SafeRegion,
                 unactuated_dynamics: Union[Callable[[gym.Env], np.ndarray], str],
                 actuated_dynamics: Union[Callable[[gym.Env, Union[int, float, np.ndarray]], np.ndarray], str],
                 # dynamics_fn: Union[Callable[[gym.Env, Union[int, float, np.ndarray]], np.ndarray], str],
                 # safe_action_fn: Union[Callable[[gym.Env, SafeRegion, Union[int, float, np.ndarray]], Union[int, float, np.ndarray]], str],
                 punishment_fn: Optional[Union[Callable[[gym.Env, SafeRegion, Union[int, float, np.ndarray],
                                                         Union[int, float, np.ndarray]], float], str]] = None,
                 alter_action_space: Optional[gym.Space] = None,
                 transform_action_space_fn: Optional[Union[Callable, str]] = None,
                 gamma: float = .5):

        super().__init__(env)
        self._A, self._b = safe_region.halfspaces
        self._P = matrix([[1., 0], [0, 1e25]], tc='d')
        self._q = matrix([0, 0], tc='d')

        self._gamma = gamma
        self._safe_region = safe_region

        if not hasattr(self.env, "action_space"):
            warnings.warn("Environment does not have attribute ``action_space``")
        if alter_action_space is not None:
            self.action_space = alter_action_space

        if isinstance(unactuated_dynamics, str):
            fn = getattr(self.env, unactuated_dynamics)
            if not callable(fn):
                raise ValueError(f"Attribute {fn} is not a method")
            self._unactuated_dynamics = fn
        else:
            self._unactuated_dynamics = unactuated_dynamics

        if isinstance(actuated_dynamics, str):
            fn = getattr(self.env, actuated_dynamics)
            if not callable(fn):
                raise ValueError(f"Attribute {fn} is not a method")
            self._actuated_dynamics = fn
        else:
            self._actuated_dynamics = actuated_dynamics

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

        G = matrix(np.asarray([[np.dot(p, self._actuated_dynamics(self.env, action)), -1] for p in self._A], dtype=np.double), tc='d')
        h = matrix(np.asarray([-np.dot(self._A[i], self._unactuated_dynamics(self.env))
                               - np.dot(self._A[i], self._actuated_dynamics(self.env, action))
                               + (1 - self._gamma) * np.dot(self._A[i], self.env.state) +  # TODO
                               self._gamma * self._b[i] for i in range(len(self.A))],
                              dtype=np.double), tc='d')

        sol = solvers.qp(self._P, self._q, G, h)
        action_bar = sol['x'][:-1]

        obs, reward, done, info = self.env.step(action + action_bar)

        info["cbf"] = {"action": action, "action_bar": action_bar}

        if self._punishment_fn is not None:
            reward += self._punishment_fn(self.env, self._safe_region, action, action_bar)

        return obs, reward, done, info
