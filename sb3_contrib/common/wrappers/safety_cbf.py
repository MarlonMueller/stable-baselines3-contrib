from typing import Optional, Union, Callable

import gym
import numpy as np

from cvxopt import matrix
from cvxopt import solvers

from stable_baselines3.common import env_util
from stable_baselines3.common.type_aliases import GymStepReturn
from sb3_contrib.common.safety.safe_region import SafeRegion

# state from observations
class SafetyCBF(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 safe_region: Union[SafeRegion, np.ndarray],
                 f: np.ndarray,
                 g: np.ndarray,
                 gamma: float = .5,
                 punishment_fn: Optional[Union[str, Callable[[gym.Env, SafeRegion, float, float], float]]] = None):

        super(SafetyCBF, self).__init__(env)

        self._f = f
        self._g = g
        self._gamma = gamma

        if punishment_fn is not None:
            if isinstance(punishment_fn, str):
                found_method = getattr(self.env, punishment_fn)
                if not callable(found_method):
                    raise ValueError(f"Environment attribute {punishment_fn} is not a method")
                self._punishment_fn = found_method

            else:
                self._punishment_fn = punishment_fn

        self._safe_region = safe_region

        self._A, self._b = self.env.safe_region.halfspaces
        self._P = matrix([[1., 0], [0, 1e25]], tc='d')
        self._q = matrix([0, 0], tc='d')


    def step(self, action: Union[float, np.ndarray]) -> GymStepReturn:

        G = matrix(np.asarray([[np.dot(p, self._g), -1] for p in self.A], dtype=np.double), tc='d')
        h = matrix(np.asarray([ -np.dot(self.A[i], self._f)
                                -np.dot(self.A[i], self._g) * action
                                +(1 - self.gamma) * np.dot(self.A[i], self.env.state) +
                                self.gamma * self.b[i] for i in range(len(self.A))],
                                dtype=np.double), tc='d')
        sol = solvers.qp(self._P, self._q, G, h)
        action_bar = sol['x'][:-1]

        obs, reward, done, info = self.env.step(action + action_bar)

        if self._punishment_fn is not None:
            reward += self._punishment_fn(self.env, self._safe_region, action, action_bar)

        info['b'] = abs(action_bar - action)

        return obs, reward, done, info






