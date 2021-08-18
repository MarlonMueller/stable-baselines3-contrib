from typing import Optional, Union, Callable

import gym
import numpy as np

from stable_baselines3.common.type_aliases import GymStepReturn
from sb3_contrib.common.safety.safe_region import SafeRegion

# TODO: Fix Callable typing
# state from observations
#punishment function (e.g. punish based on state)

class SafetyShield(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 safe_region: Union[SafeRegion, np.ndarray],
                 is_safe_action_fn: Union[str, Callable[[gym.Env, float], bool]],
                 safe_action_fn: Union[str, Callable[[gym.Env], np.ndarray]],
                 punishment: Optional[float] = None):

        super(SafetyShield, self).__init__(env)
        self.env_unwrapped = self.unwrapped

        # TODO VecEnv
        if isinstance(is_safe_action_fn, str):
            found_method = getattr(self.env, is_safe_action_fn)
            if not callable(found_method):
                raise ValueError(f"Environment attribute {is_safe_action_fn} is not a method")
            self.env_unwrapped.is_safe_action = found_method

        else:
            # self.env could be wrapped, if defined in env is useless
            self.env.is_safe_action = is_safe_action_fn

        if isinstance(safe_action_fn, str):
            found_method = getattr(self.env, safe_action_fn)
            if not callable(found_method):
                raise ValueError(f"Environment attribute {safe_action_fn} is not a method")
            self.env_unwrapped._safe_action_fn = found_method

        else:
            self.env.is_safe_action = is_safe_action_fn = safe_action_fn

        self._safe_region = safe_region
        self._punishment = punishment

    def step(self, action: Union[float, np.ndarray]) -> GymStepReturn:


        if self.unwrapped._is_safe_action_fn(self._safe_region, action):

            action_shield = self._safe_action_fn(self.env, self._safe_region)

            obs, reward, done, info = self.env.step(action_shield)
            info['s'] = abs(action_shield - action)

            if self._punishment is not None:
                reward = self._punishment

        else:
            obs, reward, done, info = self.env.step(action)
            info['s'] = 0.

        return obs, reward, done, info






