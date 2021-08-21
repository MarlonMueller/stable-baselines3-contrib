from typing import Optional, Union, Callable

import gym
import numpy as np

from stable_baselines3.common.type_aliases import GymStepReturn
from sb3_contrib.common.safety.safe_region import SafeRegion

# TODO: Fix Callable typing
# state from observations

class SafetyShield(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 safe_region: Union[SafeRegion, np.ndarray],
                 is_safe_action_fn: Union[str, Callable[[gym.Env, SafeRegion, float], bool]],
                 safe_action_fn: Union[str, Callable[[gym.Env, SafeRegion], np.ndarray]],
                 punishment_fn: Optional[Union[str, Callable[[gym.Env, SafeRegion, float, float], float]]] = None):

        super(SafetyShield, self).__init__(env)
        #print(env.action_space)

        # TODO VecEnv
        if isinstance(is_safe_action_fn, str):
            found_method = getattr(self.env, is_safe_action_fn)
            if not callable(found_method):
                raise ValueError(f"Environment attribute {is_safe_action_fn} is not a method")
            self._is_safe_action_fn = found_method

        else:
            # self.env could be wrapped, if defined in env is useless
            self._is_safe_action_fn = is_safe_action_fn

        if isinstance(safe_action_fn, str):
            found_method = getattr(self.env, safe_action_fn)
            if not callable(found_method):
                raise ValueError(f"Environment attribute {safe_action_fn} is not a method")
            self._safe_action_fn = found_method

        else:
            self._safe_action_fn = safe_action_fn

        if punishment_fn is not None:
            if isinstance(punishment_fn, str):
                found_method = getattr(self.env, punishment_fn)
                if not callable(found_method):
                    raise ValueError(f"Environment attribute {punishment_fn} is not a method")
                self._punishment_fn = found_method

            else:
                self._punishment_fn = punishment_fn
        else:
            self._punishment_fn = None

        self._safe_region = safe_region

    #TODO: Auch für int wenn safe_action auch int?
    def step(self, action: Union[float, np.ndarray]) -> GymStepReturn:


        if not self._is_safe_action_fn(self.env, self._safe_region, action):

            action_shield = self._safe_action_fn(self.env, self._safe_region)

            obs, reward, done, info = self.env.step(action_shield)
            info['s'] = abs(action_shield - action)

            if self._punishment_fn is not None:
                reward += self._punishment_fn(self.env, self._safe_region, action, action_shield)

        else:
            obs, reward, done, info = self.env.step(action)
            info['s'] = 0.

        return obs, reward, done, info






