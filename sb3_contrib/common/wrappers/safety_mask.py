from typing import Optional, Union, Callable

import gym
import numpy as np

from stable_baselines3.common.type_aliases import GymStepReturn
from sb3_contrib.common.safety.safe_region import SafeRegion
from sb3_contrib.common.wrappers import ActionMasker

# TODO: Fix Callable typing
# state from observations
# TODO: Maybe do not rely on action masker
#TODO: Check if matches action space

class SafetyMask(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 safe_region: Union[SafeRegion, np.ndarray],
                 safe_mask_fn: Union[str, Callable[[gym.Env, SafeRegion, float], np.ndarray]],
                 safe_action_fn: Union[str, Callable[[gym.Env, SafeRegion], np.ndarray]],
                 punishment_fn: Optional[Union[str, Callable[[gym.Env, SafeRegion, float, float], float]]] = None):

        super(SafetyMask, self).__init__(env)

        # TODO VecEnv
        if isinstance(safe_mask_fn, str):
            found_method = getattr(self.env, safe_mask_fn)
            if not callable(found_method):
                raise ValueError(f"Environment attribute {safe_mask_fn} is not a method")
            self._safe_mask_fn = found_method

        else:
            # self.env could be wrapped, if defined in env is useless
            self._safe_mask_fn = safe_mask_fn

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

        print(self.env.action_space)

        def _mask_fn(env: gym.Env) -> np.ndarray:

            mask = self._safe_mask_fn(env, self._safe_region)

            # Extend discrete action space by one
            # Mask out always except if not any
            # If chosen as action -> choose fail safe (cont.)

            # SafetyFlag: If set allow cont. actions
            # Discrete Wrapper

            if not mask.any():
                pass

            #TODO: Remove
            if not mask.any():
                print("BackUp needed?")

            return mask

        self.env = ActionMasker(self.env, action_mask_fn=_mask_fn)






