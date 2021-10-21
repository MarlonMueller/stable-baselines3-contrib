from typing import Union, Callable, Optional

import gym, warnings
import numpy as np
from sb3_contrib.common.safety.safe_region import SafeRegion
from stable_baselines3.common.type_aliases import GymStepReturn


# TODO: Fix Callable typing, VecEnvs, Works for ints?
# state from observations
# Continuous to discrete via transform action space should work

class SafetyShield(gym.Wrapper):
    """

    :param env: Gym environment to be wrapped
    :param safe_region: Safe region object
    :param dynamics_fn: Unbounded function ...
    :param safe_action_fn: Unbounded function ...
    :param punishment_fn: Unbounded function ...
    :param alter_action_space: ...
    :param transform_action_space_fn ...
    """

    def __init__(self,
                 env: gym.Env,
                 safe_region: SafeRegion,
                 dynamics_fn: Union[Callable[[gym.Env, Union[int, float, np.ndarray]], np.ndarray], str],
                 safe_action_fn: Union[Callable[[gym.Env, SafeRegion, Union[int, float, np.ndarray]], Union[
                     int, float, np.ndarray]], str],
                 punishment_fn: Optional[Union[Callable[[gym.Env, SafeRegion, Union[int, float, np.ndarray],
                                                         Union[int, float, np.ndarray]], float], str]] = None,
                 alter_action_space: Optional[gym.Space] = None,
                 transform_action_space_fn: Optional[Union[Callable, str]] = None):

        super().__init__(env)
        self._safe_region = safe_region

        if not hasattr(self.env, "action_space"):
            warnings.warn("Environment does not have attribute ``action_space``")
        if alter_action_space is not None:
            self.action_space = alter_action_space

        if isinstance(dynamics_fn, str):
            fn = getattr(self.env, dynamics_fn)
            if not callable(fn):
                raise ValueError(f"Attribute {fn} is not a method")
            self._dynamics_fn = fn
        else:
            self._dynamics_fn = dynamics_fn

        if isinstance(safe_action_fn, str):
            fn = getattr(self.env, safe_action_fn)
            if not callable(fn):
                raise ValueError(f"Attribute {fn} is not a method")
            self._safe_action_fn = fn
        else:
            self._safe_action_fn = dynamics_fn

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

        if not self._dynamics_fn(self.env, action) in self._safe_region:

            action_shield = self._safe_action_fn(self.env, self._safe_region, action)
            obs, reward, done, info = self.env.step(action_shield)

            if self._punishment_fn is not None:
                punishment = self._punishment_fn(self.env, self._safe_region, action, action_shield)
                info["shield"] = {"action": action,
                                  "action_shield": action_shield,
                                  "reward": reward,
                                  "punishment": punishment}
                reward += punishment
            else:
                info["shield"] = {"action": action,
                                  "action_shield": action_shield,
                                  "reward": reward,
                                  "punishment": None}

        else:
            obs, reward, done, info = self.env.step(action)
            info["shield"] = {"action": action,
                              "action_shield": None,
                              "reward": reward,
                              "punishment": None}

        return obs, reward, done, info
