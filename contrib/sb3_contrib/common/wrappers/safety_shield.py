from typing import Union, Callable, Optional

import gym, warnings
import numpy as np
from sb3_contrib.common.safety.safe_region import SafeRegion
from stable_baselines3.common.type_aliases import GymStepReturn


class SafetyShield(gym.Wrapper):

    """
    Safety wrapper that uses post-posed shielding to stay inside a safe region
    Inspired by https://arxiv.org/pdf/1708.08611.pdf

    @param env: Gym environment to be wrapped
    @param safe_region: Safe region instance
    @param dynamics_fn: Dynamics function of the variables that need to be rendered safe
    @param safe_action_fn: Verified fail safe action function
    @param punishment_fn: Optional reward punishment function
    @param alter_action_space: Alternative gym action space
    @param transform_action_space_fn: Action space transformation function

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

        # Alter action space if necessary
        if alter_action_space is not None:
            self.action_space = alter_action_space

        # Fetch unbounded functions
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

        if alter_action_space is not None:
            if self._transform_action_space_fn is None:
                raise ValueError("transform_action_space_fn is None")

    def step(self, action) -> GymStepReturn:

        # Transform action if necessary
        if self._transform_action_space_fn is not None:
            action = self._transform_action_space_fn(action)

        if not self._dynamics_fn(self.env, action) in self._safe_region:

            # Fallback to verified fail-safe action
            safe_action = self._safe_action_fn(self.env, self._safe_region, action)
            obs, reward, done, info = self.env.step(safe_action)

            # Optional reward punishment
            if self._punishment_fn is not None:
                punishment = self._punishment_fn(self.env, self._safe_region, action, safe_action)
                info["shield"] = {"action_rl": action,
                                  "safe_action": safe_action,
                                  "reward": reward,
                                  "punishment": punishment}
                reward += punishment
            else:
                info["shield"] = {"action_rl": action,
                                  "action_shield": safe_action,
                                  "reward": reward,
                                  "punishment": None}

        else:

            # RL action is safe
            obs, reward, done, info = self.env.step(action)
            info["shield"] = {"action_rl": action,
                              "safe_action": None,
                              "reward": reward,
                              "punishment": None}

        return obs, reward, done, info
