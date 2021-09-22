from typing import Union, Callable, Optional

import gym, warnings
import numpy as np
from sb3_contrib.common.safety.safe_region import SafeRegion
from stable_baselines3.common.type_aliases import GymStepReturn
from sb3_contrib.common.wrappers.action_masker import ActionMasker


# TODO: Fix Callable typing
# state from observations
# TODO: Maybe do not rely on action masker
# TODO: Check if matches action
# TODO: Punishment function
# Contunuous (CBF and Shield should work)
# Check for discrete action space
#Experiment fail safe backup not implemented?

class SafetyMask(gym.Wrapper):
    """

    :param env: Gym environment to be wrapped
    :param safe_region: Safe region object
    :param dynamics_fn: Unbounded function ...
    :param safe_action_fn: Unbounded function ...
    (:param punishment_fn: Unbounded function ...)
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

        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("Action space is not ``gym.spaces.Discrete`` instance")

        #TODO: Check that Discrete

        self._num_actions = self.action_space.n + 1
        self.action_space = gym.spaces.Discrete(self._num_actions)

        def _mask_fn(_: gym.Env) -> np.ndarray:

            mask = np.zeros(self._num_actions)
            for i in range(self._num_actions-1):
                if self._transform_action_space_fn is not None:
                    action = self._transform_action_space_fn(i)
                else:
                    action = i
                if self._dynamics_fn(self.env, action) in self._safe_region:
                    mask[i] = True

            if not mask.any():
                mask[-1] = True

            self._last_mask = mask
            return mask

        self.env = ActionMasker(self.env, action_mask_fn=_mask_fn)

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

        if action == self._num_actions - 1:
            zero_mask = True
        else:
            zero_mask = False

        if self._transform_action_space_fn is not None:
            action = self._transform_action_space_fn(action)

        if zero_mask:
            #TODO: Explain None
            action_mask = self._safe_action_fn(self.env, self._safe_region, None)
            obs, reward, done, info = self.env.step(action_mask)

            #TODO: Update Notebook info returns
            #TODO: Explain args punishment in notebook
            if self._punishment_fn is not None:
                punishment = self._punishment_fn(self.env, self._safe_region, action_mask, self._last_mask)
                info["mask"] = {"action": 0,
                                "mask": self._last_mask,
                                "action_mask": action_mask,
                                "reward": reward,
                                "punishment": punishment,
                                }
                reward += punishment
            else:
                info["mask"] = {"action": 0,
                                "mask": self._last_mask,
                                "action_mask": action_mask,
                                "reward": reward,
                                "punishment": None
                                }
        else:
            obs, reward, done, info = self.env.step(action)
            if self._punishment_fn is not None:
                punishment = self._punishment_fn(self.env, self._safe_region, action, self._last_mask)
                info["mask"] = {"action": action,
                                "mask": self._last_mask,
                                "action_mask": None,
                                "reward": reward,
                                "punishment": punishment}
                reward += punishment
            else:
                info["mask"] = {"action": action,
                                "mask": self._last_mask,
                                "action_mask": None,
                                "reward": reward,
                                "punishment": None
                                }

        # Pot. Mask Punishment
        return obs, reward, done, info
