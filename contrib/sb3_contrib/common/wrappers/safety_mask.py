from typing import Union, Callable, Optional

import numpy as np
import gym, warnings
from sb3_contrib.common.safety.safe_region import SafeRegion
from stable_baselines3.common.type_aliases import GymStepReturn
from sb3_contrib.common.wrappers.action_masker import ActionMasker


class SafetyMask(gym.Wrapper):

    """
    Safety wrapper that uses action masking to stay inside a safe region
    Inspired by http://mediatum.ub.tum.de/doc/1548735/256213.pdf
    Background on Invalid Action Masking: https://arxiv.org/abs/2006.14171

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

        # Action masking only supports discrete action spaces
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("Action space is not ``gym.spaces.Discrete`` instance")

        # Extend action space with auxiliary action
        self._num_actions = self.action_space.n + 1
        self.action_space = gym.spaces.Discrete(self._num_actions)

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
            self._safe_action_fn = safe_action_fn

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

        # Apply Action Masker (expects the action mask)
        def _mask_fn(_: gym.Env) -> np.ndarray:
            return self._mask
        self.env = ActionMasker(self.env, action_mask_fn=_mask_fn)


    def compute_mask(self):
        """
        Computes the action mask.
        Safe actions are set to True.
        If no safe action exists, the last auxiliary action is set to True.

        @return: action mask
        """
        mask = np.zeros(self._num_actions)
        for i in range(self._num_actions - 1):
            if self._transform_action_space_fn is not None:
                action = self._transform_action_space_fn(i)
            else:
                action = i
            if self._dynamics_fn(self.env, action) in self._safe_region:
                mask[i] = True
        if not mask.any():
            mask[-1] = True
        return mask

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._mask = self.compute_mask()
        return obs

    def step(self, action) -> GymStepReturn:

        # Check if the auxiliary action is used (action_mask=[0,...,0,1])
        if action == self._num_actions - 1:
            zero_mask = True
        else:
            zero_mask = False

        # Transform action if necessary
        if self._transform_action_space_fn is not None:
            action = self._transform_action_space_fn(action)

        if zero_mask:

            # Fallback to verified fail-safe action
            safe_action = self._safe_action_fn(self.env, self._safe_region, None)
            obs, reward, done, info = self.env.step(safe_action)

            # Precompute next mask
            next_mask = self.compute_mask()

            if self._punishment_fn is not None:
                punishment = self._punishment_fn(self.env, self._safe_region, safe_action, self._mask, next_mask)
                info["mask"] = {"action_rl": 0,
                                "last_mask": self._mask,
                                "next_mask": next_mask,
                                "safe_action": safe_action,
                                "reward_rl": reward,
                                "punishment": punishment,
                                }
                reward += punishment
            else:
                info["mask"] = {"action_rl": 0,
                                "last_mask": self._mask,
                                "next_mask": next_mask,
                                "safe_action": safe_action,
                                "reward_rl": reward,
                                "punishment": None
                                }
        else:

            # RL action is safe
            obs, reward, done, info = self.env.step(action)

            # Precompute next mask
            next_mask = self.compute_mask()

            #Optional reward punishment
            if self._punishment_fn is not None:
                punishment = self._punishment_fn(self.env, self._safe_region, action, self._mask, next_mask)
                info["mask"] = {"action_rl": action,
                                "last_mask": self._mask,
                                "next_mask": next_mask,
                                "safe_action": None,
                                "reward_rl": reward,
                                "punishment": punishment}
                reward += punishment
            else:
                info["mask"] = {"action_rl": action,
                                "last_mask": self._mask,
                                "next_mask": next_mask,
                                "safe_action": None,
                                "reward_rl": reward,
                                "punishment": None
                                }

        self._mask = next_mask
        return obs, reward, done, info
