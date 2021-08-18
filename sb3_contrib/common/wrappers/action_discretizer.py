from typing import Union, Callable

import numpy as np
from stable_baselines3.common.type_aliases import GymStepReturn

import gym
import warnings

class ActionDiscretizer(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 disc_action_space: gym.Space,
                 transform_fn: Callable[[int], float]):

        super(ActionDiscretizer, self).__init__(env)

        #TODO: Check (if unbounded and) matching
        if not hasattr(self.env, "action_space"):
            warnings.warn("Environment does not have attribute ``action_space``")

        self.action_space = disc_action_space
        self.transform_fn = transform_fn

    def step(self, action: Union[float, np.ndarray]) -> GymStepReturn:
        return self.env.step(self.transform_fn(action))





