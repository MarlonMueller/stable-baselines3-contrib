import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback

class PendulumRolloutCallback(BaseCallback):

    """
    Extends logged values within Tensorboard.
    #TODO: HandleVecEnvs

    Notes
    ----------
    Used for evaluation (see main) without learning.
    As a result, the passed information is 'info' instead of 'infos'.
    """

    def __init__(self, safe_region, verbose=0):
        super(PendulumRolloutCallback, self).__init__(verbose)

        # TODO: Could (Maybe) directly via Monitor (guarantee that top wrapper?) / check VecEnvs
        # Manually update timesteps (self.model.timesteps refers to the trained model)
        # Note: self.num_timesteps not set since on_step only called via .learn
        # Note: start at -1 for repeated _on_rollout_start calls
        self.num_steps = -1

        #TODO: As in Train
        #from pendulum.mathematical_pendulum.envs.pendulum_region_of_attraction import RegionOfAttraction
        #self.roa = RegionOfAttraction()
        self._safe_region = safe_region



    def _on_rollout_start(self) -> None:
        """

        Notes
        -------
        This callback is used for rolling out episodes after training (see main)
        The usage of _on_rollout_start(self) is redefined compared to its use in BaseAlgorithms ('collecting rollouts').
        As a result, '_on_rollout_start' is called for each new episode rollout.

        """

        self.num_steps += 1

        # See warning in _on_step
        state = self.training_env.get_attr('state')[0]

        # Log initial state
        self.logger.record('main/theta', state[0])
        self.logger.record('main/omega', state[1])
        self.logger.dump(step=self.num_steps)

    def _on_step(self) -> bool:
        """

        Returns
        -------
        If the callback returns False, training is aborted early.
        """

        # Note: Envs are always wrapped in VecEnvs (i.e. for a single env a DummyVecEnv is used) (.item())
        info = self.locals.get('info')[0]

        if "episode" in info.keys():
            # Log episode reward (SB3 only tracks ep_rew_mean with ep_info_buffer) using updated locals (SB3's intended way)
            # Note: ep_info_buffer (update in BaseAlgorithm) is set after callback evaluation (and restricted due to size)
            self.logger.record('main/episode_reward', info['episode']['r'])
            # self.logger.record('main/episode_length', infos['episode']['l'], exclude='tensorboard')
            # self.logger.record('main/episode_time',infos['episode']['t'], exclude='tensorboard')
            self.logger.dump(step=self.num_steps)

        self.num_steps += 1


        # Warning: Assuming VecEnv instance as outermost wrapper.
        # Alternatively, use locals directly, unwrap or store reference to unwrapped env.
        # _get_attr_ not defined for VecEnvs (get_attr) but for Wrappers
        state = self.training_env.get_attr('state')[0]
        #print(state)

        self.logger.record('main/theta', state[0])
        self.logger.record('main/omega', state[1])

        if "mask" in info.keys():
            action_rl = info['mask']["action"]
            reward = info['mask']["reward"]
            mask = info['mask']["last_mask"][:-1]
            self.logger.record("main/masked",np.count_nonzero(mask == 0))
            if info['mask']["action_mask"] is not None:
                self.logger.record("main/lqr",abs(info['mask']["action_mask"]))
            if info['mask']["punishment"] is not None:
                self.logger.record("main/punish",info['mask']["punishment"])
        elif "shield" in info.keys():
            action_rl = info['shield']["action"]
            reward = info['shield']["reward"]
            if info['shield']["action_shield"] is not None:
                self.logger.record("main/correction", abs(action_rl - info['shield']["action_shield"]))
            if info['shield']["punishment"] is not None:
                self.logger.record("main/punish", info['shield']["punishment"])

        elif "cbf" in info.keys():
            action_rl = info['cbf']["action"]
            reward  = info['cbf']["reward"]
            self.logger.record("main/correction",info['cbf']["action_bar"])
            if info['cbf']["punishment"] is not None:
                self.logger.record("main/punish", info['cbf']["punishment"])

        else:
            action_rl = info['standard']["action"]
            reward = info['standard']["reward"]

        self.logger.record('main/actionrl', action_rl)
        self.logger.record('main/reward', reward)

        if state not in self._safe_region:
            self.logger.record('main/violation', True)
            if "shield" in info.keys() and info['shield']["action_shield"] is not None:
                self.logger.record('main/realviolation', False)
            elif "mask" in info.keys() and info['mask']["action_mask"] is not None:
                self.logger.record('main/realviolation', False)
            elif "cbf" in info.keys() and info['cbf']["epsilon"] <= 1e-10:
                self.logger.record('main/realviolation', False)
            else:
                self.logger.record('main/realviolation', True)
        else:
            self.logger.record('main/realviolation', False)
            self.logger.record('main/violation', False)

        self.logger.dump(step=self.num_steps)

        return True




