import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback

class PendulumTrainCallback(BaseCallback):

    """
    Extends logged values within Tensorboard.
    #TODO: HandleVecEnvs
    """

    def __init__(self, verbose=0):
        super(PendulumTrainCallback, self).__init__(verbose)

        # TODO: Could (Maybe) directly via Monitor (guarantee that top wrapper?) / check VecEnvs
        # Manually update timesteps (self.model.timesteps refers to the trained model)
        # Note: self.num_timesteps not set since on_step only called via .learn
        self.num_steps = 0

        self.safety_intervention = 0

        #self.roa = self.locals
        self.safe_episode = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.roa = self.training_env.get_attr('roa')[0] #[0] due to VecEnvs

    def _on_step(self) -> bool:
        """

        Notes
        -------
        Does not capture initial step (called after step).

        Returns
        -------
        If the callback returns False, training is aborted early.
        """

        # TODO: Update episode length if nothing else needed / use infos?
        self.num_steps += 1

        # Note: Envs while learning are always wrapped in VecEnvs (i.e. for a single env a DummyVecEnv is used) (.item())
        infos = self.locals.get('infos')[0]
        if "episode" in infos.keys():

            # Log episode reward (SB3 only tracks ep_rew_mean with ep_info_buffer) using updated locals (SB3's intended way)
            # Note: ep_info_buffer (update in BaseAlgorithm) is set after callback evaluation (and restricted due to size)

            self.logger.record('main/episode_reward', infos['episode']['r'])
            self.logger.record('main/episode_length', infos['episode']['l'])
            # self.logger.record('main/episode_time',infos['episode']['t'], exclude='tensorboard')

            self.logger.record('main/no_violation', self.safe_episode)
            self.logger.record('main/safety_intervention', self.safety_intervention / infos['episode']['l'])
            self.safe_episode = True
            self.num_shield = 0

            self.logger.dump(step=self.num_steps)

        # Only get_attr not __getattr_
        if self.training_env.get_attr('state')[0] not in self.roa: #[0][0] not necessary
            self.safe_episode = False

        if 's' in infos.keys():
            self.safety_intervention += infos['s']

        if 'b' in infos.keys():
            self.safety_intervention += infos['b']

        if 'm' in infos.keys():
            self.safety_intervention += infos['m']


        return True