import numpy as np
from stable_baselines3.common import logger
import warnings
from stable_baselines3.common.callbacks import BaseCallback

class PendulumTrainCallback(BaseCallback):

    """
    Extends logged values within Tensorboard.
    #TODO: HandleVecEnvs
    """

    def __init__(self, safe_region, verbose=0):
        super(PendulumTrainCallback, self).__init__(verbose)

        # TODO: Could (Maybe) directly via Monitor (guarantee that top wrapper?) / check VecEnvs
        # Manually update timesteps (self.model.timesteps refers to the trained model)
        # Note: self.num_timesteps not set since on_step only called via .learn
        self.num_steps = 0

        self.safe_episode = True
        self.total_abs_theta = 0
        self.total_abs_thdot = 0
        self.max_abs_theta = 0
        self.max_abs_thdot = 0

        self.total_abs_action_rl = 0
        self.max_abs_action_rl = 0

        self.total_abs_safety_correction = 0
        self.max_abs_safety_correction = 0

        self._safe_region = safe_region

    def _on_step(self) -> bool:

        # TODO: Update episode length if nothing else needed / use infos?
        self.num_steps += 1

        # Note: Envs while learning are always wrapped in VecEnvs (i.e. for a single env a DummyVecEnv is used) (.item())
        infos = self.locals.get('infos')[0]

        # Only get_attr not __getattr_
        # state = self.training_env.get_attr('state')[0]

        # Torque?
        # Reflected (as well as max - maybe max even more interesting) in reward
        state = self.training_env.get_attr('state')[0]
        abs_theta, abs_thdot = abs(state)
        self.total_abs_theta += abs_theta
        self.total_abs_thdot += abs_thdot
        if abs_theta > self.max_abs_theta:
            self.max_abs_theta = abs_theta
        if abs_thdot > self.max_abs_thdot:
            self.max_abs_thdot = abs_thdot

        # Avg. masked out actions
        # Avg. shield use
        # % of max force used or more?
        # max distance middle
        # max distance border
        # speed / reward
        # max theta/theta dot used / avg

        # TODO: Get total action RL without safety
        # TODO: Maybe check if safety != 0 or not
        # TODO: Masking FailSafeAction

        if state not in self._safe_region:  # [0][0] not necessary
            self.safe_episode = False

        # TODO: Discuss shield evaluation: Difference or total safe action?
        if 'shield' in infos.keys():
            action_rl = infos['shield']["action"]
            if infos['shield']["action_shield"] is not None:
                correction = abs(action_rl - infos['shield']["action_shield"])
                self.total_abs_safety_correction += correction
                if correction > self.max_abs_safety_correction:
                    self.max_abs_safety_correction = correction

            if abs(action_rl) > self.max_abs_action_rl:
                self.max_abs_action_rl = abs(action_rl)
            self.total_abs_action_rl += abs(action_rl)

        if 'cbf' in infos.keys():
            action_rl = infos['cbf']["action"]
            correction = abs(infos['cbf']["action_bar"])
            self.total_abs_safety_correction += correction
            if correction > self.max_abs_safety_correction:
                self.max_abs_safety_correction = correction

            if abs(action_rl) > self.max_abs_action_rl:
                self.max_abs_action_rl = abs(action_rl)
            self.total_abs_action_rl += abs(action_rl)

        if 'mask' in infos.keys():
            # TODO: compare action rl
            action_rl = infos['mask']["action"]

            mask = infos['mask']["mask"][:-1]
            correction = np.count_nonzero(mask == 0)
            self.total_abs_safety_correction += correction
            if correction > self.max_abs_safety_correction:
                self.max_abs_safety_correction = correction
            self.total_abs_action_rl += len(mask)  # TODO

            if abs(action_rl) > self.max_abs_action_rl:
                self.max_abs_action_rl = abs(action_rl)

        if "episode" in infos.keys():

            # Log episode reward (SB3 only tracks ep_rew_mean with ep_info_buffer) using updated locals (SB3's intended way)
            # Note: ep_info_buffer (update in BaseAlgorithm) is set after callback evaluation (and restricted due to size)

            self.logger.record('main/episode_reward', infos['episode']['r'])
            #self.logger.record('main/episode_length', infos['episode']['l'])
            self.logger.record('main/episode_time',infos['episode']['t']) # exclude='tensorboard'

            self.logger.record('main/avg_abs_theta', self.total_abs_theta / infos['episode']['l'])
            self.logger.record('main/avg_abs_thdot', self.total_abs_thdot / infos['episode']['l'])
            self.logger.record('main/max_abs_theta', self.max_abs_theta)
            self.logger.record('main/max_abs_thdot', self.max_abs_thdot)  # TODO: Maybe not abs and draw upper and lower boundary

            if 'mask' in infos.keys() or 'shield' in infos.keys() or 'cbf' in infos.keys():

                self.logger.record('main/avg_abs_action_rl', self.total_abs_action_rl / infos['episode']['l'])
                self.logger.record('main/avg_abs_safety_correction', self.total_abs_safety_correction / infos['episode']['l'])
                self.logger.record('main/max_abs_action_rl', self.max_abs_action_rl)
                self.logger.record('main/max_abs_safety_correction', self.max_abs_safety_correction)

                # TODO: Uncorrected action?
                # self.logger.record('main/total_abs_action_rl', self.total_abs_action_rl)
                # self.logger.record('main/total_abs_safety_correction', self.total_abs_safety_correction)
                if self.total_abs_action_rl == 0:
                    warnings.warn(
                        f"self.total_action_rl: 0 and total_safety_correction: {self.total_abs_safety_correction}")
                    self.logger.record('main/rel_abs_safety_correction', 0)
                else:
                    self.logger.record('main/rel_abs_safety_correction',
                                       self.total_abs_safety_correction / self.total_abs_action_rl)

            # Max rl action not needed (guaranteed to be safe)
            # Same basically for max thdot / theta - but good for visualisation with boundary? -> better than avg?!

            #max! torque

            # Max Theta (Safety - Implicit) + Avg. Theta (learns) - do not use max / is implicit
            # Max Thdot (Safety) + Avg. Thdot (learns) - is safe constraint
            # Torque LQR would excert?
            # Max Torque Overall (not interesting - safe + rl -> result) (learn)  + Avg. Torque (not necessary) !!!TODO: TORQUE != TORQUE LQR; COULD BE LARGER! SEE ACTION SPACE

            # Max ActionRL + Avg. ActionRL #max not interesting? außer außbrecher unerwünscht
            # Max Correction + Avg. Correction #max not interesting? außer außbrecher unerwünscht
            # Rel. Correction / ActionRL
            # Time
            # Safety Boundary
            # Reward

            # Entropie etc.?

            # No safe violation; max thdot/max torque != action rl, max/avg theta (avg redundant - aber trotzde ok)
            #Avg safety correction, Avg action rl -> , Rel. Abs. correction
            # Time, Avg. Total distance to ROA?

            self.logger.record('main/no_violation', self.safe_episode)


            self.safe_episode = True
            self.total_abs_theta = 0
            self.total_abs_thdot = 0
            self.max_abs_theta = 0
            self.max_abs_thdot = 0

            self.total_abs_action_rl = 0
            self.max_abs_action_rl = 0
            self.total_abs_safety_correction = 0
            self.max_abs_safety_correction = 0

            self.logger.dump(step=self.num_steps)

        return True