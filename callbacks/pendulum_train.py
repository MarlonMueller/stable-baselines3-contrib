import warnings
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class PendulumTrainCallback(BaseCallback):

    """
    Extends the logged values for the inverted pendulum problem by following measurements

        episode_reward: Cumulated reward
        episode_length: Episode length
        episode_time: Measured by class Monitor(gym.Wrapper)

        Average and maximal absolute angular velocity/displacement throughout the episode
        - avg_abs_theta
        - avg_abs_thdot
        - max_abs_theta
        - max_abs_thdot

        Average and maximal absolute action of the policy throughout the episode
        - avg_abs_action_rl
        - max_abs_action_rl

        avg_reward_rl: Average reward (excluding reward punishment)

        Safety Constraints
        - safe_episode: True iff ROA is never left
        - safe_episode_excl_approx: True iff ROA is only left whenever the fail-safe controller is active

        Average and maximal absolute safety correction by the wrappers (see README.md)
        - avg_abs_safety_correction
        - max_abs_safety_correction
        - max_abs_safety_correction_mask_lqr
        - avg_abs_safety_correction_mask_lqr

        avg_punishment: Average reward punishment
        rel_abs_safety_correction: Relative absolute safety correction (total_abs_safety_correction/total_abs_action_rl)


    Assumes DummyVecEnv instance as outermost wrapper
    """

    def __init__(self, safe_region, verbose=0):
        super(PendulumTrainCallback, self).__init__(verbose)

        # Manually update timesteps
        # (self.model.timesteps refers to the trained model)
        # (self.num_timesteps is not set since on_step is only called via .learn)
        self.num_steps = 0
        self._safe_region = safe_region
        self._reset()

    def _reset(self):

        self.total_abs_theta = 0
        self.total_abs_thdot = 0
        self.max_abs_theta = 0
        self.max_abs_thdot = 0

        self.safe_episode = True
        self.safe_episode_excl_approx = True

        self.total_abs_action_rl = 0
        self.max_abs_action_rl = 0

        self.max_abs_safety_correction = 0
        self.total_abs_safety_correction = 0

        self.max_abs_safety_correction_mask_lqr = 0
        self.total_abs_safety_correction_mask_lqr = 0

        self.total_safey_measure = 0
        self.max_safety_measure = 0

        self.total_reward_rl = 0
        self.total_punishment = 0


    def _on_step(self) -> bool:

        self.num_steps += 1

        # VecEnvs only define get_attr not __getattr__; alternatively, use locals directly
        infos = self.locals.get('infos')[0]
        state = self.training_env.get_attr('state')[0]

        abs_theta, abs_thdot = abs(state)
        self.total_abs_theta += abs_theta
        self.total_abs_thdot += abs_thdot
        if abs_theta > self.max_abs_theta:
            self.max_abs_theta = abs_theta
        if abs_thdot > self.max_abs_thdot:
            self.max_abs_thdot = abs_thdot

        if state not in self._safe_region:
            # State is outside of ROA
            self.safe_episode = False
            # Check whether fail-safe controller is active
            if "shield" in infos.keys():
                if infos['shield']["safe_action"] is None:
                    self.safe_episode_excl_approx = False
            if "mask" in infos.keys():
                if infos['mask']["safe_action"] is None:
                    self.safe_episode_excl_approx = False
            else:
                self.safe_episode_excl_approx = False
            # In case CBFs with a slack variable are used, check for e.g. epsilon >= 1e-20

        # Safety measure redundant since already captured via reward
        # theta, thdot = state
        # det_12 = 10.62620981660255
        # max_theta = 3.092505268377452
        # safety_measure = max(abs((theta * 3.436116964863835) / det_12),
        #     abs((theta * 9.326603190344699 + thdot * max_theta) / det_12))
        # self.total_safey_measure += safety_measure
        # if safety_measure > self.max_safety_measure:
        #     self.max_safety_measure = safety_measure

        if "shield" in infos.keys():
            infos = infos['shield']
            if infos["safe_action"] is not None:
                action_rl = infos["action_rl"]
                correction = abs(action_rl - infos["safe_action"])
                self.total_abs_safety_correction += correction
                if correction > self.max_abs_safety_correction:
                    self.max_abs_safety_correction = correction
                if infos["punishment"] is not None:
                    self.total_punishment += infos["punishment"]

        elif "cbf" in infos.keys():
            infos = infos['cbf']
            correction = abs(infos["compensation"])
            self.total_abs_safety_correction += correction
            if correction > self.max_abs_safety_correction:
                self.max_abs_safety_correction = correction
            if infos["punishment"] is not None:
                self.total_punishment += infos["punishment"]

        elif "mask" in infos.keys():
            infos = infos['mask']
            mask = infos["last_mask"][:-1]
            correction = np.count_nonzero(mask == 0)
            self.total_abs_safety_correction += correction
            if correction > self.max_abs_safety_correction:
                self.max_abs_safety_correction = correction
            if infos["safe_action"] is not None:
                self.total_abs_safety_correction_mask_lqr += infos["safe_action"]
                if infos["safe_action"] > self.max_abs_safety_correction_mask_lqr:
                    self.max_abs_safety_correction_mask_lqr = infos["safe_action"]
            if infos["punishment"] is not None:
                self.total_punishment += infos["punishment"]

        else:
            infos = infos['standard']

        action_rl = infos["action_rl"]
        self.total_reward_rl += infos["reward_rl"]
        if abs(action_rl) > self.max_abs_action_rl:
            self.max_abs_action_rl = abs(action_rl)
        self.total_abs_action_rl += abs(action_rl)


        if "episode" in infos.keys():

            # SB3 only tracks ep_rew_mean with ep_info_buffer using updated locals
            # ep_info_buffer (update in BaseAlgorithm) is set after callback evaluation (and restricted due to size)

            self.logger.record('main/episode_reward', infos['episode']['r'])
            self.logger.record('main/episode_length', infos['episode']['l'])
            self.logger.record('main/episode_time',infos['episode']['t'])

            self.logger.record('main/avg_abs_theta', self.total_abs_theta / infos['episode']['l'])
            self.logger.record('main/avg_abs_thdot', self.total_abs_thdot / infos['episode']['l'])
            self.logger.record('main/max_abs_theta', self.max_abs_theta)
            self.logger.record('main/max_abs_thdot', self.max_abs_thdot)

            #self.logger.record('main/max_safety_measure_per_step', self.max_safety_measure)
            #self.logger.record('main/avg_safety_measure_per_step', self.total_safey_measure / infos['episode']['l'])

            self.logger.record('main/max_abs_action_rl', self.max_abs_action_rl)
            self.logger.record('main/avg_abs_action_rl', self.total_abs_action_rl / infos['episode']['l'])
            self.logger.record('main/avg_reward_rl', self.total_reward_rl / infos['episode']['l'])

            if 'mask' in infos.keys() or 'shield' in infos.keys() or 'cbf' in infos.keys():

                self.logger.record('main/max_abs_safety_correction',
                                   self.max_abs_safety_correction)
                self.logger.record('main/avg_abs_safety_correction',
                                   self.total_abs_safety_correction / infos['episode']['l'])

                if 'mask' in infos.keys():
                    self.logger.record('main/max_abs_safety_correction_mask_lqr',
                                       self.max_abs_safety_correction_mask_lqr)
                    self.logger.record('main/avg_abs_safety_correction_mask_lqr',
                                       self.total_abs_safety_correction_mask_lqr / infos['episode']['l'])

                self.logger.record('main/avg_punishment', self.total_punishment / infos['episode']['l'])

                if self.total_abs_action_rl == 0:
                    warnings.warn(f"self.total_action_rl: 0 and total_safety_correction: {self.total_abs_safety_correction}")
                    self.logger.record('main/rel_abs_safety_correction', 0)
                else:
                    self.logger.record('main/rel_abs_safety_correction', self.total_abs_safety_correction / self.total_abs_action_rl)


            self.logger.record('main/safe_episode', self.safe_episode)
            self.logger.record('main/safe_episode_excl_approx', self.safe_episode_excl_approx)

            self._reset()
            self.logger.dump(step=self.num_steps)

        return True