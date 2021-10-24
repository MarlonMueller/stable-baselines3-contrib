import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback

class PendulumRolloutCallback(BaseCallback):

    """
    Extends the logged values for the inverted pendulum problem by following measurements

    episode_reward: Cumulated reward
    episode_length: Episode length
    episode_time: Measured by class Monitor(gym.Wrapper)

    theta: Angular displacement
    thdot: Angular velocity
    action_rl: Action of the policy
    reward_rl: Reward (excluding reward punishment)

    Safety Constraints
    - safe: True iff state is inside ROA
    - safe_excl_approx: True iff state is inside ROA or fail-safe controller is active

    Safety Correction (see README.md)
    - safety_correction
    - safety_correction_mask_lqr

    punishment: Reward punishment

    Assumes DummyVecEnv instance as outermost wrapper
    Note that 'info' is used instead of 'infos'
    """

    def __init__(self, safe_region, verbose=0):
        super(PendulumRolloutCallback, self).__init__(verbose)
        # Start at -1 for repeated _on_rollout_start calls
        self.num_steps = -1
        self._safe_region = safe_region

    def _on_rollout_start(self) -> None:

        """
        The usage of _on_rollout_start(self) is redefined compared to its use in BaseAlgorithms ('collecting rollouts').
        As a result, '_on_rollout_start' is called for each new episode rollout.
        """

        self.num_steps += 1

        # Log initial state
        state = self.training_env.get_attr('state')[0]
        self.logger.record('main/theta', state[0])
        self.logger.record('main/omega', state[1])
        self.logger.dump(step=self.num_steps)

    def _on_step(self) -> bool:

        info = self.locals.get('info')[0]
        if "episode" in info.keys():

            # SB3 only tracks ep_rew_mean with ep_info_buffer using updated locals
            # ep_info_buffer (update in BaseAlgorithm) is set after callback evaluation (and restricted due to size)
            self.logger.record('main/episode_reward', info['episode']['r'])
            self.logger.record('main/episode_length', info['episode']['l'])
            self.logger.record('main/episode_time',info['episode']['t'])
            self.logger.dump(step=self.num_steps)

        self.num_steps += 1

        # VecEnvs only define get_attr not __getattr__; alternatively, use locals directly
        state = self.training_env.get_attr('state')[0]

        # Log state
        self.logger.record('main/theta', state[0])
        self.logger.record('main/omega', state[1])

        if "mask" in info.keys():
            action_rl = info['mask']["action_rl"]
            reward_rl = info['mask']["reward_rl"]
            mask = info['mask']["last_mask"][:-1]
            self.logger.record("main/safety_correction",np.count_nonzero(mask == 0))
            if info['mask']["safe_action"] is not None:
                self.logger.record("main/safety_correction_mask_lqr", abs(info['mask']["safe_action"]))
            if info['mask']["punishment"] is not None:
                self.logger.record("main/punishment",info['mask']["punishment"])

        elif "shield" in info.keys():
            action_rl = info['shield']["action_rl"]
            reward_rl = info['shield']["reward_rl"]
            if info['shield']["action_shield"] is not None:
                self.logger.record("main/safety_correction", abs(action_rl - info['shield']["safe_action"]))
            if info['shield']["punishment"] is not None:
                self.logger.record("main/punishment", info['shield']["punishment"])

        elif "cbf" in info.keys():
            action_rl = info['cbf']["action_rl"]
            reward_rl = info['cbf']["reward_rl"]
            self.logger.record("main/safety_correction",abs(info['cbf']["compensation"]))
            if info['cbf']["punishment"] is not None:
                self.logger.record("main/punishment", info['cbf']["punishment"])

        else:
            action_rl = info['standard']["action_rl"]
            reward_rl = info['standard']["reward_rl"]

        self.logger.record('main/action_rl', action_rl)
        self.logger.record('main/reward_rl', reward_rl)

        if state not in self._safe_region:
            # State is outside of ROA
            self.logger.record('safe', False)
            # Check whether fail-safe controller is active
            if "shield" in info.keys() and info['shield']["safe_action"] is not None:
                    self.logger.record('safe_excl_approx', True)
            if "mask" in info.keys() and info['mask']["safe_action"] is not None:
                    self.logger.record('safe_excl_approx', True)
            else:
                self.logger.record('safe_excl_approx', False)
            # In case CBFs with a slack variable are used, check for e.g. epsilon >= 1e-20
        else:
            self.logger.record('safe', True)
            self.logger.record('safe_excl_approx', True)

        self.logger.dump(step=self.num_steps)
        return True




