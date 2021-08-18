from sb3_contrib.common.envs.invalid_actions_env import InvalidActionEnv

from sb3_contrib.common.envs.pendulum.pendulum_env import PendulumEnv
from gym.envs.registration import register
register(
    id='Pendulum-v0',
    max_episode_steps=100,
    entry_point='sb3_contrib.common.envs.pendulum.pendulum_env:PendulumEnv',
    #kwargs={'.' : .}
)

