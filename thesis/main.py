import os, argparse, logging, importlib, time, random
from typing import Union

from gym.wrappers import TimeLimit
from stable_baselines3.common.utils import configure_logger
from thesis.callbacks.pendulum_train import PendulumTrainCallback
from thesis.callbacks.pendulum_rollout import PendulumRolloutCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from sb3_contrib.common.wrappers import SafetyMask
from sb3_contrib.common.maskable.utils import is_masking_supported
from thesis.util import remove_tf_logs, rename_tf_events, load_model, save_model
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.a2c import A2C
from stable_baselines3 import HER, A2C, PPO, DQN

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.a2c_mask import MaskableA2C

import gym
import numpy as np
from numpy import pi

logger = logging.getLogger(__name__)
#Try different optimizers

def main(**kwargs):

    logger.info(f"kargs {kwargs}")

    module = importlib.import_module('stable_baselines3')

    if 'safety' in kwargs and kwargs['safety'] == "mask":
        if kwargs['algorithm'] == "A2C":
            base_algorithm = MaskableA2C
        elif kwargs['algorithm'] == "PPO":
            base_algorithm = MaskablePPO
        else:
            raise ValueError(f"No masking support for {kwargs['algorithm']}")
    else:
        base_algorithm = getattr(module, kwargs['algorithm'])

    if kwargs['name'] == 'DEBUG':
        name = 'DEBUG'
        kwargs['total_timesteps'] = 1e3
        remove_tf_logs(name + '_1', name + '_E_1')
    else:
        name = f"{kwargs['name']}_{kwargs['algorithm']}"

    from gym.envs.registration import register
    register(
        id='MathPendulum-v0',
        max_episode_steps=100,
        entry_point='sb3_contrib.common.envs.pendulum.math_pendulum_env:MathPendulumEnv',
    )

    # Initialize environment
    if kwargs['env_id'] not in [env_spec.id for env_spec in gym.envs.registry.all()]:
         KeyError(f"Environment {kwargs['env_id']} is not registered")
    env = gym.make(kwargs['env_id'])

    #TODO
    #if 'safety' not in kwargs:
    #    env.action_space = gym.Discrete(15)

    # If not wrapped in VecEnv (__getattr__ set)
    env_spec = env.spec
    # TODO: Not in Notebook currently
    if 'rollout' in kwargs and kwargs['rollout']:
        # SB3 uses VecEnvs which reset the environment directly after done is set to true.
        # As a result, logging each state results in env_spec.max_episode_steps -1 entries.
        # To log env_spec.max_episode_steps this simple modification is used.
        env = TimeLimit(env.unwrapped, max_episode_steps=env_spec.max_episode_steps + 1)

    # Define safe regions
    from sb3_contrib.common.safety.safe_region import SafeRegion
    #TODO: PendulumSafeRegion
    max_thdot = 5.890486225480862
    vertices = np.array([
        [-pi, max_thdot],  # LeftUp
        [-0.785398163397448, max_thdot],  # RightUp
        [pi, -max_thdot],  # RightLow
        [0.785398163397448, -max_thdot]  # LeftLow
    ])
    safe_region = SafeRegion(vertices=vertices)

    if 'safety' in kwargs and kwargs['safety'] is not None:

        if kwargs['safety'] == "shield":
            from sb3_contrib.common.wrappers import SafetyShield

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                              action: Union[int, float, np.ndarray],
                              action_shield: Union[int, float, np.ndarray]) -> float:
                return -abs(action - action_shield)

            # Wrap with SafetyShield
            env = SafetyShield(
                env=env,
                safe_region=safe_region,
                dynamics_fn=dynamics_fn,
                safe_action_fn="safe_action",  # Method already in env (LQR controller)
                punishment_fn=punishment_fn,
                transform_action_space_fn=lambda a: 2 * (a - 7),
                alter_action_space=gym.spaces.Discrete(15))

        elif kwargs['safety'] == "cbf":
            from sb3_contrib.common.wrappers import SafetyCBF

            def punishment_fn(env: gym.Env, safe_region:SafeRegion, action: float, action_bar: float) -> float:
                return -abs(action - action_bar)

            #TODO, f und g as other methods in env?
            #TODO Erklärung Problem
            #TODO Liste Thesis

            env = SafetyCBF(
                env=env,
                safe_region=safe_region,
                punishment_fn=None
            )

        elif kwargs['safety'] == "mask":
            from sb3_contrib.common.wrappers import SafetyMask

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                              action: Union[int, float, np.ndarray],
                              action_mask: Union[int, float, np.ndarray]) -> float:
                return -(action_mask ** 8)

            # Wrap with SafetyMask
            env = SafetyMask(
                env=env,
                safe_region=safe_region,
                dynamics_fn=dynamics_fn,
                safe_action_fn="safe_action",  # Method already in env (LQR controller)
                punishment_fn=punishment_fn,
                transform_action_space_fn=lambda a: 2 * (a - 7),
                alter_action_space=gym.spaces.Discrete(15))



    if not is_wrapped(env, Monitor):
        env = Monitor(env)

    if 'train' in kwargs and kwargs['train']:

        env = DummyVecEnv([lambda: env])

        if 'safety' in kwargs and kwargs['safety'] == "mask":
            from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
            model = base_algorithm(MaskableActorCriticPolicy, env, verbose=0, tensorboard_log=os.getcwd() + '/tensorboard/')
        else:
            model = base_algorithm(MlpPolicy, env, verbose=0, tensorboard_log=os.getcwd() + '/tensorboard/')

        #TODO: Remove
        #from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        #model = base_algorithm(MaskableActorCriticPolicy, env, verbose=0, tensorboard_log=os.getcwd() + '/tensorboard/')

        callback = CallbackList([PendulumTrainCallback(safe_region=safe_region)])

        model.learn(total_timesteps=kwargs['total_timesteps'],
                    tb_log_name=name,
                    callback=callback)
                    # log_interval=log_interval)

        save_model(name, model)

    elif 'rollout' in kwargs and kwargs['rollout']:

        env = DummyVecEnv([lambda: env])

        model = None
        callback = None

        if name != "DEBUG":

            model = load_model(name + '.zip', base_algorithm)
            model.set_env(env)


            callback = CallbackList([PendulumRolloutCallback(safe_region=safe_region)])

            # TODO: Not needed for training(?)
            _logger = configure_logger(verbose=0, tb_log_name=name + '_E', tensorboard_log=os.getcwd() + '/tensorboard/')
            model.set_logger(logger=_logger)

            callback.init_callback(model=model)


        render = False
        if 'render' in kwargs and kwargs['render']:
            render = True

        env_safe_action = False
        if 'safety' in kwargs and kwargs['safety'] == "env_safe_action":
            env_safe_action = True

        #rollout(env, model, safe_region=safe_region, num_episodes=1, callback=callback, env_safe_action=env_safe_action, render=render, sleep=.05)
        rollout(env, model, safe_region=safe_region, num_episodes=1, callback=callback, env_safe_action=env_safe_action,
                render=render, sleep=.1)

    if 'env' in locals(): env.close()
    rename_tf_events()


def rollout(env, model=None, safe_region=None, num_episodes=1, callback=None, env_safe_action=False, render=False, rgb_array=False, sleep=0.1):


    is_vec_env = isinstance(env, VecEnv)
    if is_vec_env:
        if env.num_envs != 1:
            logger.warning(f"You must pass only one environment when using this function")
        is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)

    if not is_monitor_wrapped:
        logger.warning(f"Evaluation environment is not wrapped with a ``Monitor`` wrapper.")

    frames = []
    reset = True
    for episode in range(num_episodes):

        done, state = False, None

        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or reset:
            obs = env.reset()
            reset = False

        # Give access to local variables
        if callback:
            callback.update_locals(locals())
            callback.on_rollout_start()

        while not done:

            if render:
                # Does not render last step
                if rgb_array:
                    frame = env.render(mode='rgb_array')
                    frames.append(frame)
                else:
                    env.render()

            time.sleep(sleep)

            if model is not None:

                action, state = model.predict(obs, state=state) #deterministic=deterministic
                action = action[0] #Action is dict

                #TODO: Check if masking used - also rollout masking without model!
                #if use_masking:
                #    action_masks = get_action_masks(env)
                #    actions, state = model.predict(obs,state=state, action_masks=action_masks)


            elif env_safe_action:
                #TODO: Fix/Easier? / Could check for callable etc.
                action = env.get_attr('safe_action')[0](env, safe_region)

            else:
                #TODO: Sample is [] for box and otherwise not?
                action = env.action_space.sample()
                if isinstance(action, np.ndarray):
                    action = action.item()

                if is_masking_supported(env):
                    mask = get_action_masks(env)[0]
                    action = random.choice(np.argwhere(mask==True))[0]

            obs, reward, done, info = env.step([action])

            if render:
                # Prevent render after reset
                if not is_vec_env or is_vec_env and not done:
                    env.render()

            # Do not plot reset for last episode
            if callback and not (done and episode == num_episodes - 1):
                # Give access to local variables
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

    time.sleep(sleep)
    env.close()

    if rgb_array:
        return frames


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, default='PPO', required=False,
                        help='RL algorithm')
    parser.add_argument('-e', '--env_id', type=str, default='MathPendulum-v0', required=False,
                        help='ID of a registered environment')
    parser.add_argument('-t', '--total_timesteps', type=int, default=20e4, required=False,  # 400
                        help='Total timesteps to train model') #TODO: Episodes
    parser.add_argument('-n', '--name', type=str, default='DEBUG', required=False,
                        help='Base name for generated data')
    parser.add_argument('-s', '--safety', type=str, default=None, required=False,
                        help='Safety method')
    args, unknown = parser.parse_known_args()
    return vars(args)

if __name__ == '__main__':
    args = parse_arguments()

    # Debug
    #args['rollout'] = True
    #args['render'] = True
    #args['safety'] = "shield"

    args["train"] = True
    #args["name"]="NoSafety"
    #args["safety"] = "shield"
    #args["name"] = "Shield"
    #args["name"] = "Shield_Punish"
    args["safety"] = "mask"
    args["name"] = "Mask"
    #args["name"] = "Mask_Punish"
    # args["safety"] = "cbf"

    #args['train'] = True
    #args['safety'] = 'mask'
    #args['name'] = 'noSafetyTest'

    #args['rollout'] = True
    #args['render'] = True
    #args['safety'] = 'mask'

    #args['name'] = 'maskTest'

    main(**args)