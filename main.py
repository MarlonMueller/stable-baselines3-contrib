import os, sys

from typing import Union, Callable
from stable_baselines3.common.type_aliases import GymStepReturn

import argparse, logging, importlib, time, random
from gym.wrappers import TimeLimit

from callbacks.pendulum_train import PendulumTrainCallback
from callbacks.pendulum_rollout import PendulumRolloutCallback

from stable_baselines3.common.utils import configure_logger
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.env_util import is_wrapped

from sb3_contrib.common.maskable.utils import is_masking_supported
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

import gym
import numpy as np
from torch import nn as nn
from util import remove_tf_logs, rename_tf_events, load_model, save_model, tf_events_to_plot

from sb3_contrib import SafeRegion
from sb3_contrib import MaskableA2C, MaskablePPO
from sb3_contrib import SafetyMask, SafetyCBF, SafetyShield

logger = logging.getLogger(__name__)

#TODO: State
#TODO: New typing, state, more support

def main(**kwargs):
    logger.info(f"kargs {kwargs}")
    module = importlib.import_module('stable_baselines3')

    # Use SB3's mlp policy by default
    policy = MlpPolicy

    # Masking is only supported for A2C and PPO
    if 'safety' in kwargs and kwargs['safety'] == "mask":
        policy = MaskableActorCriticPolicy
        if kwargs['algorithm'] == "A2C":
            base_algorithm = MaskableA2C
        elif kwargs['algorithm'] == "PPO":
            base_algorithm = MaskablePPO
        else:
            raise ValueError(f"No masking support for {kwargs['algorithm']}")
    else:
        base_algorithm = getattr(module, kwargs['algorithm'])

    # Uncomment and configure for debug settings
    # if kwargs['name'] == 'DEBUG':
    #     name = 'DEBUG'
    #     kwargs['total_timesteps'] = 1e3
    #     remove_tf_logs(name + '_1', name + '_E_1')

    name = f"{kwargs['name']}_{kwargs['algorithm']}"

    # Initialize environment
    if 'env' not in kwargs:
        raise ValueError(f"env not in kwargs")
    elif kwargs['env'] not in [env_spec.id for env_spec in gym.envs.registry.all()]:
        raise KeyError(f"Environment {kwargs['env']} is not registered")

    if "init" in kwargs and kwargs["init"] == "equilibrium":
        env = gym.make(kwargs['env'], init="equilibrium")
    else:
        env = gym.make(kwargs['env'], init="random")

    env_spec = env.spec
    # Care VecEnv does not specify __getattr__
    if 'rollout' in kwargs and kwargs['rollout']:
        # SB3 uses VecEnvs which reset the environment directly after done is set to true.
        # As a result, logging each state results in env_spec.max_episode_steps -1 entries.
        # To log env_spec.max_episode_steps this simple modification is used.
        env = TimeLimit(env.unwrapped, max_episode_steps=env_spec.max_episode_steps + 1)

    # Define the precomputed safe region
    theta_roa = 3.092505268377452
    vertices = np.array([
        [-theta_roa, 12.762720155208534],  # LeftUp
        [theta_roa, -5.890486225480862],  # RightUp
        [theta_roa, -12.762720155208534],  # RightLow
        [-theta_roa, 5.890486225480862]  # LeftLow
    ])
    safe_region = SafeRegion(vertices=vertices)

    # Adjust action space
    alter_action_space = gym.spaces.Discrete(21)
    if "action_space" in kwargs and kwargs["action_space"] == "small":
        transform_action_space_fn = lambda a: 0.65 * (a - 10)
    else:
        transform_action_space_fn = lambda a: 3 * (a - 10)

    if 'safety' in kwargs and kwargs['safety'] is not None:

        # Shield wrapper
        if kwargs['safety'] == "shield":

            def safe_action_fn(env: gym.Env, safe_region: SafeRegion, action: float) -> float:
                # Precomputed LQR gain matrix
                gain_matrix = [19.670836678497427, 6.351509533724627]
                # Note that __getattr__ is not implemented in VecEnvs
                return -np.dot(gain_matrix, env.get_attr("state")[0])

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            if "punishment" in kwargs and kwargs["punishment"] == "default":
                def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                  action_rl: Union[int, float, np.ndarray],
                                  safety_correction: Union[int, float, np.ndarray]) -> float:
                    return -abs(action_rl - safety_correction)
            else:
                punishment_fn = None

            env = SafetyShield(
                env=env,
                safe_region=safe_region,
                dynamics_fn=dynamics_fn,
                safe_action_fn=safe_action_fn,
                punishment_fn=punishment_fn,
                transform_action_space_fn=transform_action_space_fn,
                alter_action_space=alter_action_space)


        # CBF wrapper
        elif kwargs['safety'] == "cbf":

            def actuated_dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                return np.array([action * env.dt ** 2, action * env.dt])

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            if "punishment" in kwargs and kwargs["punishment"] == "default":
                def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                  action_rl: Union[int, float, np.ndarray],
                                  safety_correction: Union[int, float, np.ndarray]) -> float:
                    return -abs(safety_correction)
            else:
                punishment_fn = None

            if "gamma" not in kwargs:
                kwargs["gamma"] = .5

            env = SafetyCBF(
                env=env,
                safe_region=safe_region,
                dynamics_fn=dynamics_fn,
                actuated_dynamics_fn=actuated_dynamics_fn,
                # unactuated_dynamics_fn=unactuated_dynamics_fn
                punishment_fn=punishment_fn,
                transform_action_space_fn=transform_action_space_fn,
                alter_action_space=alter_action_space,
                gamma=kwargs["gamma"])

        # Mask wrapper
        elif kwargs['safety'] == "mask":

            def safe_action_fn(env: gym.Env, safe_region: SafeRegion, action: float) -> float:
                # Precomputed LQR gain matrix
                gain_matrix = [19.670836678497427, 6.351509533724627]
                # Note that __getattr__ is not implemented in VecEnvs
                return -np.dot(gain_matrix, env.get_attr("state")[0])

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            if "punishment" in kwargs and kwargs["punishment"] == "default":
                def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                  action_rl: Union[int, float, np.ndarray],
                                  mask: Union[int, float, np.ndarray],
                                  next_mask: Union[int, float, np.ndarray]) -> float:
                    if mask[-1] == 1:
                        return min(0, np.sum(next_mask[:-1]) - np.sum(mask[:-1]), -abs(action_rl))
                    else:
                        return min(0, np.sum(next_mask[:-1]) - np.sum(mask[:-1]))
            else:
                punishment_fn = None

            env = SafetyMask(
                env=env,
                safe_region=safe_region,
                dynamics_fn=dynamics_fn,
                safe_action_fn=safe_action_fn,
                punishment_fn=punishment_fn,
                transform_action_space_fn=transform_action_space_fn,
                alter_action_space=alter_action_space)

    else:
        # Adjust the action space and log infos even if no wrapper is applied
        class ActionInfoWrapper(gym.Wrapper):
            def __init__(self, env, alter_action_space=None,
                         transform_action_space_fn=None):
                super().__init__(env)
                self.action_space = alter_action_space
                self._transform_action_space_fn = transform_action_space_fn

            def step(self, action) -> GymStepReturn:
                action = self._transform_action_space_fn(action)
                obs, reward, done, info = self.env.step(action)
                info["standard"] = {"action": action, "reward": reward}
                return obs, reward, done, info

        env = ActionInfoWrapper(env,
                                transform_action_space_fn=transform_action_space_fn,
                                alter_action_space=alter_action_space)

    # Wrap with Monitor
    if not is_wrapped(env, Monitor):
        env = Monitor(env)

    if 'train' in kwargs and kwargs['train']:

        # VecEnvs not supported
        env = DummyVecEnv([lambda: env])

        if 'iterations' not in kwargs:
            iters = 1
        else:
            iters = kwargs['iterations']

        for iteration in range(iters):

            tensorboard_log = os.getcwd() + '/tensorboard/'
            if "group" in kwargs:
                tensorboard_log += args["group"]

            # Linear learning rate schedule
            # def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
            #     if isinstance(initial_value, str):
            #         initial_value = float(initial_value)
            #     def func(progress_remaining: float) -> float:
            #         """
            #         Progress will decrease from 1 (beginning) to 0
            #         :param progress_remaining: (float)
            #         :return: (float)
            #         """
            #         return progress_remaining * initial_value
            #     return func

            if kwargs['algorithm'] == "PPO":
                if 'flag' in kwargs and (kwargs['flag'] == 6 or kwargs['flag'] == 7 or kwargs['flag'] == 8):
                    model = base_algorithm(policy=policy, env=env, verbose=0, tensorboard_log=tensorboard_log)

                else:
                    # Tuned parameters
                    model = base_algorithm(policy=policy,
                                           env=env,
                                           verbose=0,
                                           tensorboard_log=tensorboard_log,
                                           batch_size=64,
                                           n_steps=2048,
                                           gamma=0.9,
                                           learning_rate=0.0003,
                                           ent_coef=0,
                                           clip_range=0.4,
                                           n_epochs=5,
                                           gae_lambda=0.8,
                                           max_grad_norm=0.3,
                                           vf_coef=0.5,
                                           policy_kwargs=dict(
                                               net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                               activation_fn=nn.Tanh,
                                               ortho_init=True)
                                           )


            elif kwargs['algorithm'] == "A2C":
                if 'flag' in kwargs and kwargs['flag'] <= 2:
                    model = base_algorithm(policy=policy,
                                           env=env,
                                           verbose=0,
                                           tensorboard_log=tensorboard_log)
                else:
                    # Tuned parameters
                    model = base_algorithm(policy=policy,
                                           env=env,
                                           verbose=0,
                                           use_rms_prop=False,
                                           normalize_advantage=True,
                                           tensorboard_log=tensorboard_log,
                                           ent_coef=0.1,
                                           max_grad_norm=0.3,
                                           n_steps=8,
                                           gae_lambda=1.0,
                                           vf_coef=0.6,
                                           gamma=0.9,
                                           learning_rate=0.0015,
                                           use_sde=False,
                                           policy_kwargs=dict(
                                               net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                               activation_fn=nn.ReLU,
                                               ortho_init=True)
                                           )

            callback = CallbackList([PendulumTrainCallback(safe_region=safe_region)])
            model.learn(total_timesteps=kwargs['total_timesteps'],
                        tb_log_name=name,
                        callback=callback)
            # log_interval=log_interval)

            if 'save_model' in kwargs and kwargs['save_model']:
                save_model(kwargs["group"] + "/" + str(iteration + 1), model)

    elif 'rollout' in kwargs and kwargs['rollout']:

        # VecEnvs not supported
        env = DummyVecEnv([lambda: env])

        model = None
        callback = None

        if 'model' in kwargs:
            mode = load_model(kwargs['model'], policy)
            model.set_env(env)

            callback = CallbackList([PendulumRolloutCallback(safe_region=safe_region)])
            _logger = configure_logger(verbose=0, tb_log_name=name,
                                       tensorboard_log=os.getcwd() + '/tensorboard/')
            model.set_logger(logger=_logger)
            callback.init_callback(model=model)

        render = False
        if 'render' in kwargs and kwargs['render']:
            render = True

        env_safe_action = False
        if 'safety' in kwargs and kwargs['safety'] == "env_safe_action":
            env_safe_action = True

        rollout(env, model,
                safe_region=safe_region,
                num_episodes=kwargs['iterations'],
                callback=callback,
                env_safe_action=env_safe_action,
                render=render,
                sleep=kwargs['sleep'])

    if 'env' in locals(): env.close()

    if "group" in kwargs:
        rename_tf_events(kwargs["group"])


def rollout(env, model=None, safe_region=None, num_episodes=1, callback=None, env_safe_action=False, render=False,
            rgb_array=False, sleep=0.1):
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

    # Rollout of Maskable model needs auxiliary mask
    mask = np.ones(22)
    mask[-1] = 0

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
                if rgb_array:
                    frame = env.render(mode='rgb_array')
                    frames.append(frame)
                else:
                    env.render()

            time.sleep(sleep)

            # Use trained model if provided
            if model is not None:

                # TODO: Rollout masking
                action, state = model.predict(obs, state=state)
                # action, state = model.predict(obs, state=state, action_masks=mask)
                action = action[0]

            elif env_safe_action:
                action = env.get_attr('safe_action')[0](env, safe_region, None)

            else:
                action = env.action_space.sample()
                if isinstance(action, np.ndarray):
                    action = action.item()

                if is_masking_supported(env):
                    mask = get_action_masks(env)[0]
                    action = random.choice(np.argwhere(mask == True))[0]

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
    parser.add_argument('-alg', '--algorithm', type=str, default='PPO',
                        help='RL algorithm')
    parser.add_argument('-env', type=str, default='MathPendulum-v0',
                        help='ID of a registered environment')
    parser.add_argument('-steps', '--total_timesteps', type=int, default=10e4,
                        help='Total timesteps to train the model')
    parser.add_argument('-safety', '--safety', type=str, default=None,
                        help="Safety approach \'mask\',\'shield\', \'cbf\' or None")
    parser.add_argument('--flag', type=int, default=-1,
                        help='Flag to specify a default configuration')
    parser.add_argument('--model', type=int, default=0,
                        help='Model to rollout, e.g., model.zip')
    parser.add_argument('--sleep', type=float, default=0.1,
                        help='Sleep time [Sec.] in between steps whilst rolling out')
    parser.add_argument('--save_model', type=bool, default=True, help=
    'Whether to save the model after training or not')
    parser.add_argument('--iterations', type=int, default=1, required=False,
                        help='Multiple training or deployment runs')
    args, unknown = parser.parse_known_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()

    # Register environment
    from gym.envs.registration import register

    register(
        id='MathPendulum-v0',
        max_episode_steps=100,
        entry_point='sb3_contrib.common.envs.pendulum.math_pendulum_env:MathPendulumEnv',
    )

    #########
    # Rollout trained models without safety
    #########

    # args["rollout"] = True
    # args["render"] = True

    # args["safety"] = "shield"
    # args["safety"] = "env_safe_action"
    # args["init"] = "zero"
    # args["action_space"] = "small"

    # if args["flag"] == 0:
    #     for model in range(1, 6):
    #             for it in range(1,6):
    #                 args["name"] = f"PPO/{model}{it}"
    #                 main(**args)
    # elif args["flag"] == 1:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"PPO_SAS/{model}{it}"
    #             args["action_space"] = "small"
    #             main(**args)
    # elif args["flag"] == 2:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"PPO_0/{model}{it}"
    #             args["init"] = "zero"
    #             main(**args)
    # elif args["flag"] == 3:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"MASK/{model}{it}"
    #             main(**args)
    # elif args["flag"] == 4:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"MASK_SAS/{model}{it}"
    #             args["action_space"] = "small"
    #             main(**args)
    # elif args["flag"] == 5:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"MASK_0/{model}{it}"
    #             args["init"] = "zero"
    #             main(**args)
    # elif args["flag"] == 6:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"MASK_PUN/{model}{it}"
    #             main(**args)
    # elif args["flag"] == 7:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"MASK_0_PUN/{model}{it}"
    #             args["init"] = "zero"
    #             main(**args)
    # elif args["flag"] == 8:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"MASK_SAS_PUN/{model}{it}"
    #             args["action_space"] = "small"
    #             main(**args)
    # elif args["flag"] == 9:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"CBF/{model}{it}"
    #             main(**args)
    # elif args["flag"] == 10:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"CBF_SAS/{model}{it}"
    #             args["action_space"] = "small"
    #             main(**args)
    # elif args["flag"] == 11:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"CBF_0/{model}{it}"
    #             args["init"] = "zero"
    #             main(**args)
    # elif args["flag"] == 12:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"CBF_PUN/{model}{it}"
    #             main(**args)
    # elif args["flag"] == 13:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"CBF_0_PUN/{model}{it}"
    #             args["init"] = "zero"
    #             main(**args)
    # elif args["flag"] == 14:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"CBF_SAS_PUN/{model}{it}"
    #             args["action_space"] = "small"
    #             main(**args)
    # elif args["flag"] == 15:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"SHIELD/{model}{it}"
    #             main(**args)
    # elif args["flag"] == 16:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"SHIELD_SAS/{model}{it}"
    #             args["action_space"] = "small"
    #             main(**args)
    # elif args["flag"] == 17:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"SHIELD_0/{model}{it}"
    #             args["init"] = "zero"
    #             main(**args)
    # elif args["flag"] == 18:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"SHIELD_PUN/{model}{it}"
    #             main(**args)
    # elif args["flag"] == 19:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"SHIELD_0_PUN/{model}{it}"
    #             args["init"] = "zero"
    #             main(**args)
    # elif args["flag"] == 20:
    #     for model in range(1, 6):
    #         for it in range(1,6):
    #             args["name"] = f"SHIELD_SAS_PUN/{model}{it}"
    #             args["action_space"] = "small"
    #             main(**args)

    #########
    # Train
    #########

    args["train"] = True
    args["name"] = "train"
    args['iterations'] = 5
    args['total_timesteps'] = 8e4

    if args["flag"] == 0:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_UNTUNED"
    elif args["flag"] == 1:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_UNTUNED_SAS"
        args["action_space"] = "small"
    elif args["flag"] == 2:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_UNTUNED_0"
        args["init"] = "equilibrium"
    elif args["flag"] == 3:
        args["algorithm"] = "A2C"
        args['group'] = "A2C"
    elif args["flag"] == 4:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_SAS"
        args["action_space"] = "small"
    elif args["flag"] == 5:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_0"
        args["init"] = "equilibrium"
    elif args["flag"] == 6:
        args['group'] = "PPO_UNTUNED"
    elif args["flag"] == 7:
        args['group'] = "PPO_UNTUNED_SAS"
        args["action_space"] = "small"
    elif args["flag"] == 8:
        args['group'] = "PPO_UNTUNED_0"
        args["init"] = "equilibrium"
    elif args["flag"] == 9:
        args['group'] = "PPO"
    elif args["flag"] == 10:
        args['group'] = "PPO_SAS"
        args["action_space"] = "small"
    elif args["flag"] == 11:
        args['group'] = "PPO_0"
        args["init"] = "equilibrium"
    elif args["flag"] == 12:
        args['group'] = "MASK"
        args["safety"] = "mask"
    elif args["flag"] == 13:
        args['group'] = "MASK_SAS"
        args["safety"] = "mask"
        args["action_space"] = "small"
    elif args["flag"] == 14:
        args['group'] = "MASK_0"
        args["safety"] = "mask"
        args["init"] = "equilibrium"
    elif args["flag"] == 15:
        args['group'] = "MASK_PUN"
        args["safety"] = "mask"
        args["punishment"] = "punish"
    elif args["flag"] == 16:
        args['group'] = "MASK_SAS_PUN"
        args["safety"] = "mask"
        args["action_space"] = "small"
        args["punishment"] = "punish"
    elif args["flag"] == 17:
        args['group'] = "MASK_0_PUN"
        args["safety"] = "mask"
        args["init"] = "equilibrium"
        args["punishment"] = "punish"
    elif args["flag"] == 18:
        args['group'] = "CBF"
        args["safety"] = "cbf"
    elif args["flag"] == 19:
        args['group'] = "CBF_SAS"
        args["safety"] = "cbf"
        args["action_space"] = "small"
    elif args["flag"] == 20:
        args['group'] = "CBF_0"
        args["safety"] = "cbf"
        args["init"] = "equilibrium"
    elif args["flag"] == 21:
        args['group'] = "CBF_PUN"
        args["safety"] = "cbf"
        args["punishment"] = "punish"
    elif args["flag"] == 22:
        args['group'] = "CBF_SAS_PUN"
        args["safety"] = "cbf"
        args["action_space"] = "small"
        args["punishment"] = "punish"
    elif args["flag"] == 23:
        args['group'] = "CBF_0_PUN"
        args["safety"] = "cbf"
        args["init"] = "equilibrium"
        args["punishment"] = "punish"
    elif args["flag"] == 24:
        args['group'] = "SHIELD"
        args["safety"] = "shield"
    elif args["flag"] == 25:
        args['group'] = "SHIELD_SAS"
        args["safety"] = "shield"
        args["action_space"] = "small"
    elif args["flag"] == 26:
        args['group'] = "SHIELD_0"
        args["safety"] = "shield"
        args["init"] = "equilibrium"
    elif args["flag"] == 27:
        args['group'] = "SHIELD_PUN"
        args["safety"] = "shield"
        args["punishment"] = "punish"
    elif args["flag"] == 28:
        args['group'] = "SHIELD_SAS_PUN"
        args["safety"] = "shield"
        args["action_space"] = "small"
        args["punishment"] = "punish"
    elif args["flag"] == 29:
        args['group'] = "SHIELD_0_PUN"
        args["safety"] = "shield"
        args["init"] = "equilibrium"
        args["punishment"] = "punish"
    elif args["flag"] == 30:
        args['group'] = "CBF_95"
        args["safety"] = "cbf"
        args["gamma"] = 0.95
    elif args["flag"] == 31:
        args['group'] = "CBF_SAS_95"
        args["safety"] = "cbf"
        args["action_space"] = "small"
        args["gamma"] = 0.95
    elif args["flag"] == 32:
        args['group'] = "CBF_0_95"
        args["gamma"] = 0.95
        args["safety"] = "cbf"
        args["init"] = "equilibrium"
    elif args["flag"] == 33:
        args['group'] = "CBF_01"
        args["safety"] = "cbf"
        args["gamma"] = 0.1
    elif args["flag"] == 34:
        args['group'] = "CBF_SAS_01"
        args["safety"] = "cbf"
        args["action_space"] = "small"
        args["gamma"] = 0.1
    elif args["flag"] == 35:
        args['group'] = "CBF_0_01"
        args["gamma"] = 0.1
        args["safety"] = "cbf"
        args["init"] = "equilibrium"
    elif args["flag"] == 36:
        args["punishment"] = "punish"
        args['group'] = "CBF_PUN_01"
        args["safety"] = "cbf"
        args["gamma"] = 0.1
    elif args["flag"] == 37:
        args["punishment"] = "punish"
        args['group'] = "CBF_SAS_PUN_01"
        args["safety"] = "cbf"
        args["action_space"] = "small"
        args["gamma"] = 0.1
    elif args["flag"] == 38:
        args["punishment"] = "punish"
        args['group'] = "CBF_0_PUN_01"
        args["gamma"] = 0.1
        args["safety"] = "cbf"
        args["init"] = "equilibrium"
    main(**args)

    tags = [
        # "main/avg_abs_action_rl",  # ?
        # "main/avg_abs_safety_correction",  #
        # "main/avg_abs_masklqr_correction",
        # "main/avg_abs_thdot",  # ?
        # "main/avg_abs_theta",  # ?
        # "main/avg_safety_measure",  #
        # "main/episode_reward",  #
        # "main/episode_time",  #
        # "main/max_abs_action_rl",  # ??
        # "main/max_abs_safety_correction",  #
        # "main/max_abs_thdot",  # ?
        # "main/max_abs_theta",  # ?
        # "main/max_safety_measure",  # ?
        # "main/no_violation",  #
        # "main/rel_abs_safety_correction",
        # "main/avg_step_punishment",  #
        # "main/avg_step_reward_rl"  # ???
        # "main/theta"
    ]

    # PRELIMINARY
    # dirss = []
    # for alg in ["PPO"]:#"A2C"]: # ["PPO", "A2C"]
    #     args["algorithm"] = alg
    #     for safety in ["no_safety"]:
    #         args["safety"] = safety
    #         for action_space in ["small"]: #",verysmall", "normal", "large"]: #TODO
    #             args["action_space"] = action_space
    #             for init in ["zero", "random"]:
    #                 args["init"] = init
    #                 for reward in ["safety"]:
    #                     args["reward"] = reward
    #                     args["group"] = f"{alg}_{action_space}_{init}"
    #                     dirss.append(args["group"])
    #                     #if not os.path.isdir(os.getcwd() + f"/tensorboard/{args['group']}"):
    #                     #    main(**args)
    #                     print(f"Finished training {args['group']} ...")

    from util import tf_events_to_plot, external_legend_res

    for tag in tags:
        if tag == "main/avg_abs_action_rl":
            y_label = "Absolute action per step"  # \overline{\left(\left|a\\right|\\right)}$"
        elif tag == "main/avg_abs_thdot":
            y_label = "$\mathrm{Mean\ absolute\ } \overline{\left(\left|\dot{\\theta}\\right|\\right)}$"
        elif tag == "main/avg_abs_theta":
            y_label = "$\mathrm{Mean\ absolute\ } \overline{\left(\left|\\theta\\right|\\right)}$"
        elif tag == "main/avg_step_reward_rl":
            y_label = "Reward per step excl. $r_{\mathrm{t}}^{\mathrm{PUN}}$"  # %$\overline{r}
        elif tag == "main/reward":
            y_label = "Reward"  # %$\overline{r}
        elif tag == "main/episode_reward":
            y_label = "Episode return"  # ${r_{\mathrm{Episode}}}$
        elif tag == "main/max_safety_measure":
            y_label = "Maximal reward $r_{\mathrm{max}}$"
        elif tag == "main/no_violation":
            y_label = "Safety violation"
        # elif tag == "main/avg_abs_safety_correction":
        #    y_label = "Safety correction $|a_{\mathrm{t}}^{\mathrm{CBF}}|$"
        else:
            y_label = "Angular displacement $\\theta$"
            # y_label = "$|\min(0,m_{\mathrm{t}}^0-m_{\mathrm{t}+1}^0,-|a_{\mathrm{t}}^{\mathrm{VER}}|)|$"

        dirsss = [
            #     ["A2C_UNTUNED_SAS", "A2C_UNTUNED", "A2C_UNTUNED_0"],
            #     ["A2C_SAS", "A2C", "A2C_0"],
            #     ["PPO_UNTUNED_SAS", "PPO_UNTUNED", "PPO_UNTUNED_0"],
            #     ["PPO_SAS", "PPO", "PPO_0"],
            #     ["SHIELD_SAS", "SHIELD", "SHIELD_0"],
            #     ["SHIELD_SAS_PUN", "SHIELD_PUN", "SHIELD_0_PUN"],
            #     ["CBF_SAS", "CBF", "CBF_0"],
            #     ["CBF_SAS_PUN", "CBF_PUN", "CBF_0_PUN"],
            #     ["CBF_SAS_95", "CBF_95", "CBF_0_95"],
            #     ["CBF_SAS_95", "CBF_SAS", "CBF_SAS_01"],
            #     ["MASK_SAS", "MASK", "MASK_0"],
            #     ["MASK_SAS_PUN", "MASK_PUN", "MASK_0_PUN"],
        ]

        # for i, dirss in enumerate(dirsss):
        #     tf_events_to_plot(dirss=dirss,  # "standard"
        #                       tags=[tag],
        #                       x_label='Step',
        #                       # x_label='Episode',
        #                       y_label=y_label,
        #                       width=2.5,  # 5   #2.5 -> 2
        #                       height=2.5,  # 2.5
        #                       episode_length=100,
        #                       window_size=11,  # 41
        #                       save_as=f"pdfs/{i}{tag.split('/')[1]}")
