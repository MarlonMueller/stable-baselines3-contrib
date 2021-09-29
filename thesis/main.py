import os, argparse, logging, importlib, time, random
from typing import Union, Callable

from gym.wrappers import TimeLimit
from stable_baselines3.common.type_aliases import GymStepReturn
from stable_baselines3.common.utils import configure_logger

from thesis.callbacks.pendulum_train import PendulumTrainCallback
from thesis.callbacks.pendulum_rollout import PendulumRolloutCallback

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib.common.wrappers import SafetyMask
from sb3_contrib.common.maskable.utils import is_masking_supported
from thesis.util import remove_tf_logs, rename_tf_events, load_model, save_model, tf_events_to_plot
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.a2c import A2C
from stable_baselines3 import HER, A2C, PPO, DQN

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.a2c_mask import MaskableA2C
from torch import nn as nn

import gym
import numpy as np
from numpy import pi

logger = logging.getLogger(__name__)


# Try different optimizers

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

    # Initialize environment
    if kwargs['env_id'] not in [env_spec.id for env_spec in gym.envs.registry.all()]:
        KeyError(f"Environment {kwargs['env_id']} is not registered")

    # obs = None
    # if "obs" in kwargs and kwargs["obs"]:
    #     obs = True

    if "init" in kwargs and kwargs["init"] == "zero":
        env = gym.make(kwargs['env_id'], init="zero")
    else:
        env = gym.make(kwargs['env_id'])

    # if "init" in kwargs and kwargs["init"] == "random":
    #     if "reward" in kwargs and kwargs["reward"] == "opposing":
    #         env = gym.make(kwargs['env_id'], init="random", reward="opposing")
    #     else:
    #         env = gym.make(kwargs['env_id'], init="random")
    # else:
    #     if "reward" in kwargs and kwargs["reward"] == "opposing":
    #         env = gym.make(kwargs['env_id'], reward="opposing")
    #     else:
    #         env = gym.make(kwargs['env_id'])

    # TODO
    # if 'safety' not in kwargs:
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
    # TODO: PendulumSafeRegion

    theta_roa = 3.092505268377452
    vertices = np.array([
        [-theta_roa, 12.762720155208534],  # LeftUp
        [theta_roa, -5.890486225480862],  # RightUp
        [theta_roa, -12.762720155208534],  # RightLow
        [-theta_roa, 5.890486225480862]  # LeftLow
    ])
    safe_region = SafeRegion(vertices=vertices)

    # if "action_space" in kwargs and kwargs["action_space"] == "large":
    #     transform_action_space_fn = lambda a: 5 * (a - 10) if a <= 9 else 5 * (a - 9)
    #     alter_action_space = gym.spaces.Discrete(20)
    if "action_space" in kwargs and kwargs["action_space"] == "small":
        transform_action_space_fn = lambda a: 0.65 * (a - 10)
        alter_action_space = gym.spaces.Discrete(21)
    else:
        transform_action_space_fn = lambda a: 3 * (a - 10)
        alter_action_space = gym.spaces.Discrete(21)

    if 'safety' in kwargs and kwargs['safety'] is not None:

        if kwargs['safety'] == "shield":
            from sb3_contrib.common.wrappers import SafetyShield

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            # return -abs(action - action_shield) 1:1
            if "punishment" in kwargs:
                if kwargs["punishment"] == "punish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      action_shield: Union[int, float, np.ndarray]) -> float:
                        return -abs(action - action_shield)
                # elif kwargs["punishment"] == "lightpunish":
                #     def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                #                       action: Union[int, float, np.ndarray],
                #                       action_shield: Union[int, float, np.ndarray]) -> float:
                #         return -abs(action - action_shield) * 0.5
                # elif kwargs["punishment"] == "heavypunish":
                #     def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                #                       action: Union[int, float, np.ndarray],
                #                       action_shield: Union[int, float, np.ndarray]) -> float:
                #         return -abs(action - action_shield) * 4
                else:
                    punishment_fn = None
            else:
                punishment_fn = None

            # Wrap with SafetyShield
            env = SafetyShield(
                env=env,
                safe_region=safe_region,
                dynamics_fn=dynamics_fn,
                safe_action_fn="safe_action",  # Method already in env (LQR controller)
                punishment_fn=punishment_fn,
                transform_action_space_fn=transform_action_space_fn,
                alter_action_space=alter_action_space)


        elif kwargs['safety'] == "cbf":
            from sb3_contrib.common.wrappers import SafetyCBF

            def actuated_dynamics_fn(env: gym.Env) -> np.ndarray:
                return np.array([0, env.dt])
                #return np.array([0, (env.dt / (env.m * env.l ** 2))])
                # return np.array([(env.dt ** 2 / (env.m * env.l ** 2)), (env.dt / (env.m * env.l ** 2))])

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            if "punishment" in kwargs:
                if kwargs["punishment"] == "punish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      action_cbf: Union[int, float, np.ndarray]) -> float:
                        return -abs(action_cbf)
                # elif kwargs["punishment"] == "lightpunish":
                #     def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                #                       action: Union[int, float, np.ndarray],
                #                       action_cbf: Union[int, float, np.ndarray]) -> float:
                #         return -abs(action_cbf) * 0.5
                # elif kwargs["punishment"] == "heavypunish":
                #     def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                #                       action: Union[int, float, np.ndarray],
                #                       action_cbf: Union[int, float, np.ndarray]) -> float:
                #         return -abs(action_cbf) * 4
                else:
                    punishment_fn = None
            else:
                punishment_fn = None

            if "gamma" not in kwargs:
                kwargs["gamma"] = .5

            # Wrap with SafetyCBF
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

            # TODO, f und g as other methods in env?
            # TODO Erklärung Problem
            # TODO Liste Thesis



        elif kwargs['safety'] == "mask":
            from sb3_contrib.common.wrappers import SafetyMask

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            # We only care about the mask, fail-safe controller is not in use
            if "punishment" in kwargs:
                if kwargs["punishment"] == "punish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      mask: Union[int, float, np.ndarray],
                                      next_mask: Union[int, float, np.ndarray]) -> float:
                        return min(0, np.sum(next_mask[:-1]) - np.sum(mask[:-1]))
                # elif kwargs["punishment"] == "lightpunish":
                #     def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                #                       action: Union[int, float, np.ndarray],
                #                       mask: Union[int, float, np.ndarray]) -> float:
                #         return -(1 - (np.sum(mask)-1) / (len(mask)-1)) * 5
                # elif kwargs["punishment"] == "heavypunish":
                #     def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                #                       action: Union[int, float, np.ndarray],
                #                       mask: Union[int, float, np.ndarray]) -> float:
                #         return -(1 - (np.sum(mask)-1) / (len(mask)-1)) * 40
                else:
                    punishment_fn = None
            else:
                punishment_fn = None

            # Wrap with SafetyMask
            env = SafetyMask(
                env=env,
                safe_region=safe_region,
                dynamics_fn=dynamics_fn,
                safe_action_fn="safe_action",  # Method already in env (LQR controller)
                punishment_fn=punishment_fn,
                transform_action_space_fn=transform_action_space_fn,
                alter_action_space=alter_action_space)

    else:
        class ActionInfoWrapper(gym.Wrapper):
            def __init__(self, env, alter_action_space=None,
                         transform_action_space_fn=None):
                super().__init__(env)

                #if alter_action_space is not None:
                self.action_space = alter_action_space

                #if transform_action_space_fn is not None:
                    # if isinstance(transform_action_space_fn, str):
                    #     fn = getattr(self.env, transform_action_space_fn)
                    #     if not callable(fn):
                    #         raise ValueError(f"Attribute {fn} is not a method")
                    #     self._transform_action_space_fn = fn
                    # else:
                self._transform_action_space_fn = transform_action_space_fn
                #else:
                #    self._transform_action_space_fn = None

            def step(self, action) -> GymStepReturn:

                #if self._transform_action_space_fn is not None:
                action = self._transform_action_space_fn(action)

                obs, reward, done, info = self.env.step(action)
                info["standard"] = {"action": action, "reward": reward}
                return obs, reward, done, info

        env = ActionInfoWrapper(env,
                                transform_action_space_fn=transform_action_space_fn,
                                alter_action_space=alter_action_space)

    if not is_wrapped(env, Monitor):
        env = Monitor(env)

    if 'train' in kwargs and kwargs['train']:

        env = DummyVecEnv([lambda: env])

        for iteration in range(kwargs['iterations']):

            tensorboard_log = os.getcwd() + '/tensorboard/'
            if "group" in kwargs:
                tensorboard_log += args["group"]

            if 'safety' in kwargs and kwargs['safety'] == "mask":
                from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
                model = base_algorithm(MaskableActorCriticPolicy,
                                       env,
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
            else:

                # def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
                #     """
                #     Linear learning rate schedule.
                #
                #     :param initial_value: (float or str)
                #     :return: (function)
                #     """
                #     if isinstance(initial_value, str):
                #         initial_value = float(initial_value)
                #
                #     def func(progress_remaining: float) -> float:
                #         """
                #         Progress will decrease from 1 (beginning) to 0
                #         :param progress_remaining: (float)
                #         :return: (float)
                #         """
                #         return progress_remaining * initial_value
                #
                #     return func


                if kwargs['algorithm'] == "PPO":
                    if 'flag' in kwargs and (kwargs['flag'] == 6 or kwargs['flag'] == 7 or kwargs['flag'] == 8):
                        model = base_algorithm(
                               MlpPolicy,
                               env,
                               verbose=0,
                               tensorboard_log=tensorboard_log)

                    else:
                        model = base_algorithm(MlpPolicy,
                                               env,
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
                        model = base_algorithm(MlpPolicy,
                                                   env,
                                                   verbose=0,
                                                   tensorboard_log=tensorboard_log)
                    else:
                        model = base_algorithm(MlpPolicy,
                                               env,
                                               verbose=0,
                                               use_rms_prop=False,
                                               normalize_advantage=True,
                                               tensorboard_log=tensorboard_log,
                                               ent_coef=0.1,
                                               max_grad_norm=0.5,
                                               n_steps=2,
                                               gae_lambda=1.0,
                                               vf_coef=0.5,
                                               gamma=0.98,
                                               learning_rate=0.0015,
                                               use_sde=False,
                                               policy_kwargs=dict(
                                                   net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                                   activation_fn=nn.ReLU,
                                                   ortho_init=True)
                                               )

                #else:
                #    from stable_baselines3 import DQN
                #    model = DQN('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)

            # TODO: Remove
            # from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
            # model = base_algorithm(MaskableActorCriticPolicy, env, verbose=0, tensorboard_log=os.getcwd() + '/tensorboard/')

            callback = CallbackList([PendulumTrainCallback(safe_region=safe_region)])

            model.learn(total_timesteps=kwargs['total_timesteps'],
                        tb_log_name=name,
                        callback=callback)
            # log_interval=log_interval)

            # TODO: Overrides / maybe combine?
            # save_model(name, model)

    elif 'rollout' in kwargs and kwargs['rollout']:
        pass
        # env = DummyVecEnv([lambda: env])
        #
        # model = None
        # callback = None
        #
        # if name != "DEBUG":
        #     model = load_model(name + '.zip', base_algorithm)
        #     model.set_env(env)
        #
        #     callback = CallbackList([PendulumRolloutCallback(safe_region=safe_region)])
        #
        #     # TODO: Not needed for training(?)
        #     _logger = configure_logger(verbose=0, tb_log_name=name + '_E',
        #                                tensorboard_log=os.getcwd() + '/tensorboard/')
        #     model.set_logger(logger=_logger)
        #
        #     callback.init_callback(model=model)
        #
        # render = False
        # if 'render' in kwargs and kwargs['render']:
        #     render = True
        #
        # env_safe_action = False
        # if 'safety' in kwargs and kwargs['safety'] == "env_safe_action":
        #     env_safe_action = True
        #
        # # rollout(env, model, safe_region=safe_region, num_episodes=1, callback=callback, env_safe_action=env_safe_action, render=render, sleep=.05)
        # rollout(env, model, safe_region=safe_region, num_episodes=kwargs['iterations'], callback=callback,
        #         env_safe_action=env_safe_action,
        #         render=render, sleep=.1)

    if 'env' in locals(): env.close()

    if "group" in kwargs:
        rename_tf_events(kwargs["group"])


# def rollout(env, model=None, safe_region=None, num_episodes=1, callback=None, env_safe_action=False, render=False,
#             rgb_array=False, sleep=0.1):
#     is_vec_env = isinstance(env, VecEnv)
#     if is_vec_env:
#         if env.num_envs != 1:
#             logger.warning(f"You must pass only one environment when using this function")
#         is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
#     else:
#         is_monitor_wrapped = is_wrapped(env, Monitor)
#
#     if not is_monitor_wrapped:
#         logger.warning(f"Evaluation environment is not wrapped with a ``Monitor`` wrapper.")
#
#     frames = []
#     reset = True
#     for episode in range(num_episodes):
#
#         done, state = False, None
#
#         # Avoid double reset, as VecEnv are reset automatically.
#         if not isinstance(env, VecEnv) or reset:
#             obs = env.reset()
#             reset = False
#
#         # Give access to local variables
#         if callback:
#             callback.update_locals(locals())
#             callback.on_rollout_start()
#
#         while not done:
#
#             if render:
#                 # Does not render last step
#                 if rgb_array:
#                     frame = env.render(mode='rgb_array')
#                     frames.append(frame)
#                 else:
#                     env.render()
#
#             time.sleep(sleep)
#
#             if model is not None:
#
#                 action, state = model.predict(obs, state=state)  # deterministic=deterministic
#                 action = action[0]  # Action is dict
#                 print(action)
#
#                 # TODO: Check if masking used - also rollout masking without model!
#                 # if use_masking:
#                 #    action_masks = get_action_masks(env)
#                 #    actions, state = model.predict(obs,state=state, action_masks=action_masks)
#
#
#             elif env_safe_action:
#                 # TODO: Fix/Easier? / Could check for callable etc.
#                 action = env.get_attr('safe_action')[0](env, safe_region, None)
#
#             else:
#                 # TODO: Sample is [] for box and otherwise not?
#                 #action = env.action_space.sample()
#                 action = -30
#                 if isinstance(action, np.ndarray):
#                     action = action.item()
#
#                 if is_masking_supported(env):
#                     mask = get_action_masks(env)[0]
#                     action = random.choice(np.argwhere(mask == True))[0]
#
#             obs, reward, done, info = env.step([action])
#             # print(reward)
#
#             if render:
#                 # Prevent render after reset
#                 if not is_vec_env or is_vec_env and not done:
#                     env.render()
#
#             # Do not plot reset for last episode
#             if callback and not (done and episode == num_episodes - 1):
#                 # Give access to local variables
#                 callback.update_locals(locals())
#                 if callback.on_step() is False:
#                     return False
#
#     time.sleep(sleep)
#     env.close()
#
#     if rgb_array:
#         return frames


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, default='PPO', required=False,
                        help='RL algorithm')
    parser.add_argument('-e', '--env_id', type=str, default='MathPendulum-v0', required=False,
                        help='ID of a registered environment')
    parser.add_argument('-t', '--total_timesteps', type=int, default=20e4, required=False,  # 400
                        help='Total timesteps to train model')  # TODO: Episodes
    parser.add_argument('-n', '--name', type=str, default='DEBUG', required=False,
                        help='Base name for generated data')
    parser.add_argument('-s', '--safety', type=str, default=None, required=False,
                        help='Safety method')
    parser.add_argument('-i', '--iterations', type=int, default=1, required=False)
    parser.add_argument('-f', '--flag', type=int, default=0)
    args, unknown = parser.parse_known_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()

    from gym.envs.registration import register

    register(
        id='MathPendulum-v0',
        max_episode_steps=100,
        entry_point='sb3_contrib.common.envs.pendulum.math_pendulum_env:MathPendulumEnv',
    )


    args["train"] = True
    args["name"] = "run"
    args['iterations'] = 5
    args['total_timesteps'] = 8e4
    args["algorithm"] = "PPO"


    if args["flag"] == 0:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_UNTUNED"
    if args["flag"] == 1:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_UNTUNED_SAS"
        args["action_space"] = "small"
    if args["flag"] == 2:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_UNTUNED_INIT"
        args["init"] = "zero"

    if args["flag"] == 3:
        args["algorithm"] = "A2C"
        args['group'] = "A2C"
    if args["flag"] == 4:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_SAS"
        args["action_space"] = "small"
    if args["flag"] == 5:
        args["algorithm"] = "A2C"
        args['group'] = "A2C_INIT"
        args["init"] = "zero"

    if args["flag"] == 6:
        args['group'] = "PPO_UNTUNED"
    if args["flag"] == 7:
        args['group'] = "PPO_UNTUNED_SAS"
        args["action_space"] = "small"
    if args["flag"] == 8:
        args['group'] = "PPO_UNTUNED_INIT"
        args["init"] = "zero"

    if args["flag"] == 9:
        args['group'] = "PPO"
    if args["flag"] == 10:
        args['group'] = "PPO_SAS"
        args["action_space"] = "small"
    if args["flag"] == 11:
        args['group'] = "PPO_INIT"
        args["init"] = "zero"

    if args["flag"] == 12:
        args['group'] = "MASK"
        args["safety"] = "mask"
    if args["flag"] == 13:
        args['group'] = "MASK_SAS"
        args["safety"] = "mask"
        args["action_space"] = "small"
    if args["flag"] == 14:
        args['group'] = "MASK_INIT"
        args["safety"] = "mask"
        args["init"] = "zero"

    if args["flag"] == 15:
        args['group'] = "MASK_PUN"
        args["safety"] = "mask"
        args["punishment"] = "punish"
    if args["flag"] == 16:
        args['group'] = "MASK_SAS_PUN"
        args["safety"] = "mask"
        args["action_space"] = "small"
        args["punishment"] = "punish"
    if args["flag"] == 17:
        args['group'] = "MASK_INIT_PUN"
        args["safety"] = "mask"
        args["init"] = "zero"
        args["punishment"] = "punish"

    if args["flag"] == 18:
        args['group'] = "CBF"
        args["safety"] = "cbf"
    if args["flag"] == 19:
        args['group'] = "CBF_SAS"
        args["safety"] = "cbf"
        args["action_space"] = "small"
    if args["flag"] == 20:
        args['group'] = "CBF_INIT"
        args["safety"] = "cbf"
        args["init"] = "zero"

    if args["flag"] == 21:
        args['group'] = "CBF_PUN"
        args["safety"] = "cbf"
        args["punishment"] = "punish"
    if args["flag"] == 22:
        args['group'] = "CBF_SAS_PUN"
        args["safety"] = "cbf"
        args["action_space"] = "small"
        args["punishment"] = "punish"
    if args["flag"] == 23:
        args['group'] = "CBF_INIT_PUN"
        args["safety"] = "cbf"
        args["init"] = "zero"
        args["punishment"] = "punish"

    if args["flag"] == 24:
        args['group'] = "SHIELD"
        args["safety"] = "shield"
    if args["flag"] == 25:
        args['group'] = "SHIELD_SAS"
        args["safety"] = "shield"
        args["action_space"] = "small"
    if args["flag"] == 26:
        args['group'] = "SHIELD_INIT"
        args["safety"] = "shield"
        args["init"] = "zero"

    if args["flag"] == 27:
        args['group'] = "SHIELD_PUN"
        args["safety"] = "shield"
        args["punishment"] = "punish"
    if args["flag"] == 28:
        args['group'] = "SHIELD_SAS_PUN"
        args["safety"] = "shield"
        args["action_space"] = "small"
        args["punishment"] = "punish"
    if args["flag"] == 29:
        args['group'] = "SHIELD_INIT_PUN"
        args["safety"] = "shield"
        args["init"] = "zero"
        args["punishment"] = "punish"

    main(**args)

    # if args["flag"] == 0:
    #     args['group'] = "CBF_5"
    #     args["safety"] = "cbf"
    # if args["flag"] == 1:
    #     args['group'] = "CBF_SAS_5"
    #     args["safety"] = "cbf"
    #     args["action_space"] = "small"
    # if args["flag"] == 2:
    #     args['group'] = "CBF_INIT_5"
    #     args["safety"] = "cbf"
    #     args["init"] = "zero"
    # if args["flag"] == 3:
    #     args['group'] = "CBF_95"
    #     args["safety"] = "cbf"
    #     args["gamma"] = 0.95
    # if args["flag"] == 4:
    #     args['group'] = "CBF_SAS_95"
    #     args["safety"] = "cbf"
    #     args["action_space"] = "small"
    #     args["gamma"] = 0.95
    # if args["flag"] == 5:
    #     args['group'] = "CBF_INIT_95"
    #     args["gamma"] = 0.95
    #     args["safety"] = "cbf"
    #     args["init"] = "zero"
    # if args["flag"] == 6:
    #     args['group'] = "CBF_05"
    #     args["safety"] = "cbf"
    #     args["gamma"] = 0.05
    # if args["flag"] == 7:
    #     args['group'] = "CBF_SAS_05"
    #     args["safety"] = "cbf"
    #     args["action_space"] = "small"
    #     args["gamma"] = 0.05
    # if args["flag"] == 8:
    #     args['group'] = "CBF_INIT_05"
    #     args["gamma"] = 0.05
    #     args["safety"] = "cbf"
    #     args["init"] = "zero"
    # if args["flag"] == 9:
    #     args['group'] = "MASK_PUNH"
    #     args["safety"] = "mask"
    #     args["punishment"] = "punish"
    # if args["flag"] == 10:
    #     args['group'] = "MASK_SAS_PUNH"
    #     args["safety"] = "mask"
    #     args["action_space"] = "small"
    #     args["punishment"] = "punish"
    # if args["flag"] == 11:
    #     args['group'] = "MASK_INIT_PUNH"
    #     args["safety"] = "mask"
    #     args["init"] = "zero"
    #     args["punishment"] = "punish"
    # main(**args)



    #tags = [
        # "main/avg_abs_action_rl",  # ?
        #"main/avg_abs_safety_correction",  #
        # "main/avg_abs_thdot",  # ?
        # "main/avg_abs_theta",  # ?
        # "main/avg_safety_measure",  #
        #"main/episode_reward",  #
        # "main/episode_time",  #
        # "main/max_abs_action_rl",  # ??
        # "main/max_abs_safety_correction",  #
        # "main/max_abs_thdot",  # ?
        # "main/max_abs_theta",  # ?
        # "main/max_safety_measure",  # ?
        # "main/no_violation",  #
        # "main/rel_abs_safety_correction",
        # "main/avg_step_punishment",  #
        #"main/avg_step_reward_rl"  # ???
    #]

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

    from thesis.util import tf_events_to_plot, external_legend_res

    # for tag in tags:
    #     if tag == "main/avg_abs_action_rl":
    #         y_label = "Absolute action per step"# \overline{\left(\left|a\\right|\\right)}$"
    #     elif tag == "main/avg_abs_thdot":
    #         y_label = "$\mathrm{Mean\ absolute\ } \overline{\left(\left|\dot{\\theta}\\right|\\right)}$"
    #     elif tag == "main/avg_abs_theta":
    #         y_label = "$\mathrm{Mean\ absolute\ } \overline{\left(\left|\\theta\\right|\\right)}$"
    #     elif tag == "main/avg_step_reward_rl":
    #         y_label = "Reward per step" #%$\overline{r}
    #     elif tag == "main/episode_reward":
    #         y_label = "Episode return" #${r_{\mathrm{Episode}}}$
    #     elif tag == "main/max_safety_measure":
    #         y_label = "Maximal reward $r_{\mathrm{max}}$"
    #     elif tag == "main/no_violation":
    #         y_label = "Mean safety violation"
    #     elif tag == "main/avg_abs_safety_correction":
    #         y_label = "Safety correction per step"
    #     else:
    #         y_label = ''
    #
    #     dirsss = [
            #["A2C_UNTUNED", "PPO_UNTUNED"]
            #["PPO", "PPO_UNTUNED"],#Log/NotLog
            #["PPO_SAS", "PPO", "PPO_INIT"],
            #["A2C_UNTUNED_SAS", "A2C_UNTUNED", "A2C_UNTUNED_INIT"],
            #["A2C_SAS", "A2C", "A2C_INIT"],
            #["PPO_UNTUNED_SAS", "PPO_UNTUNED", "PPO_UNTUNED_INIT"],
            #["PPO_SAS", "PPO", "PPO_INIT"],
            #["MASK_SAS", "MASK", "MASK_INIT"],
            #["SHIELD_SAS", "SHIELD", "SHIELD_INIT"],
            #["MASK", "SHIELD"] #SAME FOR CBF NO VIOLATION PLOT
            #["CBF_SAS", "CBF", "CBF_INIT"],
            #["MASK_SAS_PUN", "MASK_PUN", "MASK_INIT_PUN"],
            #["SHIELD_SAS_PUN", "SHIELD_PUN", "SHIELD_INIT_PUN"],
            #["CBF_SAS_PUN", "CBF_PUN", "CBF_INIT_PUN"],
            #["MASK_SAS_PUNH", "MASK_PUNH", "MASK_INIT_PUNH"],
            #["CBF_SAS_05", "CBF_05", "CBF_INIT_05"],
            #["CBF_SAS_5", "CBF_5", "CBF_INIT_5"],
            #["CBF_SAS_95", "CBF_95", "CBF_INIT_95"],
        #]
        # for i, dirss in enumerate(dirsss):
        #     tf_events_to_plot(dirss=dirss, #"standard"
        #                       tags=[tag],
        #                       x_label='Episode',
        #                       y_label=y_label,
        #                       width=2.5, #5
        #                       height=2.5, #2.5
        #                       episode_length=100,
        #                       window_size=11, #41
        #                       save_as=f"pdfs/{i}{tag.split('/')[1]}")

    # labels = []
    # for label in dirss:
    #    labels.append(label.replace('_','/'))

    # external_legend_res(labels=labels, save_as=f"pdfs/leg_{tag.split('/')[1]}")

    # for alg in ["PPO", "A2C"]: #TODO: A2C run
    #     args["algorithm"] = alg
    #     for safety in ["no_safety", "shield", "mask", "cbf"]:
    #         args["safety"] = safety
    #         for action_space in ["safetorqueas", "unsafetorqueas"]:
    #             args["action_space"] = action_space
    #             for init in ["zero", "random"]:
    #                 args["init"] = init
    #                 for reward in ["opposing", "safety"]:
    #                     args["reward"] = reward
    #                     if not args["safety"] == "no_safety":
    #                         for punishment in ["nopunish", "lightpunish", "heavypunish"]:
    #                             args["punishment"] = punishment
    #                             if safety=="cbf":
    #                                 for gamma in [0.25, 0.75]:
    #                                     args["gamma"] = gamma
    #                                     args["group"] = f"{alg}_{safety}_{action_space}_{init}_{reward}_{punishment}_{str(gamma)}"
    #                                     if not os.path.isdir(os.getcwd() + f"/tensorboard/{args['group']}"):
    #                                         main(**args)
    #                                     print(f"Finished training {args['group']} ...")
    #                             else:
    #                                 args["group"] = f"{alg}_{safety}_{action_space}_{init}_{reward}_{punishment}"
    #                                 if not os.path.isdir(os.getcwd() + f"/tensorboard/{args['group']}"):
    #                                     main(**args)
    #                                 print(f"Finished training {args['group']} ...")
    #                             #time.sleep(3)
    #                     else:
    #                         args["group"] = f"{alg}_{safety}_{action_space}_{init}_{reward}"
    #                         if not os.path.isdir(os.getcwd() + f"/tensorboard/{args['group']}"):
    #                             main(**args)
    #                         print(f"Finished training {args['group']} ...")


    # os.system("say The program finished.")

    ########################

    # args["train"] = True
    # args['iterations'] = 2
    # args['total_timesteps'] = 1e4
    # args["name"]="NoSafety"
    # args["safety"] = "shield"

    # name = "test"
    # args["group"] = name
    # args["name"] = name
    # args["name"] = "Shield_Punish"
    # args["safety"] = "mask"
    # args["name"] = "Mask"
    # args["name"] = "Mask_Punish"
    # args["safety"] = "cbf"

    # args['train'] = True
    # args['safety'] = 'mask'
    # args['name'] = 'noSafetyTest'

    # args['rollout'] = True
    # args['render'] = True
    # args['safety'] = 'cbf'
    # args["gamma"] = 0.99999
    # main(**args)

    # args['name'] = 'maskTest'

    # main(**args)
