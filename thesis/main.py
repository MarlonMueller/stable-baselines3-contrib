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

    # Initialize environment
    if kwargs['env_id'] not in [env_spec.id for env_spec in gym.envs.registry.all()]:
         KeyError(f"Environment {kwargs['env_id']} is not registered")

    if "init" in kwargs and kwargs["init"] == "random":
        if "reward" in kwargs and kwargs["reward"] == "opposing":
            env = gym.make(kwargs['env_id'], init="random", reward="opposing")
        else:
            env = gym.make(kwargs['env_id'], init="random")
    else:
        if "reward" in kwargs and kwargs["reward"] == "opposing":
            env = gym.make(kwargs['env_id'], reward="opposing")
        else:
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

    theta_roa = 3.092505268377452
    vertices = np.array([
         [-theta_roa, 12.762720155208534],  # LeftUp
         [theta_roa, -5.890486225480862],  # RightUp
         [theta_roa, -12.762720155208534],  # RightLow
         [-theta_roa, 5.890486225480862]  # LeftLow
    ])
    safe_region = SafeRegion(vertices=vertices)

    if "action_space" in kwargs and kwargs["action_space"] == "large":
        transform_action_space_fn = lambda a: 2 * (a - 30)
        alter_action_space = gym.spaces.Discrete(61)
    elif "action_space" in kwargs and kwargs["action_space"] == "small":
        transform_action_space_fn = lambda a: 2 * (a - 8)
        alter_action_space = gym.spaces.Discrete(17)
    elif "action_space" in kwargs and kwargs["action_space"] == "verysmall":
        transform_action_space_fn = lambda a: (a - 8)
        alter_action_space = gym.spaces.Discrete(17)
    else:
        transform_action_space_fn = lambda a: 2 * (a - 15)
        alter_action_space = gym.spaces.Discrete(31)

    if 'safety' in kwargs and kwargs['safety'] is not None:


        if kwargs['safety'] == "shield":
            from sb3_contrib.common.wrappers import SafetyShield

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            # return -abs(action - action_shield) 1:1
            if "punishment" in kwargs:
                if kwargs["punishment"] == "lightpunish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      action_shield: Union[int, float, np.ndarray]) -> float:
                        return -abs(action - action_shield) * 0.5
                elif kwargs["punishment"] == "heavypunish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      action_shield: Union[int, float, np.ndarray]) -> float:
                        return -abs(action - action_shield) * 4
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
                return np.array([0, (env.dt / (env.m * env.l ** 2))])
                #return np.array([(env.dt ** 2 / (env.m * env.l ** 2)), (env.dt / (env.m * env.l ** 2))])

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            if "punishment" in kwargs:
                if kwargs["punishment"] == "lightpunish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      action_cbf: Union[int, float, np.ndarray]) -> float:
                        return -abs(action_cbf) * 0.5
                elif kwargs["punishment"] == "heavypunish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      action_cbf: Union[int, float, np.ndarray]) -> float:
                        return -abs(action_cbf) * 4
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

            #TODO, f und g as other methods in env?
            #TODO Erklärung Problem
            #TODO Liste Thesis



        elif kwargs['safety'] == "mask":
            from sb3_contrib.common.wrappers import SafetyMask

            def dynamics_fn(env: gym.Env, action: Union[int, float, np.ndarray]) -> np.ndarray:
                theta, thdot = env.state
                return env.dynamics(theta, thdot, action)

            # We only care about the mask, fail-safe controller is not in use
            if "punishment" in kwargs:
                if kwargs["punishment"] == "lightpunish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      mask: Union[int, float, np.ndarray]) -> float:
                        return -(1 - (np.sum(mask)-1) / (len(mask)-1)) * 5
                elif kwargs["punishment"] == "heavypunish":
                    def punishment_fn(env: gym.Env, safe_region: SafeRegion,
                                      action: Union[int, float, np.ndarray],
                                      mask: Union[int, float, np.ndarray]) -> float:
                        return -(1 - (np.sum(mask)-1) / (len(mask)-1)) * 40
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
            def __init__(self,env, alter_action_space = None,
             transform_action_space_fn = None):
                super().__init__(env)

                if alter_action_space is not None:
                    self.action_space = alter_action_space

                if transform_action_space_fn is not None:
                    if isinstance(transform_action_space_fn, str):
                        fn = getattr(self.env, transform_action_space_fn)
                        if not callable(fn):
                            raise ValueError(f"Attribute {fn} is not a method")
                        self._transform_action_space_fn = fn
                    else:
                        self._transform_action_space_fn = transform_action_space_fn
                else:
                    self._transform_action_space_fn = None

            def step(self, action) -> GymStepReturn:

                if self._transform_action_space_fn is not None:
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
                model = base_algorithm(MaskableActorCriticPolicy, env, verbose=0, tensorboard_log=tensorboard_log)
            else:

                def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
                    """
                    Linear learning rate schedule.

                    :param initial_value: (float or str)
                    :return: (function)
                    """
                    if isinstance(initial_value, str):
                        initial_value = float(initial_value)

                    def func(progress_remaining: float) -> float:
                        """
                        Progress will decrease from 1 (beginning) to 0
                        :param progress_remaining: (float)
                        :return: (float)
                        """
                        return progress_remaining * initial_value

                    return func

                if kwargs['algorithm'] == "PPO":
                    if 'flag' in kwargs and kwargs['flag']:
                        model = base_algorithm(MlpPolicy,
                                               env,
                                               verbose=0,
                                               tensorboard_log=tensorboard_log)
                    else:

                        8#TUNE1 - gut, aber ausbrecher am Ende
                        # model = base_algorithm(MlpPolicy,
                        #                    env,
                        #                    verbose=0,
                        #                    tensorboard_log=tensorboard_log,
                        #                    batch_size=4,
                        #                    n_steps=8,
                        #                    gamma=0.95,
                        #                    learning_rate=4.935774549732434e-05,
                        #                    ent_coef=0.06740124751907833,
                        #                    clip_range=0.4,
                        #                    n_epochs=20,
                        #                    gae_lambda=1.0,
                        #                    max_grad_norm=.5,
                        #                    vf_coef=0.4149614449281107,
                        #                    policy_kwargs=dict(
                        #                        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                        activation_fn=nn.ReLU,
                        #                        ortho_init=True)
                        #                    )
                        # # Tune2 - nicht gut, nicht so schlimm wie A2C in Run 1 aber keine Convergenz
                        # model = base_algorithm(MlpPolicy,
                        #                        env,
                        #                        verbose=0,
                        #                        tensorboard_log=tensorboard_log,
                        #                        batch_size=4,
                        #                        n_steps=8,
                        #                        gamma=0.9,
                        #                        learning_rate=0.00023351444497526097,
                        #                        ent_coef=3.3637015275625054e-05,
                        #                        clip_range=0.2,
                        #                        n_epochs=1,
                        #                        gae_lambda=0.92,
                        #                        max_grad_norm=5,
                        #                        vf_coef=0.39557068189434147,
                        #                        policy_kwargs=dict(
                        #                            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                            activation_fn=nn.ReLU,
                        #                            ortho_init=True)
                        #                        )
                        # Tune3 - best of now
                        # model = base_algorithm(MlpPolicy,
                        #                        env,
                        #                        verbose=0,
                        #                        tensorboard_log=tensorboard_log,
                        #                        batch_size=4,
                        #                        n_steps=8,
                        #                        gamma=0.9,
                        #                        learning_rate=0.0001252181,
                        #                        ent_coef=0.00010173,
                        #                        clip_range=0.3,
                        #                        n_epochs=2,
                        #                        gae_lambda=1.0,
                        #                        max_grad_norm=2,
                        #                        vf_coef=0.7261347,
                        #                        policy_kwargs=dict(
                        #                            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                            activation_fn=nn.ReLU,
                        #                            ortho_init=True)
                        #                        )
                        #Tune 4
                        # model = base_algorithm(MlpPolicy,
                        #                        env,
                        #                        verbose=0,
                        #                        tensorboard_log=tensorboard_log,
                        #                        batch_size=512,
                        #                        n_steps=2048,
                        #                        gamma=0.95,
                        #                        learning_rate=0.00323,
                        #                        ent_coef=0.1,
                        #                        clip_range=0.4,
                        #                        n_epochs=10,
                        #                        gae_lambda=1.0,
                        #                        max_grad_norm=5,
                        #                        vf_coef=0.99,
                        #                        policy_kwargs=dict(
                        #                            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                            activation_fn=nn.ReLU,
                        #                            ortho_init=True)
                        #                        )
                        model = base_algorithm(MlpPolicy,
                                               env,
                                               verbose=0,
                                               tensorboard_log=tensorboard_log,
                                               batch_size=2,
                                               n_steps=2,
                                               gamma=0.9,
                                               learning_rate=1e-4,
                                               ent_coef=0.00015,
                                               clip_range=0.2,
                                               n_epochs=2,
                                               gae_lambda=1.0,
                                               max_grad_norm=0.8,
                                               vf_coef=1.0,
                                               policy_kwargs=dict(
                                                   net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                                   activation_fn=nn.ReLU,
                                                   ortho_init=True)
                                               )



                                           #activation_fn=tanh) #lr_schedule, act_fn, net_arch, otho_init


                elif kwargs['algorithm'] == "A2C":
                    if 'flag' in kwargs and kwargs['flag']:
                        model = base_algorithm(MlpPolicy,
                                               env,
                                               verbose=0,
                                               tensorboard_log=tensorboard_log)
                    else:
                        #Tune1 - convergiert manchmal net
                        # model = base_algorithm(MlpPolicy,
                        #                    env,
                        #                    verbose=0,
                        #                    use_rms_prop=False,
                        #                    normalize_advantage=True,
                        #                    tensorboard_log=tensorboard_log,
                        #                    ent_coef=0.03489354223693093,
                        #                    max_grad_norm=2,
                        #                    n_steps=2,
                        #                    gae_lambda=1.0,
                        #                    vf_coef=0.6841529890300497,
                        #                    gamma=0.9,
                        #                    learning_rate=linear_schedule(0.0015845365679309144),
                        #                    use_sde=False,
                        #                    policy_kwargs=dict(
                        #                        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                        activation_fn=nn.ReLU,
                        #                        ortho_init=True)
                        #                    )
                        #Tune2 - A2 durchgehend zacken
                        # model = base_algorithm(MlpPolicy,
                        #                        env,
                        #                        verbose=0,
                        #                        use_rms_prop=False,
                        #                        normalize_advantage=True,
                        #                        tensorboard_log=tensorboard_log,
                        #                        ent_coef=0.05877945647872223,
                        #                        max_grad_norm=5,
                        #                        n_steps=2,
                        #                        gae_lambda=0.92,
                        #                        vf_coef=0.2993644201737006,
                        #                        gamma=0.95,
                        #                        learning_rate=0.001058460163315873,
                        #                        use_sde=False,
                        #                        policy_kwargs=dict(
                        #                            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                            activation_fn=nn.ReLU,
                        #                            ortho_init=True)
                        #                        )

                        # Tune3 - Katastrophe
                        # model = base_algorithm(MlpPolicy,
                        #                        env,
                        #                        verbose=0,
                        #                        use_rms_prop=False,
                        #                        normalize_advantage=True,
                        #                        tensorboard_log=tensorboard_log,
                        #                        ent_coef=0.476890,
                        #                        max_grad_norm=0.7,
                        #                        n_steps=4,
                        #                        gae_lambda=1.0,
                        #                        vf_coef=0,
                        #                        gamma=0.95,
                        #                        learning_rate=0.00088739150,
                        #                        use_sde=False,
                        #                        policy_kwargs=dict(
                        #                            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                            activation_fn=nn.Tanh,
                        #                            ortho_init=True)
                        #                        )
                        # Tune4 -Gut, Mehr Varianz
                        # model = base_algorithm(MlpPolicy,
                        #                        env,
                        #                        verbose=0,
                        #                        use_rms_prop=False,
                        #                        normalize_advantage=True,
                        #                        tensorboard_log=tensorboard_log,
                        #                        ent_coef=0.01,
                        #                        max_grad_norm=0.9,
                        #                        n_steps=2,
                        #                        gae_lambda=0.8,
                        #                        vf_coef=0.99,
                        #                        gamma=0.9,
                        #                        learning_rate=0.004,
                        #                        use_sde=False,
                        #                        policy_kwargs=dict(
                        #                            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                        #                            activation_fn=nn.Tanh,
                        #                            ortho_init=True)
                        #                        )
                        #Tune4 - 1000
                        model = base_algorithm(MlpPolicy,
                                               env,
                                               verbose=0,
                                               use_rms_prop=False,
                                               normalize_advantage=True,
                                               tensorboard_log=tensorboard_log,
                                               ent_coef=0.1,
                                               max_grad_norm=0.8,
                                               n_steps=8,
                                               gae_lambda=1.0,
                                               vf_coef=0.285,
                                               gamma=0.9,
                                               learning_rate=0.001,
                                               use_sde=False,
                                               policy_kwargs=dict(
                                                   net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                                                   activation_fn=nn.ReLU,
                                                   ortho_init=True)
                                               )
                                           #policy_kwargs="dict(log_std_init=-2, ortho_init=False)")
                    # model = base_algorithm(MlpPolicy,
                    #                        env, verbose=0,
                    #                        tensorboard_log=tensorboard_log,
                    #                        normalize_advantage=False,
                    #                        max_grad_norm=1,
                    #                        use_rms_prop=True,
                    #                        gae_lambda=0.95,
                    #                        n_steps=8,
                    #                        learning_rate=0.00730,
                    #                        ent_coef=2.5111150,
                    #                        vf_coef=0.79)
                else:
                    from stable_baselines3 import DQN
                    model = DQN('MlpPolicy',env, verbose=0, tensorboard_log=tensorboard_log)

            #TODO: Remove
            #from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
            #model = base_algorithm(MaskableActorCriticPolicy, env, verbose=0, tensorboard_log=os.getcwd() + '/tensorboard/')

            callback = CallbackList([PendulumTrainCallback(safe_region=safe_region)])

            model.learn(total_timesteps=kwargs['total_timesteps'],
                        tb_log_name=name,
                        callback=callback)
            # log_interval=log_interval)

            # TODO: Overrides / maybe combine?
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
        rollout(env, model, safe_region=safe_region, num_episodes=kwargs['iterations'], callback=callback, env_safe_action=env_safe_action,
                render=render, sleep=.1)

    if 'env' in locals(): env.close()

    if "group" in kwargs:
        rename_tf_events(kwargs["group"])


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
                #print(action)

                #TODO: Check if masking used - also rollout masking without model!
                #if use_masking:
                #    action_masks = get_action_masks(env)
                #    actions, state = model.predict(obs,state=state, action_masks=action_masks)


            elif env_safe_action:
                #TODO: Fix/Easier? / Could check for callable etc.
                action = env.get_attr('safe_action')[0](env, safe_region, None)

            else:
                #TODO: Sample is [] for box and otherwise not?
                action = env.action_space.sample()
                if isinstance(action, np.ndarray):
                    action = action.item()

                if is_masking_supported(env):
                    mask = get_action_masks(env)[0]
                    action = random.choice(np.argwhere(mask==True))[0]

            obs, reward, done, info = env.step([action])
            #print(reward)

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
    parser.add_argument('-i', '--iterations', type=int, default=1, required=False)
    parser.add_argument('-f', '--flag', type=bool, default=False)
    args, unknown = parser.parse_known_args()
    return vars(args)

if __name__ == '__main__':
    args = parse_arguments()

    # Debug
    #args['rollout'] = True
    #args['render'] = True
    #args['safety'] = "shield"

    #TODO: PPO Weird bug - total timesteps?

    ########################

    from gym.envs.registration import register

    register(
        id='MathPendulum-v0',
        max_episode_steps=100,
        entry_point='sb3_contrib.common.envs.pendulum.math_pendulum_env:MathPendulumEnv',
    )

    tags = [
        "main/avg_abs_action_rl",#
        "main/avg_abs_safety_correction",#
        "main/avg_abs_thdot",#
        "main/avg_abs_theta",#
        "main/avg_safety_measure",#
        "main/episode_reward",#
        "main/episode_time",#
        "main/max_abs_action_rl",#
        "main/max_abs_safety_correction",#
        "main/max_abs_thdot",#
        "main/max_abs_theta", #
        "main/max_safety_measure",#
        "main/no_violation", #
        "main/rel_abs_safety_correction",
        "main/avg_step_punishment", #
        "main/avg_step_reward_rl" #
    ]

    #NoSafety:
    #Violation Graph
    #Opposite: Normal / Bigger Actio Space using 0 / Random Init. (Avg. (and Max?)): Avg. Step Reward, Th, Thdot? NOT: Action; Safety
    #Safety: Normal / Bigger Actio Space using 0 / Random Init. (Avg. (and Max?)): Avg. Step Reward, Th, Thdot?

    #Shield:

    #"{alg}_{safety}_{action_space}_{init}_{reward}_{punishment}_{str(gamma)}"


    # dirss = [
    #    "PPO_no_safety_safetorqueas_zero_opposing",
    #    "PPO_no_safety_safetorqueas_random_opposing",
    #    "PPO_no_safety_unsafetorqueas_zero_opposing",
    #    "PPO_no_safety_unsafetorqueas_random_opposing",
    #     "PPO_no_safety_safetorqueas_zero_safety",
    #     "PPO_no_safety_safetorqueas_random_safety",
    #     "PPO_no_safety_unsafetorqueas_zero_safety",
    #     "PPO_no_safety_unsafetorqueas_random_safety",
    # ]

    # dirss = [
    #         "PPO_no_safety_safetorqueas_zero_safety",
    #         "PPO_no_safety_unsafetorqueas_zero_safety",
    #         "PPO_no_safety_safetorqueas_zero_opposing",
    #         "PPO_no_safety_unsafetorqueas_zero_opposing",
    #     ]
    # dirss = [
    #     "PPO_no_safety_safetorqueas_random_safety",
    #     "PPO_no_safety_unsafetorqueas_random_safety",
    #     "PPO_no_safety_safetorqueas_random_opposing",
    #     "PPO_no_safety_unsafetorqueas_random_opposing",
    # ]


    #SHIELDING
    # dirss = [
    #
    #     "PPO_shield_safetorqueas_zero_safety_nopunish",
    #     "PPO_shield_safetorqueas_zero_opposing_nopunish",
    #     "PPO_shield_safetorqueas_random_safety_nopunish",
    #     "PPO_shield_safetorqueas_random_opposing_nopunish",
    #     "PPO_shield_unsafetorqueas_zero_safety_nopunish",
    #     "PPO_shield_unsafetorqueas_zero_opposing_nopunish",
    #     "PPO_shield_unsafetorqueas_random_safety_nopunish",
    #     "PPO_shield_unsafetorqueas_random_opposing_nopunish",
    #
    # ]
    #
    # dirss = [
    #
    #     "PPO_shield_safetorqueas_zero_safety_lightpunish",
    #     "PPO_shield_safetorqueas_zero_opposing_lightpunish",
    #     "PPO_shield_safetorqueas_random_safety_lightpunish",
    #     "PPO_shield_safetorqueas_random_opposing_lightpunish",
    #     "PPO_shield_unsafetorqueas_zero_safety_lightpunish",
    #     "PPO_shield_unsafetorqueas_zero_opposing_lightpunish",
    #     "PPO_shield_unsafetorqueas_random_safety_lightpunish",
    #     "PPO_shield_unsafetorqueas_random_opposing_lightpunish",
    #
    # ]

    # dirss = [
    #
    #     "PPO_shield_safetorqueas_zero_safety_heavypunish",
    #     "PPO_shield_safetorqueas_zero_opposing_heavypunish",
    #     "PPO_shield_safetorqueas_random_safety_heavypunish",
    #     "PPO_shield_safetorqueas_random_opposing_heavypunish",
    #     "PPO_shield_unsafetorqueas_zero_safety_heavypunish",
    #     "PPO_shield_unsafetorqueas_zero_opposing_heavypunish",
    #     "PPO_shield_unsafetorqueas_random_safety_heavypunish",
    #     "PPO_shield_unsafetorqueas_random_opposing_heavypunish",
    #
    # ]

    # # MASKING
    # dirss = [
    #
    #     "PPO_mask_safetorqueas_zero_safety_nopunish",
    #     "PPO_mask_safetorqueas_zero_opposing_nopunish",
    #     "PPO_mask_safetorqueas_random_safety_nopunish",
    #     "PPO_mask_safetorqueas_random_opposing_nopunish",
    #     "PPO_mask_unsafetorqueas_zero_safety_nopunish",
    #     "PPO_mask_unsafetorqueas_zero_opposing_nopunish",
    #     "PPO_mask_unsafetorqueas_random_safety_nopunish",
    #     "PPO_mask_unsafetorqueas_random_opposing_nopunish",
    #
    # ]
    #
    # for tag in tags:
    #     tf_events_to_plot(dirss=dirss, #"standard"
    #                       tags=[tag],
    #                       x_label='Episode',
    #                       y_label='',
    #                       width=7, #5
    #                       height=3.5, #2.5
    #                       episode_length=100,
    #                       window_size=45,
    #                       save_as=f"pdfs/NOPUN{tag.split('/')[1]}")
    #
    # dirss = [
    #
    #     "PPO_mask_safetorqueas_zero_safety_lightpunish",
    #     "PPO_mask_safetorqueas_zero_opposing_lightpunish",
    #     "PPO_mask_safetorqueas_random_safety_lightpunish",
    #     "PPO_mask_safetorqueas_random_opposing_lightpunish",
    #     "PPO_mask_unsafetorqueas_zero_safety_lightpunish",
    #     "PPO_mask_unsafetorqueas_zero_opposing_lightpunish",
    #     "PPO_mask_unsafetorqueas_random_safety_lightpunish",
    #     "PPO_mask_unsafetorqueas_random_opposing_lightpunish",
    #
    # ]
    #
    # for tag in tags:
    #     tf_events_to_plot(dirss=dirss, #"standard"
    #                       tags=[tag],
    #                       x_label='Episode',
    #                       y_label='',
    #                       width=7, #5
    #                       height=3.5, #2.5
    #                       episode_length=100,
    #                       window_size=45,
    #                       save_as=f"pdfs/LIGHTPUN{tag.split('/')[1]}")
    #
    # dirss = [
    #
    #     "PPO_mask_safetorqueas_zero_safety_heavypunish",
    #     "PPO_mask_safetorqueas_zero_opposing_heavypunish",
    #     "PPO_mask_safetorqueas_random_safety_heavypunish",
    #     "PPO_mask_safetorqueas_random_opposing_heavypunish",
    #     "PPO_mask_unsafetorqueas_zero_safety_heavypunish",
    #     "PPO_mask_unsafetorqueas_zero_opposing_heavypunish",
    #     "PPO_mask_unsafetorqueas_random_safety_heavypunish",
    #     "PPO_mask_unsafetorqueas_random_opposing_heavypunish",
    #
    # ]
    #
    # for tag in tags:
    #     tf_events_to_plot(dirss=dirss, #"standard"
    #                       tags=[tag],
    #                       x_label='Episode',
    #                       y_label='',
    #                       width=7, #5
    #                       height=3.5, #2.5
    #                       episode_length=100,
    #                       window_size=45,
    #                       save_as=f"pdfs/HEAVYPUN{tag.split('/')[1]}")

    # MASKING
    # dirss = [
    #
    #     "PPO_cbf_safetorqueas_zero_safety_nopunish_0.75",
    #     "PPO_cbf_safetorqueas_zero_opposing_nopunish_0.75",
    #     "PPO_cbf_safetorqueas_random_safety_nopunish_0.75",
    #     "PPO_cbf_safetorqueas_random_opposing_nopunish_0.75",
    #     "PPO_cbf_unsafetorqueas_zero_safety_nopunish_0.75",
    #     "PPO_cbf_unsafetorqueas_zero_opposing_nopunish_0.75",
    #     "PPO_cbf_unsafetorqueas_random_safety_nopunish_0.75",
    #     "PPO_cbf_unsafetorqueas_random_opposing_nopunish_0.75",
    #
    # ]

    # for tag in tags:
    #     tf_events_to_plot(dirss=dirss,  # "standard"
    #                       tags=[tag],
    #                       x_label='Episode',
    #                       y_label='',
    #                       width=7,  # 5
    #                       height=3.5,  # 2.5
    #                       episode_length=100,
    #                       window_size=45,
    #                       save_as=f"pdfs/75NOPUN{tag.split('/')[1]}")
    #
    # dirss = [
    #
    #     "PPO_cbf_safetorqueas_zero_safety_lightpunish_0.75",
    #     "PPO_cbf_safetorqueas_zero_opposing_lightpunish_0.75",
    #     "PPO_cbf_safetorqueas_random_safety_lightpunish_0.75",
    #     "PPO_cbf_safetorqueas_random_opposing_lightpunish_0.75",
    #     "PPO_cbf_unsafetorqueas_zero_safety_lightpunish_0.75",
    #     "PPO_cbf_unsafetorqueas_zero_opposing_lightpunish_0.75",
    #     "PPO_cbf_unsafetorqueas_random_safety_lightpunish_0.75",
    #     "PPO_cbf_unsafetorqueas_random_opposing_lightpunish_0.75",
    #
    # ]
    #
    # for tag in tags:
    #     tf_events_to_plot(dirss=dirss,  # "standard"
    #                       tags=[tag],
    #                       x_label='Episode',
    #                       y_label='',
    #                       width=7,  # 5
    #                       height=3.5,  # 2.5
    #                       episode_length=100,
    #                       window_size=45,
    #                       save_as=f"pdfs/75LIGHTPUN{tag.split('/')[1]}")
    #
    # dirss = [
    #
    #     "PPO_cbf_safetorqueas_zero_safety_heavypunish_0.75",
    #     "PPO_cbf_safetorqueas_zero_opposing_heavypunish_0.75",
    #     "PPO_cbf_safetorqueas_random_safety_heavypunish_0.75",
    #     "PPO_cbf_safetorqueas_random_opposing_heavypunish_0.75",
    #     "PPO_cbf_unsafetorqueas_zero_safety_heavypunish_0.75",
    #     "PPO_cbf_unsafetorqueas_zero_opposing_heavypunish_0.75",
    #     "PPO_cbf_unsafetorqueas_random_safety_heavypunish_0.75",
    #     "PPO_cbf_unsafetorqueas_random_opposing_heavypunish_0.75",
    #
    # ]
    #
    # for tag in tags:
    #     tf_events_to_plot(dirss=dirss,  # "standard"
    #                       tags=[tag],
    #                       x_label='Episode',
    #                       y_label='',
    #                       width=7,  # 5
    #                       height=3.5,  # 2.5
    #                       episode_length=100,
    #                       window_size=45,
    #                       save_as=f"pdfs/75HEAVYPUN{tag.split('/')[1]}")



    #tags = ["main/avg_abs_theta"]
    #tags = ["main/avg_abs_thdot"]
    #TODO: Avg. Reward





    #------------------------

    #All: Safety Measure / Rel? ActionRL?

    #Violation Graph?
    #Normal vs. Big Action Space using 0 and Random Init. (Avg. AND Max? Theta/Thdot) Max not needed Avg.StepReward

    #Safety Measure later/Rel SafetyMeasureLater

    #remove_tf_logs()

    # args['rollout'] = True
    # args['render'] = True
    # #args['safety'] = 'mask'
    # args["algorithm"] = "A2C"
    # args['name'] = 'TUNED/run'
    # main(**args)

    args["train"] = True
    args["name"] = "run"
    args['iterations'] = 10
    #args["safety"] = "standard"
    # # #
    if not args['flag']:
        args['total_timesteps'] = 5e4
        #args['group'] ="A2C_TUNED_MODEL"
        #args["algorithm"] = "A2C"
        #main(**args)
        args['group'] = "PPO_TUNED_OBS"
        args["algorithm"] = "PPO"
        main(**args)
    # else:
    #     print("Test")
    #     args['total_timesteps'] = 10e4
    #     args['group'] = "A2C_UNTUNED_MODEL"
    #     args["algorithm"] = "A2C"
    #     main(**args)
    #     args['total_timesteps'] = 20e4
    #     args['group'] = "PPO_UNTUNED_MODEL"
    #     args["algorithm"] = "PPO"
    #     main(**args)


    #
    # args["safety"] = "shield"
    # #args["safety"] = "mask"
    # #args["safety"] = "mask"
    # args["punishment"] = "heavypunish"
    #
    # args["init"] = "random"
    # #args["reward"] = "opposing"
    # args["group"] = "test"
    # main(**args)

    tags = [
        #"main/avg_abs_action_rl",  # ?
        #"main/avg_abs_safety_correction",  #
        #"main/avg_abs_thdot",  # ?
        #"main/avg_abs_theta",  # ?
        #"main/avg_safety_measure",  #
        #"main/episode_reward",  #
        #"main/episode_time",  #
        #"main/max_abs_action_rl",  # ??
        #"main/max_abs_safety_correction",  #
        #"main/max_abs_thdot",  # ?
        #"main/max_abs_theta",  # ?
        #"main/max_safety_measure",  # ?
        #"main/no_violation",  #
        #"main/rel_abs_safety_correction",
        #"main/avg_step_punishment",  #
        #"main/avg_step_reward_rl"  # ???
    ]

    #PRELIMINARY
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
    for tag in tags:
        if tag == "main/avg_abs_action_rl":
           y_label = "$\mathrm{Mean\ absolute\ action\ } \overline{\left(\left|a\\right|\\right)}$"
        elif tag == "main/avg_abs_thdot":
           y_label = "$\mathrm{Mean\ absolute\ } \overline{\left(\left|\dot{\\theta}\\right|\\right)}$"
        elif tag == "main/avg_abs_theta":
           y_label = "$\mathrm{Mean\ absolute\ } \overline{\left(\left|\\theta\\right|\\right)}$"
        elif tag == "main/avg_step_reward_rl":
            y_label = "Mean reward per step $\overline{r}$"
        elif tag == "main/episode_reward":
            y_label = "Episode reward ${r_{\mathrm{Episode}}}$"
        elif tag== "main/max_safety_measure":
            y_label = "Maximal reward $r_{\mathrm{max}}$"
        elif tag == "main/no_violation":
            y_label = "States $s$ in ROA"
        else:
           y_label = ''

        #dirss = ["PPO_UNTUNED", "A2C_UNTUNED"]
        #dirss = ["PPO_TUNED", "A2C_TUNED"]
        dirss = ["PPO_TUNED_MODEL_OBS"]
        #dirss = ["PPO_TUNED"]
        tf_events_to_plot(dirss=dirss, #"standard"
                          tags=[tag],
                          x_label='Episode',
                          y_label=y_label,
                          width=2.5, #5
                          height=2.5, #2.5
                          episode_length=100,
                          window_size=41, #41
                          save_as=f"pdfs/{tag.split('/')[1]}")

    #labels = []
    #for label in dirss:
    #    labels.append(label.replace('_','/'))

    #external_legend_res(labels=labels, save_as=f"pdfs/leg_{tag.split('/')[1]}")


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




    #args["group"] = "standard"
    #main(**args)

    #Algorithm
    # Different reward function
    #Rollout after training
    # Do not start at 0,0, train at 0,0 - start random
    # Choose bigger action space than allowed
    # Same or different safety vs. reward function

    # Punish + diff. Punishment not just add reward but replace
    #args["safety"] = "shield"
    #args["group"] = "shield"
    #main(**args)

    # Punish + Change method of punishment
    #args["safety"] = "mask"
    #args["group"] = "mask"
    #main(**args)

    # Dif.. Gamma + Rollout, Punish + diff. Punishment
    #args["safety"] = "cbf"
    #args["group"] = "cbf"
    #main(**args)
    #tags = ["test"]
    #from thesis.util import tf_events_to_plot
    #for tag in tags:
    #    tf_events_to_plot(dirss=["test"], #"standard"
    #                      tags=[tag],
    #                      x_label='Episode',
    #                      y_label='',
    #                      width=5,
    #                      height=2.5,
    #                      episode_length=100,
    #                      window_size=11,
    #                      save_as=f"pdfs/{tag.split('/')[1]}")

    os.system("say The program finished.")

    ########################

    #args["train"] = True
    #args['iterations'] = 2
    #args['total_timesteps'] = 1e4
    #args["name"]="NoSafety"
    #args["safety"] = "shield"

    #name = "test"
    #args["group"] = name
    #args["name"] = name
    #args["name"] = "Shield_Punish"
    #args["safety"] = "mask"
    #args["name"] = "Mask"
    #args["name"] = "Mask_Punish"
    # args["safety"] = "cbf"

    #args['train'] = True
    #args['safety'] = 'mask'
    #args['name'] = 'noSafetyTest'

    #args['rollout'] = True
    #args['render'] = True
    #args['safety'] = 'mask'

    #args['name'] = 'maskTest'

    #main(**args)