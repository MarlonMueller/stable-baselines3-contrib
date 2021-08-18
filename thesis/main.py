import os, argparse, logging, importlib, time
from stable_baselines3.common.utils import configure_logger
from thesis.callbacks.pendulum_train import PendulumTrainCallback
from thesis.callbacks.pendulum_rollout import PendulumRolloutCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from thesis.util import remove_tf_logs, rename_tf_events
import gym
import numpy as np
from numpy import pi

logger = logging.getLogger(__name__)

def main(**kwargs):

    logger.info(f"kargs {kwargs}")

    module = importlib.import_module('stable_baselines3')
    base_algorithm = getattr(module, kwargs['algorithm'])

    if kwargs['name'] == 'DEBUG':
        name = 'DEBUG'
        kwargs['total_timesteps'] = 1e3
        remove_tf_logs(name + '_1', name + '_E_1')
    else:
        name = f"{kwargs['name']}_{kwargs['algorithm']}"

    # Define safe regions
    from sb3_contrib.common.safety.safe_region import SafeRegion
    max_thdot = 5.890486225480862
    vertices = np.array([
        [-pi, max_thdot],  # LeftUp
        [-0.785398163397448, max_thdot],  # RightUp
        [pi, -max_thdot],  # RightLow
        [0.785398163397448, -max_thdot]  # LeftLow
    ])
    safe_region = SafeRegion(vertices=vertices)

    #TODO: Safe_region only for visuals
    from gym.envs.registration import register
    register(
        id='MathPendulum-v0',
        max_episode_steps=100,
        entry_point='sb3_contrib.common.envs.pendulum.math_pendulum_env:MathPendulumEnv',
        # kwargs={'safe_region': True}
    )

    # Initialize environment
    if kwargs['env_id'] not in [env_spec.id for env_spec in gym.envs.registry.all()]:
         KeyError(f"Environment {kwargs['env_id']} is not registered")
    env = gym.make(kwargs['env_id'], safe_region=safe_region)

    from sb3_contrib.common.wrappers import ActionDiscretizer
    from gym.spaces import Discrete

    def tr(action):
        return 2*(action-7)

    env = ActionDiscretizer(
        env=env,
        disc_action_space=Discrete(15),
        transform_fn=tr
        #transform_fn=lambda a : 2*(a-7)
    )

    if 'safety' in kwargs and kwargs['safety'] is not None:

        if kwargs['safety'] == "shield":
            from sb3_contrib.common.wrappers import SafetyShield

            env = SafetyShield(
                env=env,
                safe_region=safe_region,
                is_safe_action_fn="is_safe_action",
                safe_action_fn="safe_action",
                punishment=None
            )

        elif kwargs['safety'] == "cbf":
            from sb3_contrib.common.wrappers import SafetyCBF
            #TODO, f und g as other methods in env?

            env = SafetyCBF(
                env=env,
                safe_region=safe_region,
                punishment=None
            )

        elif kwargs['safety'] == "mask":
            from sb3_contrib.common.wrappers import SafetyMask
            pass

        #TODO: Finish Main/Util Refactor/UtilPendulum/PendulumROA

    if not is_wrapped(env, Monitor):
        env = Monitor(env)

    env = DummyVecEnv([lambda: env])

    if 'train' in kwargs and kwargs['train']:

        model = base_algorithm(MlpPolicy, env, verbose=0, tensorboard_log=os.getcwd() + '/tensorboard/')
        callback = CallbackList([PendulumTrainCallback()])

        model.learn(total_timesteps=kwargs['total_timesteps'],
                    tb_log_name=name,
                    callback=callback)
                    # log_interval=log_interval)

        save_model(name, model)

    elif 'rollout' in kwargs and kwargs['rollout']:


        render = False
        if 'render' in kwargs and kwargs['render']:
            render = True

        model = None
        callback = None

        if name != "DEBUG":

            model = load_model(name + '.zip', base_algorithm)
            model.set_env(env)

            callback = CallbackList([PendulumRolloutCallback()])
            callback.init_callback(model=model)

            configure_logger(verbose=0, tb_log_name=name + '_E',
                             tensorboard_log=os.getcwd() + '/tensorboard/')


        rollout(env, model, num_episodes=1, callback=callback, env_safe_action=False, render=render, sleep=.25)

    if 'env' in locals(): env.close()
    rename_tf_events()

def rollout(env, model=None, num_episodes=1, callback=None, env_safe_action=False, render=False, sleep=0.):

    """ 'Rollout' policy.

    For a given number of episodes, the environment is stepped through.
    If no model is given, one can sample from the action_space or use the environment's safe action.

    Notes
    ----------
    Adaption of SB3's evaluate_policy: Callbacks of type BaseCallback can be passed.
    The usage of _on_rollout_start(self) is redefined compared to its use in BaseAlgorithms ('collecting rollouts').
    As a result, '_on_rollout_start' is called for each new episode rollout.

    Due to initial state logging (&incremented max_steps) each episode will result
    in env_spec.max_episode_steps + 1 entries for env_spec.max_episode_steps + 2 states.

    This results in 'clean' logs for the first episode and has to be considered for the following if num_episodes > 1.
    For further information, see main().

    Parameters
    ----------
    env
    model
    num_episodes
        Number of episodes to be evaluated.
    callback : BaseCallback
    env_safe_action
        If no model is given and true, the environment's safe action are taken.
    render
    sleep
        Time in seconds between each step.
    """
    is_vec_env = isinstance(env, VecEnv)
    if is_vec_env:
        if env.num_envs != 1:
            logger.warning(f"You must pass only one environment when using this function")
        is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)

    if not is_monitor_wrapped:
        # Note: Fails due to VecEnvs not storing forwarding reward range
        # env = Monitor(env)
        logger.warning(f"Evaluation environment is not wrapped with a ``Monitor`` wrapper.")

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
                env.render()

            if model is not None:
                action, state = model.predict(obs, state=state) #deterministic=deterministic
                action = action[0] #Action is dict
                #action = 14
                #print(action)
            elif env_safe_action:
                #TODO: Fix/Easier?
                #action = env.env_method('safe_action')
                action = env.get_attr('safe_action')[0]
            else:
                action = env.action_space.sample()

            # Note: Could still be DummyVecEnv -> e.g. done = [...]
            #obs, reward, done, info = env.step(action) TODO: Continuous sample returns []
            #print(action)
            #action = 14
            obs, reward, done, info = env.step([action])

            #print(info)

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




def save_model(name, model):
    path = os.getcwd() + '/models/'
    os.makedirs(path, exist_ok=True)
    model.save(path + name)  # TODO: Check if save_to_zip_file

def load_model(name, base: BaseAlgorithm):
    path = os.getcwd() + '/models/'
    if os.path.isfile(path + name):
        return base.load(path + name)
    else:
        raise FileNotFoundError(f'No such model {name}')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, default='A2C', required=False,
                        help='RL algorithm')
    parser.add_argument('-e', '--env_id', type=str, default='MathPendulum-v0', required=False,
                        help='ID of a registered environment')
    parser.add_argument('-t', '--total_timesteps', type=int, default=8e4, required=False,  # 400
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
    args['rollout'] = True
    args['render'] = True
    args['safety'] = "shield"

    main(**args)