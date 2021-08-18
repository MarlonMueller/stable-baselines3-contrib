import os, argparse, logging, importlib
from stable_baselines3.common.base_class import BaseAlgorithm
from thesis.util import remove_tf_logs
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

    if kwargs['env_id'] not in [env_spec.id for env_spec in gym.envs.registry.all()]:
        raise KeyError(f"Environment {kwargs['env_id']} is not registered")

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

    # Initialize environment
    #TODO: Safe_region only for visuals
    env = gym.make(kwargs['env_id'], safe_region=safe_region)

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
    parser.add_argument('-e', '--env_id', type=str, default='Pendulum-v0', required=False,
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
    main(**args)