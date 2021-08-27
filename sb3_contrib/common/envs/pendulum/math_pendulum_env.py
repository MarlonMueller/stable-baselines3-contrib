from typing import Union, Optional, Tuple

from os import path
import numpy as np
from gym import Env
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym.spaces import Box
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.common.safety.safe_region import SafeRegion

import numpy as np
from numpy import sin, cos, pi

class MathPendulumEnv(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    @staticmethod
    def is_safe_action(env: Env, safe_region: SafeRegion, action: float):
        theta, thdot = env.state
        state = env.dynamics(theta, thdot, action)
        return state in safe_region

    @staticmethod
    def safe_action(env: Env, safe_region: SafeRegion, action: float):
        # LQR controller
        # TODO: Maybe restrict torque here? Visuals.
        #gain_matrix = [19.670836678497427, 6.351509533724627]
        gain_matrix = [18.224698834878474, 5.874625145435321]

        #TODO: GETATTR NOT IMPLEMENTED IN VECENVS
        if isinstance(env, DummyVecEnv):
            return -np.dot(gain_matrix, env.get_attr("state")[0])
        else:
            return -np.dot(gain_matrix, env.state)

    def __init__(self, init=None, reward=None):

        # Length
        self.l = 1.
        # Mass
        self.m = 1.
        # Gravity
        self.g = 9.81
        # Timestep
        self.dt = .05

        self.init = init #TODO
        self.reward = reward

        self.rng = np.random.default_rng()

        #TODO: Remove
        from gym.spaces import Discrete
        #self.action_space = Discrete(15)

        max_torque = 30.898877999566082
        self.action_space = Box(
            low=-max_torque,
            high=max_torque,
            shape=(1,),
            dtype=np.float32
        )

        obs_high = np.array([1., 1., np.inf], dtype=np.float32)
        self.observation_space = Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        self.last_action = None
        self.viewer = None

        #TODO: Remove
        from thesis.pendulum_roa import PendulumRegionOfAttraction
        max_thdot = 5.890486225480862
        vertices = np.array([
            [-pi, max_thdot],  # LeftUp
            [-0.785398163397448, max_thdot],  # RightUp
            [pi, -max_thdot],  # RightLow
            [0.785398163397448, -max_thdot]  # LeftLow
        ])
        self._safe_region = PendulumRegionOfAttraction(vertices=vertices)

        self.reset()


    def reset(self) -> GymObs:

        # Start at theta=0; thdot=0

        if self.init is not None and self.init == "random":
            self.state = np.asarray(self._safe_region.sample())
        else:
            self.state = np.array([0, 0])

        self.last_action = None

        return self._get_obs(*self.state)

    def step(self, action: Union[float, np.ndarray]) -> GymStepReturn:
        theta, thdot = self.state

        #self.state[:] = self.dynamics(theta, thdot, action)
        theta, thdot =  self.dynamics(theta, thdot, action)
        self.state = np.array([theta, thdot])

        self.last_action = action
        return self._get_obs(theta, thdot), self._get_reward(theta, thdot, action), False, {}

    def dynamics(self, theta: float, thdot: float, torque: float) -> Tuple[float, float]:
        new_theta = theta + self.dt * thdot
        new_thdot = thdot + self.dt * ((self.g / self.l) * sin(theta) + 1. / (self.m * self.l ** 2) * torque)
        return [new_theta, new_thdot]

    def _get_obs(self, theta, thdot) -> GymObs:
        return np.array([cos(theta), sin(theta), thdot])

    def _get_reward(self, theta: float, thdot: float, action: Union[int, np.ndarray]) -> float:
        if self.reward is not None: #TODO: Clean
            # #Opposing -
            return -(.5 * abs(action) + abs(self._norm_theta(theta)) + abs(.1 * thdot))
        else: #Safety
            det_12 = 6.939565594515956
            max_thdot = 5.890486225480862
            return -(4.5 * max(abs((-thdot * 1.1780972450961726) / det_12),
                                 abs((theta * max_thdot + thdot * 1.9634954084936205) / det_12))) ** 2

    def _norm_theta(self, theta: float) -> float:
        return ((theta + pi) % (2 * pi)) - pi

    def close(self):
        #TODO: Pot. remove
        #if self.eng in locals():
         #   self.eng.quit()

        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode: str = "human", safe_region: SafeRegion = None): #-> TODO:

        if self.viewer is None:

            self.safety_violation = False

            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.make_capsule(1, .05)
            rod.set_color(0, 0, 0)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            self.mass = rendering.make_circle(.15)
            self.mass.set_color(152 / 255, 198 / 255, 234 / 255)
            self.mass_transform = rendering.Transform()
            self.mass.add_attr(self.mass_transform)
            self.viewer.add_geom(self.mass)

            axle = rendering.make_circle(.025)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            self.imgtrans.scale = (0., 0.)

        self.viewer.add_onetime(self.img)

        thetatrans = -self.state[0] + pi / 2
        self.pole_transform.set_rotation(thetatrans)
        self.mass_transform.set_translation(cos(thetatrans), sin(thetatrans))

        if not self.safety_violation and self._safe_region is not None:
            if self.state not in self._safe_region:
                self.safety_violation = True
                self.mass.set_color(227 / 255, 114 / 255, 34 / 255)

        if self.last_action:
            self.imgtrans.scale = (self.last_action / 6, abs(self.last_action) / 6)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')