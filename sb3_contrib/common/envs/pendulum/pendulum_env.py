from typing import Union, Optional, Tuple

from os import path
from gym import Env
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from sb3_contrib.common.safety.safe_region import SafeRegion

import numpy as np
from numpy import sin, cos, pi

class PendulumEnv(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
            self,
            safe_region: Optional[Union[SafeRegion, np.ndarray]] = None,
    ):

        # Length
        self.l = 1.
        # Mass
        self.m = 1.
        # Gravity
        self.g = 9.81
        # Timestep
        self.dt = .05

        self.rng = np.random.default_rng()

        self.action_to_torque = {}

        # TODO
        self.safe_region = safe_region

        self.reset()


    def reset(self) -> GymObs:

        # Start at theta=0; thdot=0
        self.state = np.array([0,0])
        # Start within safe space
        # self.state = self.safe_region.sample()

        self.last_action = None

        return self._get_obs(*self.state)

    def step(self, action: Union[float, np.ndarray]) -> GymStepReturn:
        theta, thdot = self.state
        self.state[:] = self._dynamics(theta, thdot, action)
        self.last_action = action
        return self._get_obs(theta, thdot), self._get_reward(theta, thdot, action), False, {}

    def _dynamics(self, theta: float, thdot: float, torque: float) -> Tuple[float, float]:
        thdot += self.dt * ((self.g / self.l) * sin(theta) + 1. / (self.m * self.l ** 2) * torque)
        theta += self.dt * thdot

    def _get_obs(self, theta, thdot) -> GymObs:
        return np.array([cos(theta), sin(theta), thdot])

    def _get_reward(self, theta: float, thdot: float, action: Union[int, np.ndarray]) -> float:
        return -(self._norm_theta(theta) ** 2 + .1 * thdot ** 2 + .001 * (action ** 2))

    def _norm_theta(self, theta: float) -> float:
        return ((theta + pi) % (2 * pi)) - pi

    def is_safe_action(self, safe_region, action):
        theta, thdot = self.state
        state = self._dynamics(theta, thdot, action)
        return state in safe_region

    def safe_action(self, safe_region):
        # LQR controller
        gain_matrix = [19.670836678497427, 6.351509533724627]
        return -np.dot(gain_matrix, self.state)

    def render(self, mode: str = "human"): #-> TODO:

        if self.viewer is None:

            #TODO
            self.safety_violation = False

            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.make_capsule(1, .05)
            rod.set_color(0, 0, 0)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            self.mass = rendering.make_circle(.025)
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

        if not self.safety_violation and self.safe_region is not None:
            if self.state not in self.safe_region:
                self.safety_violation = True
                self.mass.set_color(227 / 255, 114 / 255, 34 / 255)

        if self.last_action:
            self.imgtrans.scale = (self.last_action / 2, abs(self.last_action) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')