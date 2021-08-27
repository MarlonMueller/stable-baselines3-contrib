from typing import Optional

import os
import numpy as np
import matlab.engine
from numpy import pi

from sb3_contrib.common.safety.safe_region import SafeRegion


class PendulumRegionOfAttraction(SafeRegion):

    def __init__(
            self,
            A: Optional[np.ndarray] = None,
            b: Optional[np.ndarray] = None,
            vertices: Optional[np.ndarray] = None
    ):
        super(PendulumRegionOfAttraction, self).__init__(A, b, vertices)


    @staticmethod
    def compute_roa():
        """
        """
        eng = matlab.engine.start_matlab()
        dir = os.path.dirname(os.path.abspath(__file__))
        path_matlab = eng.genpath(f'{dir}/matlab')
        eng.addpath(path_matlab, nargout=0)
        b, v = eng.regionOfAttraction(nargout=2)
        eng.quit()

        v = np.asarray(v).T
        # Swap elements for path ordering, TODO: In python
        v[[2, 3]] = v[[3, 2]]

        return np.asarray(b), v

    def __contains__(self, state):

        #Note: Check could probably done directly via max_torque

        # Derivation
        # y = mx +t; x = (y-t)/m
        # m = #RightLow - #RightUp = (- 2*max_thdot)/(pi + 0.785398163397448) = -3
        # t = y - mx =  -max_thdot + 3*pi = 3.5342917352885177
        # x = (3.5342917352885177)/(3) = 1.1780972450961726
        # (1) = [1.1780972450961726, 0]
        # (2) = [-pi + 1.1780972450961726, max_thdot + 0] = [-1.9634954084936205, max_thdot]
        # det(1,2) = 1.1780972450961726 * max_thdot = 6.939565594515956
        # -> det(S, 1)/det(1,2) && det(S,2)/det(1,2)

        max_thdot = 5.890486225480862
        vertices = np.array([
            [-pi, max_thdot],  # LeftUp
            [-0.785398163397448, max_thdot],  # RightUp
            [pi, -max_thdot],  # RightLow
            [0.785398163397448, -max_thdot]  # LeftLow
        ])



        theta, thdot = state
        cutoff = 1 + 1e-10
        det_12 = 6.939565594515956
        max_thdot = 5.890486225480862
        if (abs((-thdot * 1.1780972450961726)/det_12) <= cutoff and
                abs((theta * max_thdot + thdot * 1.9634954084936205)/det_12) <= cutoff):
             return True
        return False

    def linspace(self, num_theta=5, num_thdot=3):

        num_theta -= 1
        num_thdot -= 1

        if not (num_theta & (num_theta -1) == 0 and num_theta != 0 and
                num_thdot & (num_thdot -1) == 0 and num_thdot != 0):
            raise ValueError(f'Choose (num_theta-1) and (num_thdot-1) as powers of two')

        fac_theta = -1.
        fac_thdot = 1.
        dfac_theta = 2/num_theta
        dfac_thdot = 2/num_thdot

        states = []
        max_thdot = 5.890486225480862
        while fac_thdot >= -1:
            while fac_theta <= 1:
                theta = fac_theta * 1.1780972450961726 - fac_thdot * 1.9634954084936205
                thdot = fac_thdot * max_thdot
                states.append([theta, thdot])
                fac_theta += dfac_thdot
            fac_theta = -1.
            fac_thdot -= dfac_theta

        return states


    def sample(self):
        fac_theta, fac_thdot = self.rng.uniform(-1., 1., 2)
        max_thdot = 5.890486225480862
        theta = fac_theta * 1.1780972450961726 - fac_thdot * 1.9634954084936205
        thdot = fac_thdot * max_thdot
        return [theta, thdot]
