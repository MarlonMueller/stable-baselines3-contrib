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

        # theta_roa = 3.092505268377452
        # vertices = np.array([
        #     [-theta_roa, 12.762720155208534],  # LeftUp
        #     [theta_roa, -5.890486225480862],  # RightUp
        #     [theta_roa, -12.762720155208534],  # RightLow
        #     [-theta_roa, 5.890486225480862]  # LeftLow
        # ])

        # Derivation
        # y = mx +t; x = (y-t)/m; m = x/(y-t)
        # m = (-5.890486225480862 - 12.762720155208534) / (2*theta_roa) = -3.0158730158730167
        # t = y - mx = 12.762720155208534 - 3.0158730158730167 * theta_roa = 3.436116964863835
        # (1) = [0, 3.436116964863835]
        # (2) = [-theta_roa, 12.762720155208534-3.436116964863835] =  [-theta_roa, 9.326603190344699]
        # det(1,2) = 3.436116964863835 * theta_roa = 10.62620981660255
        # -> det(S, 1)/det(1,2) && det(S,2)/det(1,2)

        #dt^2 / 2 = (1/800)
        #9.81*u
        #(1/800)*(9.81**2/2 + 9.81*u)


        theta, thdot = state
        cutoff = 1 + 1e-10
        det_12 = 10.62620981660255
        max_theta = 3.092505268377452

        if (abs((theta * 3.436116964863835)/det_12) <= cutoff and
                abs((theta * 9.326603190344699 + thdot * max_theta)/det_12) <= cutoff):
             return True
        return False

    def linspace(self, num_theta=5, num_thdot=3):

        num_theta -= 1
        num_thdot -= 1

        if not (num_theta & (num_theta -1) == 0 and num_theta != 0 and
                num_thdot & (num_thdot -1) == 0 and num_thdot != 0):
            raise ValueError(f'Choose (num_theta-1) and (num_thdot-1) as powers of two')

        fac_1 = -1.
        fac_2 = 1.
        dfac_theta = 2/num_theta
        dfac_thdot = 2/num_thdot

        states = []
        max_theta = 3.092505268377452
        while fac_2 >= -1:
            while fac_1 <= 1:
                theta = fac_2 * (-max_theta)
                thdot = fac_1 * 3.436116964863835 + fac_2 * 9.326603190344699
                states.append([theta, thdot])
                fac_1 += dfac_thdot
            fac_1 = -1.
            fac_2 -= dfac_theta

        return states


    def sample(self):
        fac_1, fac_2 = self.rng.uniform(-1., 1., 2)
        max_theta = 3.092505268377452
        theta = fac_2 * (-max_theta)
        thdot = fac_1 * 3.436116964863835 + fac_2 * 9.326603190344699
        return [theta, thdot]
