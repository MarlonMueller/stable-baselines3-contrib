from typing import Optional

import os
import numpy as np
from sb3_contrib.common.safety.safe_region import SafeRegion

class PendulumRegionOfAttraction(SafeRegion):
    """
    Class to manage a precomputed region of attraction of the inverted pendulum task
    The region of attraction acts as the safety metric for this benchmark

    @param A: Half-space representation (Ax <= b)
    @param b: Half-space representation (Ax <= b)
    @param vertices: Vertex representation of the safe region
    """

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
        @return: safe region
        """
        import matlab.engine
        eng = matlab.engine.start_matlab()
        dir = os.path.dirname(os.path.abspath(__file__))
        path_matlab = eng.genpath(f'{dir}/matlab')
        eng.addpath(path_matlab, nargout=0)
        b, v = eng.regionOfAttraction(nargout=2)
        eng.quit()
        v = np.asarray(v).T
        # Swap elements for path ordering
        v[[2, 3]] = v[[3, 2]]
        return np.asarray(b), v

    def __contains__(self, state):
        """
        @param state: State of the variables that need to be in the safe region
        @return: Whether the state is inside or outside of the safe region
        """

        """ Derivation
        y = mx +t; x = (y-t)/m; m = x/(y-t)
        m = (-5.890486225480862 - 12.762720155208534) / (2*theta_roa) = -3.0158730158730167
        t = y - mx = 12.762720155208534 - 3.0158730158730167 * theta_roa = 3.436116964863835
        (1) = [0, 3.436116964863835]
        (2) = [-theta_roa, 12.762720155208534-3.436116964863835] =  [-theta_roa, 9.326603190344699]
        det(1,2) = 3.436116964863835 * theta_roa = 10.62620981660255
        -> det(S, 1)/det(1,2) && det(S,2)/det(1,2)
        """

        theta, thdot = state
        cutoff = 1 + 1e-10
        det_12 = 10.62620981660255
        max_theta = 3.092505268377452

        if (abs((theta * 3.436116964863835)/det_12) <= cutoff and
                abs((theta * 9.326603190344699 + thdot * max_theta)/det_12) <= cutoff):
             return True
        return False

    def linspace(self, num_theta=5, num_thdot=3):
        """
        Generates num_theta * num_thdot evenly spaced states within the safe region
        @param num_theta
        @param num_thdot
        @return: states
        """
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
        """
        @return: sample from the safe region
        """
        fac_1, fac_2 = self.rng.uniform(-1., 1., 2)
        max_theta = 3.092505268377452
        theta = fac_2 * (-max_theta)
        thdot = fac_1 * 3.436116964863835 + fac_2 * 9.326603190344699
        return [theta, thdot]
