from typing import Optional, Tuple
from pypoman import compute_polytope_halfspaces
from pypoman import compute_polytope_vertices
import numpy as np

class SafeRegion():
    """
    Class to manage the safe region, i.e., the variables that need to be rendered safe.
    Support for convex polytopes in vertex or half-space representation.

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
        if vertices is None and (A is None or b is None):
            raise ValueError("No representation provided")

        if vertices is not None:
            self._A, self._b = compute_polytope_halfspaces(vertices)
            self._v = vertices
        else:
            self._A = A
            self._b = b
            self._v = None

        self.rng = np.random.default_rng()

    def __contains__(self, state):
        """
        @param state: State of the variables that need to be in the safe region
        @return: Whether the state is inside or outside of the safe region
        """
        """
        Add insignificant value to compensate for numerical floating point calculations.
        Otherwise the polytope-defining vertices might be outside of the safe region.
        Remove value if such scales are still significant for the system.
        """
        return np.all(np.matmul(self._A, state) <= self._b + 1e-10)

    @property
    def vertices(self) -> np.ndarray:
        """
        @return: Vertex representation of the safe region
        """
        if self._v is None:
            self._v = compute_polytope_vertices(self._A, self._b)
        return self._v

    @vertices.setter
    def vertices(self, vertices: np.ndarray) -> None:
        """
        @param vertices: Vertex representation of the safe region
        """
        self._v = vertices

    @property
    def halfspaces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        @return: Half-space representation (A,b) of the safe region
        """
        return self._A, self._b

    @halfspaces.setter
    def halfspaces(self, Ab:Tuple[np.ndarray, np.ndarray]) -> None:
        """
        @param Ab: Half-space representation (A,b) of the safe region
        """
        self._A, self._b = Ab

    @classmethod
    def compute_safe_region(cls):
        """
        @return: safe region
        """
        raise NotImplementedError

    def linspace(self):
        """
        @return: evenly spaced states within the safe region
        """
        raise NotImplementedError

    def sample(self):
        """
        @return: sample from the safe region
        """
        #TODO: Implement sample functionality for convex polytopes
        raise NotImplementedError
