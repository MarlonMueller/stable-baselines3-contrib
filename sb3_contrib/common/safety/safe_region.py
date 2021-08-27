from typing import Optional, Tuple

from pypoman import compute_polytope_halfspaces
from pypoman import compute_polytope_vertices

import random

import numpy as np

#TODO: Old typing 3.7+
#TODO: Typing np.ndarray

class SafeRegion():

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

    def __contains__(self, item): #TODO
        return np.all(np.matmul(self._A, item) <= self._b + 1e-10)

    @property
    def vertices(self) -> np.ndarray:
        if self._v is None:
            self._v = compute_polytope_vertices(self._A, self._b)
        return self._v

    @vertices.setter
    def vertices(self, vertices: np.ndarray) -> None:
        self._v = vertices

    @property
    def halfspaces(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._A, self._b

    @halfspaces.setter
    def halfspaces(self, Ab:Tuple[np.ndarray, np.ndarray]) -> None:
        self._A, self._b = Ab

    @classmethod
    def compute_safe_region(cls):
        raise NotImplementedError

    def linspace(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError