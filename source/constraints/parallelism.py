from .constraint import ConstraintProposition, GeometricObject
from ..utils.scaled_sigmoid import scaled_sigmoid
import numpy as np
from typing import Iterable


class Parallelism(ConstraintProposition):
    """The TWO arguments must be parallel.
    """
    def __init__(self, arguments: Iterable[GeometricObject]):
        """Init method."""
        super().__init__(arguments)

    @staticmethod
    def arity():
        return 2 # binary

    def badness(self):
        theta0 = self.arguments[0].rot[2] / 180 * np.pi
        theta1 = self.arguments[1].rot[2] / 180 * np.pi
        diff = abs(theta0 - theta1)
        return 1 - np.cos(diff)
    
    def __str__(self) -> str:
        """To string method.
        """
        return f"{str(self.arguments[0])} and {str(self.arguments[1])} must be parallel in z-rotation."