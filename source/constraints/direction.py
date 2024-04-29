from .constraint import ConstraintProposition, GeometricObject
from ..utils.scaled_sigmoid import scaled_sigmoid
import numpy as np
from typing import Iterable


class DirectionTowards(ConstraintProposition):
    """First argument must look at second argument in terms of z-axis rotation.
    """
    def __init__(self, arguments: Iterable[GeometricObject]):
        """Init method."""
        super().__init__(arguments)

    @staticmethod
    def arity():
        return 2 # binary

    def badness(self):
        """See superclass documentation for details.
        """
        displacement_vector = (self.arguments[1].loc - self.arguments[0].loc)[:2]
        displacement_vector = displacement_vector / np.linalg.norm(displacement_vector)
        rad = np.deg2rad(self.arguments[0].rot[2])
        z_rot_vector = np.array([np.sin(rad), np.cos(rad)])
        return (1 + (z_rot_vector @ displacement_vector)) / 2
    
    def __str__(self) -> str:
        """To string method.
        """
        return f"Object {str(self.arguments[0])} must be directed at object {str(self.arguments[1])} in z-rotation."