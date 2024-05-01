from .constraint import ConstraintProposition, GeometricObject, Cuboid
from ..utils.scaled_sigmoid import scaled_sigmoid
import numpy as np
from typing import Iterable


class Cover(ConstraintProposition):
    """
    arguments[1] must be covered by arguments[0].
    """
    def __init__(self, arguments: Iterable[Cuboid]):
        """Init method."""
        super().__init__(arguments)

    @staticmethod
    def arity():
        return 2

    def badness(self):
        """See superclass documentation for details.
        """
        obj0, obj1 = self.arguments
        rect0, rect1 = obj0.to_2d_rect(), obj1.to_2d_rect()
        intersection = rect0.intersection(rect1)

        score = 1 - intersection.area / rect1.area
        return score
    
    
    def __str__(self) -> str:
        obj0, obj1 = self.arguments
        return f"{obj0.name} covers {obj1.name}, which means {obj1.name}'s 2D projection must be inside {obj0.name}'s 2D projection."
    
    @classmethod
    def random(cls, arguments):
        raise ValueError("Random generation of NoOverlap constraint is not supported.")