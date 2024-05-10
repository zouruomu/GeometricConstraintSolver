from ..constraint import ConstraintProposition, GeometricObject, Cuboid
from ...utils.scaled_sigmoid import scaled_sigmoid
import numpy as np
from typing import Iterable


from ..proximity import Proximity
from ..target import Target


class BackToBack(ConstraintProposition):
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
        badness = Proximity([obj0, obj1]).badness() + (2 - Target([obj0, obj1]).badness() - Target([obj1, obj0]).badness()) / 2

        return badness / 2
    
    
    def __str__(self) -> str:
        obj0, obj1 = self.arguments
        return f"{obj0.name} and {obj1.name} are back to back"
    
    @classmethod
    def random(cls, arguments):
        return cls(arguments)