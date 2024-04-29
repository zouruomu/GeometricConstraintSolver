
import numpy as np
from scipy.spatial import distance_matrix

from .constraint import ConstraintProposition, GeometricObject


class Direction(ConstraintProposition):
    """
    the second object's direction to the first object
    """
    def __init__(self, arguments, direction: str = "random"):
        super().__init__(arguments)
        assert direction in ["left", "right", "up", "down", "front", "back", "random"], "Invalid direction."
        if direction == "random":
            direction = np.random.choice(["left", "right", "front", "back"]) #FIXME: add z-axis
        self.direction = direction

    @staticmethod
    def arity():
        return 2

    def badness(self):
        return int(self._badness())
            
    def _badness(self) -> bool:
        obj1, obj2 = self.arguments
        if self.direction == "left":
            return obj2.loc[0] < obj1.loc[0]
        elif self.direction == "right":
            return obj2.loc[0] > obj1.loc[0]
        elif self.direction == "up":
            raise NotImplementedError
        elif self.direction == "down":
            raise NotImplementedError
        elif self.direction == "front":
            return obj2.loc[1] < obj1.loc[1]
        elif self.direction == "back":
            return obj2.loc[1] > obj1.loc[1]
        else:
            raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{str(self.arguments[1])} must be on the {self.direction} of {str(self.arguments[0])}."