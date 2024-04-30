
import numpy as np
from scipy.spatial import distance_matrix

from .constraint import ConstraintProposition, GeometricObject


class Direction(ConstraintProposition):
    """
    the second object's direction to the first object
    """
    def __init__(self, arguments, direction: str = "random"):
        super().__init__(arguments)
        assert direction in ["left", "right", "up", "down", "front", "back"], "Invalid direction."
        self.direction = direction

    @staticmethod
    def arity():
        return 2

    def badness(self):
        obj1, obj2 = self.arguments
        rel_vec = (obj2.loc - obj1.loc)[:2]
        rel_vec = rel_vec / (1e-8 + np.linalg.norm(rel_vec))
        if self.direction == "left":
            ref_vec = np.array([1, 0])
        elif self.direction == "right":
            ref_vec = np.array([-1, 0])
        elif self.direction == "up":
            raise NotImplementedError
        elif self.direction == "down":
            raise NotImplementedError
        elif self.direction == "front":
            ref_vec = np.array([0, 1])
        elif self.direction == "back":
            ref_vec = np.array([0, -1])
        else:
            raise NotImplementedError
        return (np.dot(rel_vec, ref_vec) + 1) / 2
    
    def __str__(self) -> str:
        return f"{str(self.arguments[1])} must be on the {self.direction} of {str(self.arguments[0])}."

    @classmethod
    def random(cls, arguments):
        # direction = np.random.choice(["left", "right", "up", "down", "front", "back"])
        direction = np.random.choice(["left", "right", "front", "back"]) #FIXME: add z-axis
        return cls(arguments, direction=direction)
    
    def save_kwargs(self) -> dict:
        return {"direction": self.direction}