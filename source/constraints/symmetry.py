import numpy as np
from scipy.spatial import distance_matrix
from ..utils.scaled_sigmoid import scaled_sigmoid
from copy import deepcopy

from .constraint import ConstraintProposition, GeometricObject


class Symmetry(ConstraintProposition):
    def __init__(self, arguments, clamp=False):
        super().__init__(arguments)
        self.clamp = clamp

    @staticmethod
    def arity():
        return (3, -1)

    def badness(self):
        x_badness = self._badness_along_axis(self.arguments, "x")
        y_badness = self._badness_along_axis(self.arguments, "y")
        badness = min(x_badness, y_badness)
        if self.clamp:
            return min(badness, 1)
        else:
            return badness
        
    def _badness_along_axis(self, objects, axis):
        """
        Mirror the object along the axis.
        """
        axis_dim = {
            "x": 0,
            "y": 1,
            "z": 2
        }[axis]
        
        # FIXME: mean not median actually
        median_objects_virtual = np.mean([obj.loc for obj in objects], axis=0)
        median = median_objects_virtual[axis_dim]

        all_objects_loc = np.stack([obj.loc for obj in objects])
        mirrored_objects_loc = deepcopy(all_objects_loc)

        mirrored_objects_loc[:, axis_dim] = 2 * median - all_objects_loc[:, axis_dim]
        dists = distance_matrix(mirrored_objects_loc, all_objects_loc).min(-1)
        dists = dists / (np.linalg.norm(mirrored_objects_loc - median_objects_virtual, axis=1) + 1e-7)

        badness = np.mean(dists) 
        badness = np.clip(badness, 0, 1)
        assert (badness >= 0 - 1e-3).all() and (badness <= 1 + 1e-3).all(), f"Badness should be between 0 and 1. get {badness}"

        return badness
    
    def __str__(self) -> str:
        """To string method.
        """
        obj_names = ""
        for obj in self.arguments[:-1]:
            obj_names += f"{str(obj)}, "
        obj_names += f"and {str(self.arguments[-1])}"
        return f"{obj_names} must be symmetrical."
    
    def save_kwargs(self) -> dict:
        return {"clamp": self.clamp}