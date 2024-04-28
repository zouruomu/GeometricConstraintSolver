import numpy as np
from scipy.spatial import distance_matrix
from copy import deepcopy

from .constraint import ConstraintProposition, GeometricObject


class Symmetry(ConstraintProposition):
    """A proposition that specifies that the TWO passed arguments must be symmetrical around a point.
    A list objects should be symetrical around their median point.

    NOTE: This constraint proposition is binary ("_B").

    Inherits from the Object abstract class. Symmetry around a point is defined as having the vector
    from argument_1 to the point be the opposite of the one from argument_2 to the point.

    Attributes:
        Superclass attributes.
        point: a list of form [x,y,z] specifying the point around which the arguments are symmetrical.
    """
    def __init__(self, arguments, clamp=False):
        super().__init__(arguments)
        self.clamp = clamp

    @property
    def arity(self):
        return None

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
        assert (badness >= 0).all() and (badness <= 1).all(), "Badness should be between 0 and 1."

        return badness
