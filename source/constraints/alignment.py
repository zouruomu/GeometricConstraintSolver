from .constraint import ConstraintProposition, GeometricObject
from ..utils.scaled_sigmoid import scaled_sigmoid
import numpy as np
from typing import Iterable


class TranslationalAlignment(ConstraintProposition):
    """A proposition that represents a notion of translational alignment between objects.

    NOTE: This constraint proposition is flexible-arity. It can be instantiated with any
          number of arguments. The output is simply the mean absolute deviation along the
          specified dimension at the specified location.

    Inherits from the Object abstract class. Alignment is defined by the self.dimension and
    self.location attriutes. The former represents the dimension along which alignment should be
    considered, and the latter represents which part of the objects one wants aligned (the
    left-most point? The center? Etc.). For example, if one wishes to articulate the constraint
    that "the top-most points on the given objects are aligned", one would set dimension="z"
    and location="bounding_max". If one wants to have the objects with centers aligned on the
    X-axis, one would set dimension="x" and location="center".

    Attributes:
        Superclass attributes.
        dimension: str, one of "x", "y", "z", or "random", the dimension on which to assert alignment.
        location: str, one of "bounding_min", "bounding_max", or "center", where to check alignment.
    """
    def __init__(self, arguments: Iterable[GeometricObject], dimension="random", location="random"):
        """Init method."""
        super().__init__(arguments)

        # check to make sure dimension and location are valid
        if dimension not in ["x", "y", "z"]:
            raise ValueError(f"Argument 'dimension' must be one of 'x', 'y', 'z'.")
        if location not in ["bounding_min", "bounding_max", "center"]:
            raise ValueError(f"Argument 'dimension' must be one of 'bounding_min', 'bounding_max', 'center'.")

        # store attributes
        self.dimension = dimension
        self.location = location

    @staticmethod
    def arity():
        return (2, -1)

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as the mean absolute deviation between all objects
        passed along the specified dimension and location.
        """
        # convert str dimension to index
        dim_idx = 0
        if self.dimension == "x": dim_idx = 0
        elif self.dimension == "y": dim_idx = 1
        elif self.dimension == "z": dim_idx = 2

        # get values to compare for all objects
        values_to_compare = []
        for argument in self.arguments:
            if self.location == "bounding_min":
                values_to_compare.append(argument.get_bounding_intervals()[dim_idx][0])
            elif self.location == "bounding_max":
                values_to_compare.append(argument.get_bounding_intervals()[dim_idx][1])
            elif self.location == "center":
                values_to_compare.append(argument.loc[dim_idx])

        values_to_compare = np.array(values_to_compare)
        return scaled_sigmoid(np.std(values_to_compare))
    
    def __str__(self) -> str:
        """To string method.
        """
        if self.location == "center": loc = "centermost"
        if self.location == "bounding_min": loc = "smallest"
        if self.location == "bounding_max": loc = "largest"
        obj_names = ""
        for obj in self.arguments[:-1]:
            obj_names += f"{str(obj)}, "
        obj_names += f"and {str(self.arguments[-1])}"
        return f"{obj_names} must be aligned along their {loc} {self.dimension}-axis values."
    
    @classmethod
    def random(cls, arguments):
        dimension = np.random.choice(["x", "y", "z"])
        location = np.random.choice(["bounding_min", "bounding_max", "center"])
        
        return cls(arguments, dimension=dimension, location=location)
    
    def save_kwargs(self) -> dict:
        return {"dimension": self.dimension, "location": self.location}
    

# class RotationalAlignment(ConstraintProposition):
#     """A proposition that represents a notion of rotational alignment between objects.
#
#     NOTE: This constraint proposition is flexible-arity. It can be instantiated with any
#           number of arguments.
#
#     Inherits from the Object abstract class. Alignment is defined by the self.dimension attribute.
#     This constraint's badness is simply the mean absolute deivation of the angles (for specified
#     dimension) of all argument objects.
#
#     Attributes:
#         Superclass attributes.
#         dimension: str, one of "x", "y", or "z", the dimension on which to assert alignment.
#     """
#     def __init__(self, arguments: Iterable[GeometricObject], dimension="z"):
#         """Init method."""
#         super().__init__(arguments)
#
#         # check to make sure dimension and location are valid
#         if dimension not in ["x", "y", "z"]:
#             raise ValueError(f"Argument 'dimension' must be one of 'x', 'y', or 'z'.")
#
#         # store attributes
#         self.dimension = dimension
#
#     @staticmethod
#     def arity():
#         return None # flexible-arity
#
#     def badness(self):
#         """See superclass documentation for details.
#         
#         In this case, badness is calculated as the mean absolute deviation between all objects
#         passed along the specified dimension.
#         """
#         # convert str dimension to index
#         dim_idx = 0
#         if self.dimension == "x": dim_idx = 0
#         elif self.dimension == "y": dim_idx = 1
#         elif self.dimension == "z": dim_idx = 2
#
#         # get values to compare for all objects
#         values_to_compare = []
#         for argument in self.arguments:
#             values_to_compare.append(argument.rot[dim_idx])
#
#         values_to_compare = np.array(values_to_compare)
#         return scaled_sigmoid(np.std(values_to_compare)/360)
#     
#     def __str__(self) -> str:
#         """To string method.
#         """
#         obj_names = ""
#         for obj in self.arguments[:-1]:
#             obj_names += f"{str(obj)}, "
#         obj_names += f"and {str(self.arguments[-1])}"
#         return f"{obj_names} must have the same rotation in the {self.dimension}-axis."