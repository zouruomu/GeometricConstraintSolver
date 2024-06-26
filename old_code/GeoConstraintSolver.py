import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, basinhopping
import matplotlib
matplotlib.rcParams['figure.dpi'] = 140
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from .constraint import ConstraintProposition

########################################## Constraints: Primitive Constraints ##########################################

class HasLocation_U(ConstraintProposition):
    """A proposition that asserts that an object has a fixed location.

    NOTE: This constraint proposition is unary ("_U").

    Inherits from the Object abstract class. The target location this constraint proposition
    asserts the object to have is defined by a 3-element list or list-like such as [10,5,None].
    The first element is assumed to be for the x dimension, the second for the y, and the third
    for the z. If any of the entries are None, then that dimension is not constrained.

    Attribtues:
        Superclass attributes.
        target_loc: list of 3 elements of form [x,y,z].
    """
    def __init__(self, arguments, target_loc):
        """Init method."""
        super().__init__(arguments)

        # check to make sure target_loc is valid and store
        if len(target_loc) != 3:
            raise ValueError(f"Argument 'target_loc' must have length 3.")
        self.target_loc = target_loc

    def define_arity(self):
        """See superclass documentation for details."""
        return 1 # unary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as how far the the current location is
        from the target location. If any dimension of self.target_loc is None, that
        dimension is not factored into the calculations.
        """
        loc = self.arguments[0].loc # unary
        total_badness = 0
        if self.target_loc[0] is not None: total_badness += abs(self.target_loc[0] - loc[0])
        if self.target_loc[1] is not None: total_badness += abs(self.target_loc[1] - loc[1])
        if self.target_loc[2] is not None: total_badness += abs(self.target_loc[2] - loc[2])
        return total_badness

class AreTranslationallyAligned_F(ConstraintProposition):
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
        dimension: str, one of "x", "y", or "z", the dimension on which to assert alignment.
        location: str, one of "bounding_min", "bounding_max", or "center", where to check alignment.
    """
    def __init__(self, arguments, dimension, location):
        """Init method."""
        super().__init__(arguments)

        # check to make sure dimension and location are valid
        if dimension not in ["x", "y", "z"]:
            raise ValueError(f"Argument 'dimension' must be one of 'x', 'y', or 'z'.")
        if location not in ["bounding_min", "bounding_max", "center"]:
            raise ValueError(f"Argument 'dimension' must be one of 'bounding_min', 'bounding_max', or 'center'.")

        # store attributes
        self.dimension = dimension
        self.location = location

    def define_arity(self):
        """See superclass documentation for details."""
        return None # flexible-arity

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

        # take the mean absolute deviation and output
        values_to_compare = np.array(values_to_compare)
        return np.abs(values_to_compare - values_to_compare.mean()).mean()

class HaveTranslationalDifference_B(ConstraintProposition):
    """A proposition that represents a notion of translational difference between TWO objects.

    NOTE: This constraint proposition is binary ("_B"). This is because the notion of "difference" is
          not readily extendable to multiple objects. If one wants to achieve the effect of having
          two groups that are aligned within themselves and differ from each other by a certain amount,
          use two AreTranslationallyAligned constraints within both groups and one HaveTranslationalDifference_B
          constraint between two given objects one from each group.

    Inherits from the Object abstract class. Translational difference is defined according to the
    same self.dimension/self.location specification as the AreTranslationallyAligned constraint,
    but only operates over two objects and allows the user to specify how far one wants the objects
    apart with the self.target_difference attribute.

    Attributes:
        Superclass attributes.
        target_difference: float, the target difference in the specified location between argument 1 and 2.
        dimension: str, one of "x", "y", or "z", the dimension on which to assert alignment.
        location: str, one of "bounding_min", "bounding_max", or "center", where to check alignment.
    """
    def __init__(self, arguments, target_difference, dimension, location):
        """Init method."""
        super().__init__(arguments)

        # check to make sure dimension and location are valid
        if dimension not in ["x", "y", "z"]:
            raise ValueError(f"Argument 'dimension' must be one of 'x', 'y', or 'z'.")
        if location not in ["bounding_min", "bounding_max", "center"]:
            raise ValueError(f"Argument 'dimension' must be one of 'bounding_min', 'bounding_max', or 'center'.")

        # store attributes
        self.target_difference = target_difference
        self.dimension = dimension
        self.location = location

    def define_arity(self):
        """See superclass documentation for details."""
        return 2 # binary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as the difference between the target and actual differences
        between object 1 and object 2.
        """
        # convert str dimension to index
        dim_idx = 0
        if self.dimension == "x": dim_idx = 0
        elif self.dimension == "y": dim_idx = 1
        elif self.dimension == "z": dim_idx = 2

        # get values to compare for both objects (binary so only two arguments)
        values_to_compare = []
        for argument in self.arguments:
            if self.location == "bounding_min":
                values_to_compare.append(argument.get_bounding_intervals()[dim_idx][0])
            elif self.location == "bounding_max":
                values_to_compare.append(argument.get_bounding_intervals()[dim_idx][1])
            elif self.location == "center":
                values_to_compare.append(argument.loc[dim_idx])

        # take differences and output (binary so only two arguments)
        actual_difference = values_to_compare[0] - values_to_compare[1]
        return abs(self.target_difference - actual_difference)

class HasRotation_U(ConstraintProposition):
    """A proposition that asserts that an object has a fixed rotation.

    NOTE: This constraint proposition is unary ("_U").

    Inherits from the Object abstract class. The target rotation this constraint proposition
    asserts the object to have is defined by a 3-element list or list-like such as [30,None,45].
    The first element is assumed to be for the x dimension, the second for the y, and the third
    for the z. If any of the entries are None, then that dimension is not constrained.

    Attribtues:
        Superclass attributes.
        target_rot: list of 3 elements of form [x,y,z].
    """
    def __init__(self, arguments, target_rot):
        """Init method."""
        super().__init__(arguments)

        # check to make sure target_rot is valid and store
        if len(target_rot) != 3:
            raise ValueError(f"Argument 'target_rot' must have length 3.")
        self.target_rot = target_rot

    def define_arity(self):
        """See superclass documentation for details."""
        return 1 # unary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as how far the the current rotation is
        from the target rotation. If any dimension of self.target_rot is None, that
        dimension is not factored into the calculations.
        """
        rot = self.arguments[0].rot # unary
        total_badness = 0
        if self.target_rot[0] is not None: total_badness += abs(self.target_rot[0] - rot[0])
        if self.target_rot[1] is not None: total_badness += abs(self.target_rot[1] - rot[1])
        if self.target_rot[2] is not None: total_badness += abs(self.target_rot[2] - rot[2])
        return total_badness

class AreRotationallyAligned_F(ConstraintProposition):
    """A proposition that represents a notion of rotational alignment between objects.

    NOTE: This constraint proposition is flexible-arity. It can be instantiated with any
          number of arguments.

    Inherits from the Object abstract class. Alignment is defined by the self.dimension attribute.
    This constraint's badness is simply the mean absolute deivation of the angles (for specified
    dimension) of all argument objects.

    Attributes:
        Superclass attributes.
        dimension: str, one of "x", "y", or "z", the dimension on which to assert alignment.
    """
    def __init__(self, arguments, dimension):
        """Init method."""
        super().__init__(arguments)

        # check to make sure dimension and location are valid
        if dimension not in ["x", "y", "z"]:
            raise ValueError(f"Argument 'dimension' must be one of 'x', 'y', or 'z'.")

        # store attributes
        self.dimension = dimension

    def define_arity(self):
        """See superclass documentation for details."""
        return None # flexible-arity

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as the mean absolute deviation between all objects
        passed along the specified dimension.
        """
        # convert str dimension to index
        dim_idx = 0
        if self.dimension == "x": dim_idx = 0
        elif self.dimension == "y": dim_idx = 1
        elif self.dimension == "z": dim_idx = 2

        # get values to compare for all objects
        values_to_compare = []
        for argument in self.arguments:
            values_to_compare.append(argument.rot[dim_idx])

        # take the mean absolute deviation and output
        values_to_compare = np.array(values_to_compare)
        return np.abs(values_to_compare - values_to_compare.mean()).mean()

class HaveRotationalDifference_B(ConstraintProposition):
    """A proposition that represents a notion of rotational difference between TWO objects.

    NOTE: This constraint proposition is binary ("_B") for the same reason as HaveTranslationalDifference_B.

    Inherits from the Object abstract class. Difference is defined by the self.dimension attribute
    but only operates over two objects and allows the user to specify how far one wants the objects
    pointed apart with the self.difference attribute. Difference is calculated as obj1 - obj2, so
    to assert that 

    Attributes:
        Superclass attributes.
        target_difference: float, the target difference in the specified location between argument 1 and 2.
        dimension: str, one of "x", "y", or "z", the dimension on which to assert alignment.
    """
    def __init__(self, arguments, target_difference, dimension):
        """Init method."""
        super().__init__(arguments)

        # check to make sure dimension and location are valid
        if dimension not in ["x", "y", "z"]:
            raise ValueError(f"Argument 'dimension' must be one of 'x', 'y', or 'z'.")

        # store attributes
        self.target_difference = target_difference
        self.dimension = dimension

    def define_arity(self):
        """See superclass documentation for details."""
        return 2 # binary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as the difference between the target and actual differences
        between object 1 and object 2.
        """
        # convert str dimension to index
        dim_idx = 0
        if self.dimension == "x": dim_idx = 0
        elif self.dimension == "y": dim_idx = 1
        elif self.dimension == "z": dim_idx = 2

        # get values to compare for all objects (binary so only two arguments)
        values_to_compare = []
        for argument in self.arguments:
            values_to_compare.append(argument.rot[dim_idx])

        # take differences and output (binary so only two arguments)
        actual_difference = values_to_compare[0] - values_to_compare[1]
        return abs(self.target_difference - actual_difference)

class HaveDistanceBetween_B(ConstraintProposition):
    """A proposition that specifies that the TWO passed arguments must have a certain distance between them.

    NOTE: This constraint proposition is binary ("_B").

    Inherits from the Object abstract class. Distance between is defined as the euclidean distance between
    the pass objects.

    Attributes:
        Superclass attributes.
        target_distance: positive float, the target distance between the two objects.
    """
    def __init__(self, arguments, target_distance):
        """Init method."""
        super().__init__(arguments)

        # check to make sure target_distance is valid and store
        if target_distance < 0:
            raise ValueError(f"Argument 'target_distance' must be >= 0.")
        self.target_distance = target_distance

    def define_arity(self):
        """See superclass documentation for details."""
        return 2 # binary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as the abs value between the actual distance and target distance.
        """
        actual_distance = np.linalg.norm(self.arguments[0].loc - self.arguments[1].loc)
        return np.abs(self.target_distance - actual_distance)

class IsPointingTowards_U(ConstraintProposition):
    """A proposition that specifies that the ONE passed argument must point towards a point.

    NOTE: This constraint proposition is unary ("_U").

    Inherits from the Object abstract class. Pointing toward a point is defined as having
    the argument object's local x-axis have the same angle as the displacement vector from
    the argument object's location to the target point.

    Attributes:
        Superclass attributes.
        target_point: a list of form [x,y,z] specifying the point to look at.
    """
    def __init__(self, arguments, target_point):
        """Init method."""
        super().__init__(arguments)

        # check to make sure target_point is valid and store
        if len(target_point) != 3:
            raise ValueError(f"Argument 'target_point' must have length 3.")
        self.target_point = target_point

    def define_arity(self):
        """See superclass documentation for details."""
        return 1 # unary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as 1 - cosine(local_x_axis, displacement)
        where displacement is the vector from the object to the target point.
        """
        # get displacement
        displacement = np.array(self.target_point) - self.arguments[0].loc
        displacement_magnitude = np.linalg.norm(displacement)

        # get local x-axis
        global_x_axis = np.array([1,0,0]) # magnitude 1
        rotation_matrix = R.from_euler(angles=self.arguments[0].rot,
                                       seq="xyz", degrees=True).as_matrix()
        local_x_axis = global_x_axis @ rotation_matrix

        # compute cosine of angle between them with law of cosines
        cosine = np.dot(displacement, local_x_axis) / displacement_magnitude # local_x_axis is unit vector
        return 1 - cosine

class AreSymmetrical_B(ConstraintProposition):
    """A proposition that specifies that the TWO passed arguments must be symmetrical around a point.

    NOTE: This constraint proposition is binary ("_B").

    Inherits from the Object abstract class. Symmetry around a point is defined as having the vector
    from argument_1 to the point be the opposite of the one from argument_2 to the point.

    Attributes:
        Superclass attributes.
        point: a list of form [x,y,z] specifying the point around which the arguments are symmetrical.
    """
    def __init__(self, arguments, point):
        """Init method."""
        super().__init__(arguments)

        # check to make sure point is valid and store
        if len(point) != 3:
            raise ValueError(f"Argument 'point' must have length 3.")
        self.point = point

    def define_arity(self):
        """See superclass documentation for details."""
        return 2 # binary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as the l1 distance from the sum of the two displacement
        vectors to the zero vector.
        """
        dis1 = np.array(self.point) - self.arguments[0].loc
        dis2 = np.array(self.point) - self.arguments[1].loc
        return np.abs(dis1 + dis2).sum()

class AreNotOverlapping_B(ConstraintProposition):
    """A proposition that specifies that BOTH passed arguments must not overlap.

    NOTE: This constraint proposition is binary ("_B").

    Inherits from the Object abstract class. Overlapping is defined as having the bounding
    boxes returned by the argument objects' get_bounding_intervals overlapping.

    Attributes:
        Superclass attributes.
    """
    def __init__(self, arguments):
        """Init method."""
        super().__init__(arguments)

    def define_arity(self):
        """See superclass documentation for details."""
        return 2 # binary

    def badness(self):
        """See superclass documentation for details.
        
        In this case, badness is calculated as the overlap area if there is an overlap
        and 0 if there is no overlap. The calculations are based on the principle that
        2 axis-aligned boxes (bounding boxes are axis-aligned) overlap if and only if
        the projections to all axes overlap.
        """
        # get bounding intervals (binary so only two arguments)
        [xmin1, xmax1], [ymin1, ymax1], [zmin1, zmax1] = self.arguments[0].get_bounding_intervals()
        [xmin2, xmax2], [ymin2, ymax2], [zmin2, zmax2] = self.arguments[1].get_bounding_intervals()

        # For any axis, we have two bounding intervals for the two arguments, and there are six cases:
        #     1. Interval 1 is strictly before interval 2.
        #     2. Interval 2 is strictly before interval 1.
        #     3. Intervals 1 and 2 overlap but one does not fully contain the other, interval 1 is first.
        #     4. Intervals 1 and 2 overlap but one does not fully contain the other, interval 2 is first.
        #     5. Intervals 1 and 2 overlap and interval 1 fully contains interval 2.
        #     6. Intervals 1 and 2 overlap and interval 2 fully contains interval 1.
        
        # In cases 1 and 2, either (max1 - min2) OR (max2 - min1) will be negative/zero and overlap area is 0.
        # In cases 3 and 4, both (max1 - min2) AND (max2 - min1) will be positive and overlap area is the min between them.
        # In cases 5 and 6, both (max1 - min2) AND (max2 - min1) will be positive and overlap area is the shorter interval.

        # HOWEVER, for cases 5 and 6, if we try to minimize the overlap area, for small steps in either
        # direction, there will not be changes in the badness until the intervals are no longer overlapping.
        # This makes minimizing the overlap area challenging. Therefore, for ease of optimization, it suffices
        # to treat cases 5 & 6 the same as 3 & 4 and let the badness always be whichever one of (max1 - min2)
        # and (max2 - min1) is smaller. In the one-interval-within-another situation of cases 5 and 6, this
        # is equivalent to letting the algorithm push the inner interval in whichever direction is closest
        # to the edge of the outer interval.
        
        if xmax1 - xmin2 <= 0 or xmax2 - xmin1 <= 0:
            x_overlap = 0 # no overlap, cases 1 and 2
        else:
            x_overlap = min(xmax1 - xmin2, xmax2 - xmin1) # has overlap, cases 3 - 6

        if ymax1 - ymin2 <= 0 or ymax2 - ymin1 <= 0:
            y_overlap = 0 # no overlap, cases 1 and 2
        else:
            y_overlap = min(ymax1 - ymin2, ymax2 - ymin1) # has overlap, cases 3 - 6

        if zmax1 - zmin2 <= 0 or zmax2 - zmin1 <= 0:
            z_overlap = 0 # no overlap, cases 1 and 2
        else:
            z_overlap = min(zmax1 - zmin2, zmax2 - zmin1) # has overlap, cases 3 - 6

        total_overlap = x_overlap * y_overlap * z_overlap
        return total_overlap

########################################## Constraints: Composite Constraints ##########################################

class IsUpright_U(ConstraintProposition):
    """A unary ("_U") constraint asserting that the passed argument has rotation [0,0,0].
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 1 # unary

    def badness(self):
        return HasRotation_U(arguments=self.arguments, target_rot=[0,0,0]).badness()

class IsAtOrigin_U(ConstraintProposition):
    """A unary ("_U") constraint asserting that the passed argument has location [0,0,0].
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 1 # unary

    def badness(self):
        return HasLocation_U(arguments=self.arguments, target_loc=[0,0,0]).badness()

class AreProximal_B(ConstraintProposition):
    """A binary ("_B") constraint asserting that the TWO passed objects are close but not overlapping.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 2 # binary

    def badness(self):
        res = (HaveDistanceBetween_B(arguments=self.arguments, target_distance=0).badness()
               + AreNotOverlapping_B(arguments=self.arguments).badness())
        return res

class HaveSameRotation_F(ConstraintProposition):
    """A flexible-arity ("_F") constraint asserting that ALL passed objects must have the same orientation.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return None # flexible-arity

    def badness(self):
        res = (AreRotationallyAligned_F(arguments=self.arguments, dimension="x").badness()
               + AreRotationallyAligned_F(arguments=self.arguments, dimension="y").badness()
               + AreRotationallyAligned_F(arguments=self.arguments, dimension="z").badness())
        return res

class AreTopAligned_F(ConstraintProposition):
    """A flexible-arity ("_F") constraint asserting that ALL passed objects are top-aligned along the z axis.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return None # flexible-arity

    def badness(self):
        return AreTranslationallyAligned_F(arguments=self.arguments, dimension="z",
                                           location="bounding_max").badness()

class AreBottomAligned_F(ConstraintProposition):
    """A flexible-arity ("_F") constraint asserting that ALL passed objects are bottom-aligned along the z axis.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return None # flexible-arity

    def badness(self):
        return AreTranslationallyAligned_F(arguments=self.arguments, dimension="z",
                                           location="bounding_min").badness()

class AreXPlusAligned_F(ConstraintProposition):
    """A flexible-arity ("_F") constraint asserting that ALL passed objects are aligned along the positive x axis.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return None # flexible-arity

    def badness(self):
        return AreTranslationallyAligned_F(arguments=self.arguments, dimension="x",
                                           location="bounding_max").badness()

class AreXMinusAligned_F(ConstraintProposition):
    """A flexible-arity ("_F") constraint asserting that ALL passed objects are aligned along the negative x axis.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return None # flexible-arity

    def badness(self):
        return AreTranslationallyAligned_F(arguments=self.arguments, dimension="x",
                                           location="bounding_min").badness()

class AreYPlusAligned_F(ConstraintProposition):
    """A flexible-arity ("_F") constraint asserting that ALL passed objects are aligned along the positive y axis.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return None # flexible-arity

    def badness(self):
        return AreTranslationallyAligned_F(arguments=self.arguments, dimension="y",
                                           location="bounding_max").badness()

class AreYMinusAligned_F(ConstraintProposition):
    """A flexible-arity ("_F") constraint asserting that ALL passed objects are aligned along the negative y axis.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return None # flexible-arity

    def badness(self):
        return AreTranslationallyAligned_F(arguments=self.arguments, dimension="y",
                                           location="bounding_min").badness()

class AreSymmetricalAround_T(ConstraintProposition):
    """A ternary ("_T") constraint asserting that the passed objects must look like:
    argument_0   <---equal--->   argument_1   <---equal--->   argument_2
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 3 # ternary

    def badness(self):
        return AreSymmetrical_B(arguments=[self.arguments[0],self.arguments[2]],
                                point=self.arguments[1].loc).badness()

# overlap defined above in primitive constraints

class AreParallelX_B(ConstraintProposition):
    """A binary ("_B") constraint asserting that the TWO passed objects must be parallel in x-rotation.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 2 # binary

    def badness(self):
        return AreRotationallyAligned_F(arguments=self.arguments, dimension="x").badness()

class AreParallelY_B(ConstraintProposition):
    """A binary ("_B") constraint asserting that the TWO passed objects must be parallel in y-rotation.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 2 # binary

    def badness(self):
        return AreRotationallyAligned_F(arguments=self.arguments, dimension="y").badness()

class AreParallelZ_B(ConstraintProposition):
    """A binary ("_B") constraint asserting that the TWO passed objects must be parallel in z-rotation.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 2 # binary

    def badness(self):
        return AreRotationallyAligned_F(arguments=self.arguments, dimension="z").badness()

class ArePerpendicularX_B(ConstraintProposition):
    """A binary ("_B") constraint asserting that the TWO passed objects must be perpendicular in x-rotation.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 2 # binary

    def badness(self):
        return HaveRotationalDifference_B(self.arguments, target_difference=90,
                                          dimension="x").badness()

class ArePerpendicularY_B(ConstraintProposition):
    """A binary ("_B") constraint asserting that the TWO passed objects must be perpendicular in y-rotation.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 2 # binary

    def badness(self):
        return HaveRotationalDifference_B(self.arguments, target_difference=90,
                                          dimension="y").badness()

class ArePerpendicularZ_B(ConstraintProposition):
    """A binary ("_B") constraint asserting that the TWO passed objects must be perpendicular in z-rotation.
    """
    def __init__(self, arguments):
        super().__init__(arguments)

    def define_arity(self):
        return 2 # binary

    def badness(self):
        return HaveRotationalDifference_B(self.arguments, target_difference=90,
                                          dimension="z").badness()

############################################# MAIN PROBLEM INTERFACE CLASS #############################################

class Problem:
    """The primary way users interact with the constraint solver.

    This class handles problem creation, adding optimizable objects and constraint propositions, solving
    the problem, and plotting the current state of all objects. Every optimizable object currently
    in the problem is given an id by the user code on addition. Any object not added to the problem
    is treated as constant and immovable.

    NOTE: When adding constraint propositions between optimizable objects, it is recommended to use the
          function get_optimizable_object with the desired object's id to prevent confusion. Any objects
          that are not explicitly id-ed and added through add_optimizable_object but are mentioned in
          any of the constraint propositions is treated as immovable.

    Attributes:
        optimizable_objects: A dictionary of GeometricObjects that are optimizable, each with a unique id.
        constraint_propositions: A list of ConstraintPropositions between GeometricObjects that we want to satisfy.
        constraint_weights: A list of floats representing the weight of each constraint proposition at the same idx.
    """
    def __init__(self):
        """Create an empty problem."""
        self.optimizable_objects = {}
        self.constraint_propositions = []
        self.constraint_weights = []

    def add_optimizable_object(self, object, id):
        """Add a GeometricObject to self.optimizable_objects with given id.
        Args:
            object: A GeometricObject to be added.
            id: The id to identify the object with.
        Returns:
            None.
        Raises:
            ValueError if an object with id is already in self.optimizable_objects.
        """
        if id in self.optimizable_objects:
            raise ValueError(f"The id passed ({id}) is already associated with an object in the problem.")
        self.optimizable_objects[id] = object

    def remove_optimizable_object(self, id):
        """Remove an GeometricObject from self.optimizable_objects by id.
        Args:
            id: The id to of the object to remove.
        Returns:
            None.
        Raises:
            ValueError if id passed is not present in self.optimizable_objects.
        """
        if id not in self.optimizable_objects:
            raise ValueError(f"The id passed ({id}) is not associated with any object in the problem.")
        self.optimizable_objects.pop(id)

    def get_optimizable_object(self, id):
        """Get pointer to an GeometricObject from self.optimizable_objects by id.
        Args:
            id: The id to of the object to get.
        Returns:
            A pointer to the desired object.
        Raises:
            ValueError if id passed is not present in self.optimizable_objects.
        """
        if id not in self.optimizable_objects:
            raise ValueError(f"The id passed ({id}) is not associated with any object in the problem.")
        return self.optimizable_objects[id]

    def add_constraint_proposition(self, proposition, weight):
        """Add a new ConstraintProposition to the problem.
        Args:
            proposition: The constraint proposition to add.
            weight: The weight of this constraint proposition.
        Returns:
            None.
        """
        self.constraint_propositions.append(proposition)
        self.constraint_weights.append(weight)

    def _flatten_optimizable_parameters(self):
        """Helper function to flatten the optimizable parameters of all optimizable objects for scipy."""
        arrays = []
        for id, object in self.optimizable_objects.items():
            arrays.append(object.get_optimizable_attr())
        return np.concatenate(arrays)

    def _recover_optimizable_parameters(self, flat_array):
        """Helper function to recover the optimizable parameters of all optimizable objects given flat array.
        NOTE: flat_array has to be one originally outputted by _flatten_optimizable_parameters."""
        obj_len = len(next(iter(self.optimizable_objects.values())).get_optimizable_attr())
        cur = 0
        for id, object in self.optimizable_objects.items():
            object.set_optimizable_attr(flat_array[cur:cur+obj_len])
            cur = cur + obj_len

    def solve(self, verbose=False):
        """Tune the optimizable parameters of all optimizable objects to best satisfy constraint propositions.
    
        This is the main function of the solver. It modifies objects in place.
    
        Args:
            verbose: bool, whether or not to print full scipy.minimize results.
        Returns:
            None. Modifies self.optimizable_objects in-place.
        """
        def objective(flat_array):
            # update the objects
            self._recover_optimizable_parameters(flat_array)

            # compute the total weighted badness measure for all constraint propositions
            total_badness = 0
            for i in range(len(self.constraint_propositions)):
                total_badness += (self.constraint_propositions[i].badness() * self.constraint_weights[i])
            return total_badness

        # compute solution, changing objects every iteration along the way
        # solution = minimize(objective, x0=self._flatten_optimizable_parameters(), method="Nelder-Mead")
        solution = minimize(objective, x0=self._flatten_optimizable_parameters(), method="powell")

        # optionally print full optimization results
        if verbose:
            print(solution)

        # set the objects to the final solution
        self._recover_optimizable_parameters(solution.x)

    def plot_on_ax(self, ax, ax_title, fixed_axes=None, elev=30, azim=40):
        """Plot the problem with all optimizable objects on axis ax.

        This generates a 3D plot.

        Args:
            ax: matplotlib axes to plot on.
            ax_title: str, title of axes.
            fixed_axes: If not None, all x/y/z-lims will be set to (-fixed_axes,fixed_axes).
            elev, azim: matplotlib 3D plot viewing angle.
        Returns:
            None. Modifies argument ax.
        """
        # define colors
        colors = list(mcolors.TABLEAU_COLORS.keys())

        # add all objects to ax
        legend_patchs = []
        color_idx = 0 # will wrap around
        for id, object in self.optimizable_objects.items(): 
            legend_patchs.append(object.add_self_to_axis(ax, label=id, color=colors[color_idx]))
            color_idx += 1
            color_idx = color_idx % len(colors)

        # configure plot info
        ax.set_title(ax_title, y=0.95)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(handles=legend_patchs, loc="upper right", bbox_to_anchor=(1.32,1))
        if fixed_axes is not None:
            ax.set_xlim(-fixed_axes,fixed_axes)
            ax.set_ylim(-fixed_axes,fixed_axes)
            ax.set_zlim(-fixed_axes,fixed_axes)

        #configure view
        ax.set_proj_type("persp",focal_length=0.2)
        ax.view_init(elev=elev,azim=azim)
