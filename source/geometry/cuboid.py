import numpy as np
from itertools import product
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation as R

from .geometric_object import GeometricObject


class Cuboid(GeometricObject):
    """A cuboid.

    Inherits from the GeometricObject abstract class. A cuboid is a unit cube by default,
    the location/rotation/size of which is determined by the object's loc/rot/scale.

    Attributes:
        Same as super-class, no additional attributes. Add additional attributes as needed.
    """
    def __init__(self, loc, rot, scale):
        super().__init__(loc, rot, scale)

    def get_bounding_intervals(self):
        """Get bounding intervals around the (possibly rotated) cuboid.

        See super-class documentation for specifics. Note that, for a cuboid, the bounding
        intervals will be perfectly tight if the cuboid is not rotated/rotated with right
        angles, but otherwise these intervals will be loose.
        """
        # get the corners first -> extreme points can only be corners
        corners = self.get_corners()

        # find the max and min along each dimension
        min_by_dim = corners.min(axis=0)
        max_by_dim = corners.max(axis=0)

        # collect into tuple of three lists and return
        return ([min_by_dim[0], max_by_dim[0]],
                [min_by_dim[1], max_by_dim[1]],
                [min_by_dim[2], max_by_dim[2]])
        
    def add_self_to_axis(self, ax, label, color):
        """Add self as 3D wireframe cuboid to ax.

        See super-class documentation for specifics. For a cuboid, we just draw a simple
        wireframe connecting all vertices.
        """
        # get the corners to draw wireframe on
        vertices = self.get_corners()

        # # generate all pairwise combinations of vertices for wireframe and plot only edges
        # for start, end in combinations(vertices, 2):
        #     l2_dist = np.linalg.norm(start - end)
        #     if (abs(l2_dist-self.scale_x) < 1e-4        # only plot the lines that are edges
        #         or abs(l2_dist-self.scale_y) < 1e-4     # identify edges if distance is close to scale
        #         or abs(l2_dist-self.scale_z) < 1e-4):   # 1e-4 tolerance added due to numerical inpercision
        #         ax.plot(*zip(start,end), color=color)

        # the above works, but does needless computation and is subject to numerical lack of precision
        # more direct method: just plot all twelve lines directly since get_corners returns corners in fixed order
        ax.plot(*zip(vertices[0],vertices[1]), color=color) # lower rectangle
        ax.plot(*zip(vertices[1],vertices[3]), color=color)
        ax.plot(*zip(vertices[3],vertices[2]), color=color)
        ax.plot(*zip(vertices[2],vertices[0]), color=color)
        ax.plot(*zip(vertices[4],vertices[5]), color=color) # upper rectangle
        ax.plot(*zip(vertices[5],vertices[7]), color=color)
        ax.plot(*zip(vertices[7],vertices[6]), color=color)
        ax.plot(*zip(vertices[6],vertices[4]), color=color)
        ax.plot(*zip(vertices[0],vertices[4]), color=color) # connecting upper and lower rectangles
        ax.plot(*zip(vertices[1],vertices[5]), color=color)
        ax.plot(*zip(vertices[2],vertices[6]), color=color)
        ax.plot(*zip(vertices[3],vertices[7]), color=color)

        # return a patch for legend
        return mpatches.Patch(color=color, label=label)
        

    def get_corners(self):
        """Get the corner vertices of the cuboid.

        Cuboid-specific helper method to faciliate plotting and bounding-interval finding.

        Args:
            None
        Returns:
            An np.array of shape (8,3) with all eight corners of the cuboid.
            NOTE: The corners are returned in default order of itertools.product([-0.5,0.5], [-0.5,0.5], [-0.5,0.5]).
        """
        # define base (unit-cube) corners
        corners = np.array(list(product([-0.5,0.5], [-0.5,0.5], [-0.5,0.5])))

        # get rotation matrix from Euler angles
        rotation_matrix = R.from_euler(angles=self.rot,
                                       seq="xyz", degrees=True).as_matrix()

        # transform base corners with loc, rot, and scale
        corners = corners * self.scale # scale
        corners = corners @ rotation_matrix # rotate
        corners = corners + self.loc # translate
        return corners

########################################### Constraints: Abstract Superclass ###########################################