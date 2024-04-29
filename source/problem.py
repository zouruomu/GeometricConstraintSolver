import numpy as np
import matplotlib.colors as mcolors
from scipy.optimize import minimize


class Problem:
    """The primary way users interact with the constraint solver.

    This class handles problem creation, adding optimizable objects and constraint propositions, solving
    the problem, and plotting the current state of all objects. Every optimizable object currently
    in the problem is given an id by the user code on addition. Any object not added to the problem
    is treated as constant and immovable.

    Attributes:
        optimizable_objects: A dictionary of GeometricObjects that are optimizable, each with a unique id.
        constraint_propositions: A list of ConstraintPropositions between GeometricObjects that we want to satisfy.
        constraint_weights: A list of floats representing the weight of each constraint proposition at the same idx.
    """
    def __init__(self):
        """Create an empty problem."""
        self.optimizable_objects = []
        self.constraint_propositions = []
        self.constraint_weights = []

    def add_optimizable_object(self, object):
        """Add a GeometricObject to self.optimizable_objects.
        Args:
            object: A GeometricObject to be added.
        Returns:
            None.
        """
        self.optimizable_objects.append(object)

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
        for object in self.optimizable_objects:
            arrays.append(object.get_optimizable_attr())
        return np.concatenate(arrays)

    def _recover_optimizable_parameters(self, flat_array):
        """Helper function to recover the optimizable parameters of all optimizable objects given flat array.
        NOTE: flat_array has to be one originally outputted by _flatten_optimizable_parameters."""
        obj_len = len(self.optimizable_objects[0].get_optimizable_attr())
        cur = 0
        for object in self.optimizable_objects:
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

    def plot_on_ax(self, ax, ax_title, fixed_axes=None, elev=30, azim=40, persp=True):
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
        for object in self.optimizable_objects: 
            legend_patchs.append(object.add_self_to_axis(ax, color=colors[color_idx]))
            color_idx += 1
            color_idx = color_idx % len(colors)

        # configure plot info
        ax.set_title(ax_title, y=1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(handles=legend_patchs, loc="upper right", bbox_to_anchor=(1.4,1))
        if fixed_axes is not None:
            ax.set_xlim(-fixed_axes,fixed_axes)
            ax.set_ylim(-fixed_axes,fixed_axes)
            ax.set_zlim(0,fixed_axes*2)

        #configure view
        if persp:
            ax.set_proj_type("persp",focal_length=0.2)
        ax.view_init(elev=elev,azim=azim)
