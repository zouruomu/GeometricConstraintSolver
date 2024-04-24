import os
import warnings
warnings.filterwarnings('ignore')
from itertools import product, combinations
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, basinhopping
import matplotlib
matplotlib.rcParams['figure.dpi'] = 140
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from tqdm import tqdm
import shutil
import pickle
from GeoConstraintSolver import *

def generate_data(data_directory, run_name, num_datapoints, axes_scale=10,
                  min_object_count=3, max_object_count=10,
                  min_constraint_count=5, max_constraint_count=15,
                  save_visualizations=False):
    """Generate data mapping from initial objects and a list of constraint propositions to final/solved objects.

    This function will create a new pickle file with name "{run_name}.pkl" in data_directory. If
    "{run_name}.pkl" already exists, it will be overriden. Unpacking "{run_name}.pkl"
    will yield a list of dictionaries (each datapoint is represented as a dictionary).

    NOTE: Current version of function only generates data where x and y rotation are both 0.

    Args:
        data_directory: str, directory in which to create a new folder to store generated data.
        run_name: str, the name of the pkl file to create will be "{run_name}.pkl".
        num_datapoints: int, number of datapoints to generate.
        axes_scale: int, the scale of the coordinates to generate, axes will range [-axes_scale, axes_scale].
        {min,max}_object_count: ints, the min/max number of objects in each datapoint (sampled uniformally).
        {min,max}_constraint_count: ints, the min/max number of constraints in each datapoint (sampled uniformally).

    Returns:
        None.
    """
    # define function to convert object to dictionary according to given specifications
    def obj_to_dict(obj, name):
        obj_dict = {
            "name": name,
            "position": obj.loc,
            "size": obj.scale,
            "orientation": obj.rot
        }
        return obj_dict

    # define function to convert constraint to dictionary according to given specifications
    def constraint_to_dict(name, args, weight):
        obj_dict = {
            "class": name,
            "args": args,
            "weight": weight
        }
        return obj_dict
    
    # define meta-configurations and all available constraints
    object_min_scale = 1
    object_max_scale = 3
    object_coord_range = axes_scale - object_max_scale/2 # actual range is [-object_coord_range, object_coord_range]
    constraints = [IsUpright_U, IsAtOrigin_U, AreProximal_B, HaveSameRotation_F,
                   AreTopAligned_F, AreBottomAligned_F, AreSymmetricalAround_T,
                   AreNotOverlapping_B, AreParallelZ_B, ArePerpendicularZ_B,
                   AreXPlusAligned_F, AreXMinusAligned_F, AreYPlusAligned_F, AreYMinusAligned_F]
    constraint_names = ["IsUpright", "IsAtOrigin", "AreProximal", "HaveSameRotation",
                        "AreTopAligned", "AreBottomAligned", "AreSymmetricalAround",
                        "AreNotOverlapping", "AreParallelZ", "ArePerpendicularZ",
                        "AreXPlusAligned", "AreXMinusAligned", "AreYPlusAligned", "AreYMinusAligned"]
    constraint_arities = [1, 1, 2, None, None, None, 3, 2, 2, 2, None, None, None, None]
    constraint_weights = np.array([1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 0.5, 0.5])
    constraint_weights = constraint_weights / constraint_weights.sum()

    # optionally setup folder to save figures in
    if save_visualizations:
        figures_folder_name = os.path.join(data_directory, f"{run_name}_visualizations/")
        if os.path.exists(figures_folder_name):
            shutil.rmtree(figures_folder_name)
        os.makedirs(figures_folder_name)

    # create holder for final dataset to pickle
    dataset = []
    
    # main datapoint generation loop
    for datapoint_id in tqdm(range(num_datapoints)):
        # create holders to be later put into a dictionary to output
        initial_objs_output = []
        constraints_output = []
        solved_objects_output = []
        
        # initialize problem and determine how many objects and contraints this datapoint will have
        problem = Problem()
        num_objs = np.random.randint(low=min_object_count, high=max_object_count+1)
        num_constraints = np.random.randint(low=min_constraint_count, high=max_constraint_count+1)  

        # create the objects (random initialization, 50% of objects will be upright)
        objs = []
        obj_names = []
        for i in range(1,num_objs+1):
            #create random object
            obj = Cuboid(loc=np.random.uniform(low=-object_coord_range ,high=object_coord_range, size=(3)),
                         rot=[0,0,0 if np.random.uniform() < 0.5 else np.random.uniform(low=-180, high=180)],
                         scale=np.random.uniform(low=object_min_scale ,high=object_max_scale, size=(3)))
            # name and add to lists and problem
            obj_name = f"Cube{i}"
            objs.append(obj)
            obj_names.append(obj_name)
            problem.add_optimizable_object(obj, obj_name)
            # add in dictionary form to initial_objs_output
            initial_objs_output.append(obj_to_dict(obj, obj_name))
        objs = np.array(objs)
        obj_names = np.array(obj_names)

        # optionally setup plot
        if save_visualizations:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,12), subplot_kw={"projection": "3d"})
            problem.plot_on_ax(ax=ax[0], ax_title="Initial Objects (Unconstrained)", fixed_axes=axes_scale)

        # add the constraints (between random objects)
        if save_visualizations: # optionally create a string to put at bottom of visualization
            description = ""
        for i in range(1,num_constraints+1):
            # find random constraint
            constraint_idx = np.random.choice(len(constraints), p=constraint_weights) # [0,len(constraints))
            constraint = constraints[constraint_idx]
            constraint_name = constraint_names[constraint_idx]
            constraint_arity = constraint_arities[constraint_idx]
            if constraint_arity is None: # for flexible arity constraints
                constraint_arity = np.random.randint(low=1, high=num_objs)

            # find random objects as arguments and find random weight
            constraint_arg_idx = np.random.choice(a=num_objs, size=constraint_arity, replace=False)
            constraint_args = objs[constraint_arg_idx]
            constraint_arg_names = obj_names[constraint_arg_idx]
            constraint_weight = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # random weight

            # add to problem, constraints_output, and optionally update description
            problem.add_constraint_proposition(constraint(arguments=constraint_args),
                                               weight=constraint_weight)
            constraints_output.append(constraint_to_dict(name=constraint_name,
                                                         args=list(constraint_arg_names),
                                                         weight=constraint_weight))
            if save_visualizations:
                description += f"Constraint {i} (with weight {constraint_weight}): {constraint_name}({constraint_arg_names}).\n"

        # solve the problem
        problem.solve()

        # re-cast z-rotation to (-180,180) add the solved objects to solved_objects_output
        for i in range(num_objs):
            modded = objs[i].loc[2] % 360 # cast to (0,360)
            objs[i].loc[2] = modded if modded <= 180 else modded - 360
            solved_objects_output.append(obj_to_dict(objs[i], obj_names[i]))

        # optionally plot the final state and save figure
        if save_visualizations:
            problem.plot_on_ax(ax=ax[1], ax_title="Solved Objects (Constrained)", fixed_axes=axes_scale)
            file_name = os.path.join(figures_folder_name, f"visualization_{datapoint_id}.png")
            plt.figtext(x=0,y=0.39,s=description)
            plt.subplots_adjust(hspace=1)
            plt.savefig(file_name)
            plt.close()

        # create dictionary for this datapoint and add to dataset
        datapoint = {
            "initial_objects":initial_objs_output,
            "constraints":constraints_output,
            "solved_objects":solved_objects_output
        }
        dataset.append(datapoint)

    # pickle dataset
    pkl_path = os.path.join(data_directory, f"{run_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)

generate_data(data_directory="./GeneratedData", run_name="Dataset8000",
              num_datapoints=8000, axes_scale=10,
              min_object_count=3, max_object_count=10,
              min_constraint_count=5, max_constraint_count=15,
              save_visualizations=False)

# generate_data(data_directory="./GeneratedData", run_name="Dataset2",
#               num_datapoints=2, axes_scale=10,
#               min_object_count=3, max_object_count=10,
#               min_constraint_count=5, max_constraint_count=15,
#               save_visualizations=True)
