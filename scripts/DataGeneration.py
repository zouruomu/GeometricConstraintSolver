import os
import warnings
warnings.filterwarnings('ignore')
from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import pickle
import multiprocessing
from multiprocessing import Process

import sys
sys.path.append("../")
from source import *

def generate_data(data_directory, run_name, num_datapoints,
                  min_object_count=3, max_object_count=10,
                  max_constraint_density=1.5,
                  save_visualizations=False):
    """Generate data mapping from initial objects and a list of constraint propositions to final/solved objects.

    This function will create a new pickle file with name "{run_name}.pkl" in data_directory. If
    "{run_name}.pkl" already exists, it will be overriden. Unpacking "{run_name}.pkl"
    will yield a list of dictionaries (each datapoint is represented as a dictionary).

    NOTE: Current version of function only generates data where x and y rotation are both 0 and
          z location is z_scale/2 (object "placed" on the z=0 plane).

    Args:
        data_directory: str, directory in which to create a new folder to store generated data.
        run_name: str, the name of the pkl file to create will be "{run_name}.pkl".
        num_datapoints: int, number of datapoints to generate.
        {min,max}_object_count: ints, the min/max number of objects in each datapoint (sampled uniformally).
        max_constraint_density: float, number of constraints will be uniform(1, round(num_objs*high))

    Returns:
        None.
    """
    # define function to convert object to dictionary according to given specifications
    def obj_to_dict(obj):
        obj_dict = {
            "name": obj.name,
            "position": obj.loc,
            "size": obj.scale,
            "orientation": obj.rot
        }
        return obj_dict

    # define function to convert constraint to dictionary according to given specifications
    def constraint_to_dict(constraint, weight):
        constraint_dict = {
            "constraint": str(constraint),
            "weight": weight
        }
        return constraint_dict
    
    # define meta-configurations and all available constraints
    object_min_scale = 1
    object_max_scale = 3
    constraint_classes = [TranslationalAlignment, RotationalAlignment, DirectionTowards, Parallelism, Perpendicularity,
                          Proximity, Symmetry]
    constraint_choice_prob = np.array([1, 0.25, 0.25, 0.25, 0.25, 1, 1])
    constraint_choice_prob = constraint_choice_prob / constraint_choice_prob.sum()

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

        # determine datapoint bounds and how many objects and contraints this datapoint will have
        scene_xmin = np.random.randint(low=-20, high=-10+1)
        scene_ymin = np.random.randint(low=-20, high=-10+1)
        scene_xmax = np.random.randint(low=10, high=20+1)
        scene_ymax = np.random.randint(low=10, high=20+1)
        num_objs = np.random.randint(low=min_object_count, high=max_object_count+1)
        num_constraints = np.random.randint(low=num_objs, high=round(num_objs*max_constraint_density)+1)  

        # initialize problem and determine 
        problem = Problem(scene_xmin=scene_xmin, scene_xmax=scene_xmax,
                          scene_ymin=scene_ymin, scene_ymax=scene_ymax,
                          scene_zmax=10)

        # create the objects (random initialization, 50% of objects will be upright)
        objs = []
        for i in range(1,num_objs+1):
            #create random object
            scale = np.random.uniform(low=object_min_scale, high=object_max_scale, size=(3))
            loc = [0,0,0]
            loc[0] = np.random.randint(low=scene_xmin+object_max_scale, high=scene_xmax-object_max_scale+1)
            loc[1] = np.random.randint(low=scene_ymin+object_max_scale, high=scene_ymax-object_max_scale+1)
            loc[2] = scale[2]/2 # place on x-y plane
            rot = [0,0,0 if np.random.uniform() < 0.5 else np.random.uniform(low=-180, high=180)]
            obj = Cuboid(loc=loc,
                         rot=rot,
                         scale=scale,
                         name=f"Cube{i}")
            # add to list and problem
            objs.append(obj)
            problem.add_optimizable_object(obj)
            # add in dictionary form to initial_objs_output
            initial_objs_output.append(obj_to_dict(obj))
        objs = np.array(objs)

        # optionally setup plot
        if save_visualizations:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12), subplot_kw={"projection": "3d"})
            problem.plot_on_ax(ax=ax[0][0], ax_title="Unconstrained Objects (Perspective View)",
                               elev=30, azim=40, persp=True)
            problem.plot_on_ax(ax=ax[0][1], ax_title="Unconstrained Objects (Top View)",
                               elev=90, azim=0, persp=False)

        # add the constraints (between random objects)
        if save_visualizations: # optionally create a string to put at middle of visualization
            description = ""
        for i in range(1,num_constraints): # num_constraints - 1 random constraints and 1 NoOverlap constraint
            # find random constraint
            constraint_class = np.random.choice(a=constraint_classes, p=constraint_choice_prob)
            constraint_arity = constraint_class.arity()
            if constraint_arity is None: # for flexible arity constraints
                constraint_arity = np.random.randint(low=1, high=max(2,num_objs/2))

            # find random objects as arguments and find weight
            constraint_args = np.random.choice(a=objs, size=constraint_arity, replace=False)
            constraint_weight = 1 # same weight for all
            # constraint_weight = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # random weight

            # create constraint object, add to problem, constraints_output, and optionally update description
            constraint_obj = constraint_class(arguments=constraint_args)
            problem.add_constraint_proposition(constraint_obj, weight=constraint_weight)
            constraints_output.append(constraint_to_dict(constraint=constraint_obj, weight=constraint_weight))
            if save_visualizations:
                description += str(constraint_obj) + "\n"

        # add the NoOverlap constraint
        no_overlap = NoOverlap(arguments=objs)
        problem.add_constraint_proposition(no_overlap, weight=1)
        constraints_output.append(constraint_to_dict(constraint=no_overlap, weight=1))
        if save_visualizations:
            description += str(no_overlap)

        # solve the problem
        problem.solve()

        # post-solve processing
        for obj in objs:
            # re-cast z-rotation to (-180,180): bound on angles is None for optimizer
            modded = obj.rot[2] % 360 # cast to (0,360)
            obj.rot[2] = modded if modded <= 180 else modded - 360

            # add the solved objects to solved_objects_output
            solved_objects_output.append(obj_to_dict(obj))

        # optionally plot the final state and save figure
        if save_visualizations:
            problem.plot_on_ax(ax=ax[1][0], ax_title="Solved Objects (Perspective View)",
                               elev=30, azim=40, persp=True)
            problem.plot_on_ax(ax=ax[1][1], ax_title="Solved Objects (Top View)",
                               elev=90, azim=0, persp=False)
            file_name = os.path.join(figures_folder_name, f"{run_name}_visualization_{datapoint_id}.png")
            plt.figtext(x=0.1,y=0.39,s=description)
            plt.subplots_adjust(hspace=1)
            plt.savefig(file_name)
            plt.close()

        # create dictionary for this datapoint and add to dataset
        datapoint = {
            "initial_objects":initial_objs_output,
            "constraints":constraints_output,
            "solved_objects":solved_objects_output,
            "scene_x_bounds": (scene_xmin-object_max_scale, scene_xmax+object_max_scale),
            "scene_y_bounds": (scene_ymin-object_max_scale, scene_ymax+object_max_scale)
        }
        dataset.append(datapoint)

    # pickle dataset
    pkl_path = os.path.join(data_directory, f"{run_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)

# generate_data(data_directory="./GeneratedData", run_name="Dataset5",
#               num_datapoints=5,
#               min_object_count=3, max_object_count=10,
#               max_constraint_density=1.5,
#               save_visualizations=True)

def generate_data_multiprocess(data_directory, run_name, num_workers, num_datapoints_per_worker,
                               min_object_count=3, max_object_count=10,
                               max_constraint_density=1.5, save_visualizations=False):
    """Multiprocessing version of generate_data, uses generate_data.

    NOTE: Total datapoints generated will be num_workers * num_datapoints_per_worker
    """
    # setup
    temp_directory = os.path.join(data_directory, f"temp")
    if os.path.exists(temp_directory):
        shutil.rmtree(temp_directory)
    os.makedirs(temp_directory)
    process_list = []

    # start all workers
    for i in range(num_workers):
        p = Process(target=generate_data, args=[temp_directory, f"Worker{i}", num_datapoints_per_worker,
                                                min_object_count, max_object_count,
                                                max_constraint_density, save_visualizations])
        p.start()
        process_list.append(p)

    # join workers
    for process in process_list:
        process.join()

    # when done, combined pickled data
    all_data = []
    for i in range(num_workers):
        pickle_file = os.path.join(temp_directory, f"Worker{i}.pkl")
        with open(pickle_file, "rb") as f:
            cur_data = pickle.load(f)
        all_data += cur_data

    # re-pickle combined dataset
    pkl_path = os.path.join(data_directory, f"{run_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_data, f)
    
    # optionally combined all visualizations
    if save_visualizations:
        # make new folder for combined visualizations
        figures_folder_dir = os.path.join(data_directory, f"{run_name}_visualizations/")
        if os.path.exists(figures_folder_dir):
            shutil.rmtree(figures_folder_dir)
        os.makedirs(figures_folder_dir)

        # loop through all temp folders
        for i in range(num_workers):
            temp_vis_dir = os.path.join(temp_directory, f"Worker{i}_visualizations/")
            for img_name in os.listdir(temp_vis_dir):
                img_dir = os.path.join(temp_vis_dir, img_name)
                shutil.move(img_dir, figures_folder_dir)

    # finally remove entire temp directory
    shutil.rmtree(temp_directory)

if __name__ == "__main__":
    generate_data_multiprocess(data_directory="./GeneratedData", run_name="Dataset10000",
                            num_workers=10, num_datapoints_per_worker=1000,
                            min_object_count=3, max_object_count=10,
                            max_constraint_density=1.5, save_visualizations=True)