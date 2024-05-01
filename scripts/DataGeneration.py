import os
from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import pickle
from multiprocessing import Pool
from functools import partial
import argparse
from copy import deepcopy

import sys
sys.path.append(".")
from source import *

def generate_data(datapoint_id,
                  data_directory, 
                  run_name,
                  vacancy_percentage=0.5,
                  max_constraint_density=1.5,
                  max_badness_tolerated=0.1,
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
        max_badness_tolerated: the max badness value to count as "satisfied"

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
            "text": str(constraint),
            "constraint": constraint.save(),
            "weight": weight,
            "final_badness": constraint.badness()
        }
        return constraint_dict
    
    # define meta-configurations and all available constraints
    object_min_scale = 2 / 10.
    object_max_scale = 580 / 10.
    scene_min_x_scale = 200 / 10.
    scene_max_x_scale = 600 / 10.
    scene_min_y_scale = 200 / 10.
    scene_max_y_scale = 600 / 10.
    scene_zmax = 600 / 10.
    constraint_classes = [TranslationalAlignment, Target, Parallelism, Perpendicularity,
                          Proximity, Symmetry, Direction]
    constraint_choice_prob = np.array([1, 0.25, 0.25, 0.25, 1, 1, 1])
    constraint_choice_prob = constraint_choice_prob / constraint_choice_prob.sum()

    if save_visualizations:
        figures_folder_name = os.path.join(data_directory, f"{run_name}_visualizations/")
    
    # main datapoint generation loop
    while True:
        # create holders to be later put into a dictionary to output
        initial_objs_output = []
        constraints_output = []
        solved_objects_output = []

        # determine datapoint bounds
        scene_xmin = 0
        scene_ymin = 0
        scene_xmax = np.random.randint(low=scene_min_x_scale, high=scene_max_x_scale+1)
        scene_ymax = np.random.randint(low=scene_min_y_scale, high=scene_max_y_scale+1)
        specific_max_scale = min(object_max_scale, scene_xmax, scene_ymax) # cannot have objects bigger than room

        # initialize problem and determine 
        problem = Problem(scene_xmin=scene_xmin, scene_xmax=scene_xmax,
                          scene_ymin=scene_ymin, scene_ymax=scene_ymax,
                          scene_zmax=scene_zmax)

        # create the objects (random initialization, 50% of objects will be upright)
        # keep adding objects until over 
        objs = []
        scene_area = (scene_xmax - scene_xmin) * (scene_ymax - scene_ymin)
        area_to_occupy = (1-vacancy_percentage) * scene_area
        cur_occupied_area = 0
        obj_counter = 0
        while True:
            # get object scale and update cur_occupied_area
            scale = np.random.uniform(low=object_min_scale, high=specific_max_scale, size=(3))
            object_area = scale[0] * scale[1]
            if cur_occupied_area + object_area > area_to_occupy:
                break
            cur_occupied_area += object_area
            obj_counter += 1
            # get object loc
            loc = [0,0,0]
            loc[0] = np.random.randint(low=scene_xmin, high=scene_xmax+1)
            loc[1] = np.random.randint(low=scene_ymin, high=scene_ymax+1)
            loc[2] = scale[2]/2 # place on x-y plane
            # get object rot
            rot = [0,0,0 if np.random.uniform() < 0.5 else np.random.uniform(low=-180, high=180)]
            # create object
            obj = Cuboid(loc=loc,
                         rot=rot,
                         scale=scale,
                         name=f"item_{obj_counter}")
            # add to list and problem
            objs.append(obj)
            problem.add_optimizable_object(obj)
            # add in dictionary form to initial_objs_output
            initial_objs_output.append(obj_to_dict(obj))
            initial_objs_output = deepcopy(initial_objs_output)
        num_objs = len(objs)
        objs = np.array(objs)

        # discard half-generated datapoint if too few objects
        if num_objs < 3:
            continue

        # optionally setup plot
        if save_visualizations:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12), subplot_kw={"projection": "3d"})
            problem.plot_on_ax(ax=ax[0][0], ax_title="Unconstrained Objects (Perspective View)",
                               elev=30, azim=40, persp=True)
            problem.plot_on_ax(ax=ax[0][1], ax_title="Unconstrained Objects (Top View)",
                               elev=90, azim=-90, persp=False)

                # get number of constraints to add
        num_constraints = np.random.randint(low=num_objs, high=round(num_objs*max_constraint_density)+1) 

        # add the constraints (between random objects)
        for i in range(1,num_constraints): # num_constraints - 1 random constraints and 1 NoOverlap constraint
            # find random constraint
            constraint_class = np.random.choice(a=constraint_classes, p=constraint_choice_prob)
            constraint_arity = constraint_class.arity()
            if isinstance(constraint_arity, int):
                ...
            elif isinstance(constraint_arity, tuple):
                assert len(constraint_arity) == 2
                low = constraint_arity[0]
                high = constraint_arity[1]
                if low == -1:
                    raise ValueError("Cannot have low=-1 in tuple constraint_arity")
                if high == -1:
                    high = max(4,num_objs/2)
                constraint_arity = np.random.randint(low=low, high=high)
            else:
                raise ValueError("constraint_arity must be int or tuple of ints")

            # find random objects as arguments and find weight
            constraint_args = np.random.choice(a=objs, size=constraint_arity, replace=False)
            constraint_weight = 1 # same weight for all
            # constraint_weight = np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # random weight

            # create constraint object and add to problem (will add to output later, after postfix)
            constraint_obj = constraint_class.random(arguments=constraint_args)
            problem.add_constraint_proposition(constraint_obj, weight=constraint_weight)

        # add the NoOverlap constraint
        problem.add_constraint_proposition(NoOverlap(arguments=objs), weight=10)

        # solve the problem
        problem.solve()

        # post-solve object processing
        for obj in objs:
            # re-cast z-rotation to (-180,180): bound on angles is None for optimizer
            modded = obj.rot[2] % 360 # cast to (0,360)
            obj.rot[2] = modded if modded <= 180 else modded - 360

            # add the solved objects to solved_objects_output
            solved_objects_output.append(obj_to_dict(obj))

        # post-solve constraint processing: add ONLY satisfied constraints (with badness below max_badness_tolerated)
        if save_visualizations: # optionally create a string to put at middle of visualization
            description = ""

        success = True
        for i in range(num_constraints):
            constraint = problem.constraint_propositions[i]
            weight = problem.constraint_weights[i]
            if isinstance(constraint, NoOverlap):
                all_iou = constraint.all_iou()
                for iou in all_iou:
                    if iou > 0.01:
                        success = False
                        break
                if not success:
                    break
            else:
                if constraint.badness() > max_badness_tolerated:
                    continue
                constraints_output.append(constraint_to_dict(constraint, weight))
                if save_visualizations:
                    description += str(constraint) + f" (badness after solving: {round(constraint.badness(),2)})\n"
        
        if len(constraints_output) == 0:
            success = False

        if success:
            # optionally plot the final state and save figure
            if save_visualizations:
                problem.plot_on_ax(ax=ax[1][0], ax_title="Solved Objects (Perspective View)",
                                   elev=30, azim=40, persp=True)
                problem.plot_on_ax(ax=ax[1][1], ax_title="Solved Objects (Top View)",
                                   elev=90, azim=-90, persp=False)
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
                "scene_x_bounds": (scene_xmin, scene_xmax),
                "scene_y_bounds": (scene_ymin, scene_ymax)
            }
            break
        
    return datapoint

# generate_data(data_directory="./GeneratedData", run_name="Dataset5",
#               num_datapoints=5,
#               vacancy_percentage=0.5,
#               max_constraint_density=1.5,
#               max_badness_tolerated=0.1,
#               save_visualizations=True)

def generate_dataset(num_workers: int, num_datapoints: int = 5, save_visualizations: bool = False):
    data_directory = "./data"
    run_name = f"Dataset{num_datapoints}"
    generate_func = partial(generate_data,
                            data_directory=data_directory,
                            run_name=run_name,
                            vacancy_percentage=0.5,
                            max_constraint_density=1.5,
                            max_badness_tolerated=0.1,
                            save_visualizations=save_visualizations)
    shutil.rmtree(run_name, ignore_errors=True)
    # optionally setup folder to save figures in
    if save_visualizations:
        figures_folder_name = os.path.join(data_directory, f"{run_name}_visualizations/")
        if os.path.exists(figures_folder_name):
            shutil.rmtree(figures_folder_name)
        os.makedirs(figures_folder_name)
        
    if num_workers <= 1:
        dataset = []
        for i in range(num_datapoints):
            dataset.append(generate_func(i))
    else:
        with Pool(num_workers) as pool:
            dataset = list(tqdm(pool.imap(generate_func, range(num_datapoints)), total=num_datapoints))
    import ipdb; ipdb.set_trace()


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--num_workers", type=int, default=0)
    args.add_argument("--save_visualizations", action="store_true")
    args.add_argument("--num_datapoints", type=int, default=10000)
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    num_workers = args.num_workers
    save_visualizations = args.save_visualizations
    
    generate_dataset(num_workers=num_workers, num_datapoints=args.num_datapoints, save_visualizations=save_visualizations)