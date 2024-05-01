import os
import warnings
warnings.filterwarnings('ignore')
from itertools import product, combinations
import numpy as np
from collections import defaultdict
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
from ..constraints import *
from importlib import import_module

"""
Example
-------
dataset = pickle.load(open("Dataset.pkl", "rb"))
datapoint = 2
print(evaluate(dataset[datapoint]["initial_objects"], dataset[datapoint]["constraints"]))
print(evaluate(dataset[datapoint]["solved_objects"], dataset[datapoint]["constraints"]))
"""

def evaluate(objs_dict_list, constraints_dict_list, return_each_badness=False):
    constraint_module = import_module("...constraints", package=__name__)
    def dict_to_obj(dict):
        name = dict["name"]
        obj = Cuboid(loc=dict["position"], rot=dict["orientation"], scale=dict["size"], name=name)
        return obj
    def dict_list_to_name_obj_map(dict_list):
        name_obj_map = {}
        for dict in dict_list:
            obj = dict_to_obj(dict)
            name_obj_map[obj.name] = obj
        return name_obj_map
    def dict_to_constraint(dict, name_obj_map):
        arguments = []
        for arg_name in dict["constraint"]["arguments"]:
            arguments.append(name_obj_map[arg_name])
        constraint_class = getattr(constraint_module, dict["constraint"]['type'])
        weight = dict["weight"]
        constraint = constraint_class(arguments, **dict["constraint"]["kwargs"])
        return constraint, weight

    name_obj_map = dict_list_to_name_obj_map(objs_dict_list)
    total_badness = 0

    each_badness = defaultdict(list)

    for constraint_dict in constraints_dict_list:
        constraint, weight = dict_to_constraint(constraint_dict, name_obj_map)
        assert weight == 1
        each_badness[constraint.__class__.__name__].append(constraint.badness() * weight)
    avg_badness = {k: np.mean(v) for k, v in each_badness.items()}
    total_badness = sum([v for k, v in avg_badness.items()])

    if return_each_badness:
        return total_badness, each_badness
    return total_badness