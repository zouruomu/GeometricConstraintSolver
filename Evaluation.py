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

def evaluate(objs_dict_list, constraints_dict_list):
    name_to_constraint = {
        "IsUpright":IsUpright_U,
        "IsAtOrigin":IsAtOrigin_U,
        "AreProximal":AreProximal_B,
        "HaveSameRotation":HaveSameRotation_F,
        "AreTopAligned":AreTopAligned_F,
        "AreBottomAligned":AreBottomAligned_F,
        "AreSymmetricalAround":AreSymmetricalAround_T,
        "AreNotOverlapping":AreNotOverlapping_B,
        "AreParallelZ":AreParallelZ_B,
        "ArePerpendicularZ":ArePerpendicularZ_B,
        "AreXPlusAligned":AreXPlusAligned_F,
        "AreXMinusAligned":AreXMinusAligned_F,
        "AreYPlusAligned":AreYPlusAligned_F,
        "AreYMinusAligned":AreYMinusAligned_F
    }
    def dict_to_obj(dict):
        name = dict["name"]
        obj = Cuboid(loc=dict["position"], rot=dict["orientation"], scale=dict["size"])
        return obj, name
    def dict_list_to_name_obj_map(dict_list):
        name_obj_map = {}
        for dict in dict_list:
            obj, name = dict_to_obj(dict)
            name_obj_map[name] = obj
        return name_obj_map
    def dict_to_constraint(dict, name_obj_map):
        arguments = []
        for arg_name in dict["args"]:
            arguments.append(name_obj_map[arg_name])
        constraint = name_to_constraint[dict["class"]](arguments)
        weight = dict["weight"]
        return constraint, weight

    name_obj_map = dict_list_to_name_obj_map(objs_dict_list)
    total_badness = 0
    for constraint_dict in constraints_dict_list:
        constraint, weight = dict_to_constraint(constraint_dict, name_obj_map)
        total_badness += constraint.badness() * weight

    return total_badness