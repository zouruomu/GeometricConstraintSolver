import numpy as np

def scaled_sigmoid(x):
    return ((1 / (1 + np.exp(-x))) - 0.5) * 2