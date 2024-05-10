from source.geometry import Cuboid
from source.constraints import Target
import numpy as np


def test_cover():
    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj2 = Cuboid((0, 3, 0), (0, 0, 0), (2, 2, 2), "")
    assert np.isclose(Target([obj1, obj2]).badness(), 1.0)
    assert np.isclose(Target([obj2, obj1]).badness(), 0.0)
