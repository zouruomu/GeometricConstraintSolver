from source.geometry import Cuboid
from source.constraints.compositional.backtoback import BackToBack
import numpy as np

def test_backtoback():
    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj2 = Cuboid((2, 0, 0), (0, 0, 180), (2, 2, 2), "")
    assert np.isclose(BackToBack([obj1, obj2]).badness(), 0.0)

    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj2 = Cuboid((1, 0, 0), (0, 0, 0), (2, 2, 2), "")
    assert BackToBack([obj1, obj2]).badness() > 0.0