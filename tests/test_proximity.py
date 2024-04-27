import numpy as np
from source.geometry import Cuboid
from source.constraints import Proximity

def test_proximity():
    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2))    
    obj2 = Cuboid((2, 2, 2), (0, 0, 0), (2, 2, 2))
    assert Proximity([obj1, obj2]).badness() == 0.0

    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2))    
    obj2 = Cuboid((2, 2, 2), (0, 0, 45), (2, 2, 2))
    assert np.isclose(Proximity([obj1, obj2]).badness(), np.sqrt(2) - 1)

    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2))    
    obj2 = Cuboid((0, 0, 0), (0, 0, 0), (1, 1, 1))    
    assert Proximity([obj1, obj2]).badness() == 0.0