from source.geometry import Cuboid
from source.constraints import Cover

def test_cover():
    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj2 = Cuboid((0, 0, 0), (0, 0, 0), (1, 1, 1), "")
    assert Cover([obj1, obj2]).badness() == 0.0

    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj2 = Cuboid((1, 0, 0), (0, 0, 0), (1, 1, 1), "")
    assert Cover([obj1, obj2]).badness() == 0.5
