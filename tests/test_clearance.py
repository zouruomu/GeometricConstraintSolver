from source.geometry import Cuboid
from source.constraints import Clearance

def test_clearance():
    obj1 = Cuboid((5, 5, 0), (0, 0, 0), (1, 1, 1), "")
    obj2 = Cuboid((0, 0, 0), (0, 0, 0), (1, 1, 1), "")
    obj3 = Cuboid((3, 0, 0), (0, 0, 0), (1, 1, 1), "")
    assert Clearance([obj1, obj2, obj3]).badness() == 0.0

    obj1 = Cuboid((2, 0.5, 0), (0, 0, 0), (1, 1, 1), "")
    obj2 = Cuboid((0, 0, 0), (0, 0, 0), (1, 1, 1), "")
    obj3 = Cuboid((4, 0, 0), (0, 0, 0), (1, 1, 1), "")
    assert Clearance([obj1, obj2, obj3]).badness() == 1.0