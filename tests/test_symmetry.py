
import numpy as np
import random
from source.geometry import Cuboid
from source.constraints import Symmetry


def test_symmetry():
    obj0 = Cuboid((1, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")    
    obj2 = Cuboid((2, 0, 0), (0, 0, 0), (2, 2, 2), "")
    assert np.isclose(Symmetry([obj0, obj1, obj2]).badness(), 0)

    obj0 = Cuboid((1, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj1 = Cuboid((2, 0, 0), (0, 0, 0), (2, 2, 2), "")    
    obj2 = Cuboid((3, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj3 = Cuboid((4, 0, 0), (0, 0, 0), (2, 2, 2), "")
    assert np.isclose(Symmetry([obj0, obj1, obj2, obj3]).badness(), 0)

    obj0 = Cuboid((0, 1, 0), (0, 0, 0), (2, 2, 2), "")
    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")    
    obj2 = Cuboid((0, 2, 0), (0, 0, 0), (2, 2, 2), "")
    assert np.isclose(Symmetry([obj0, obj1, obj2]).badness(), 0)

    obj0 = Cuboid((1, 1, 0), (0, 0, 0), (2, 2, 2), "")
    obj1 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")    
    obj2 = Cuboid((2, 0, 0), (0, 0, 0), (2, 2, 2), "")
    assert np.isclose(Symmetry([obj0, obj1, obj2]).badness(), 0)

    obj0 = Cuboid((0, 0, 0), (0, 0, 0), (2, 2, 2), "")
    obj1 = Cuboid((-1, -1, 0), (0, 0, 0), (2, 2, 2), "")    
    obj2 = Cuboid((1, 1, 0), (0, 0, 0), (2, 2, 2), "")
    assert np.isclose(Symmetry([obj0, obj1, obj2]).badness(), 2 / 3)
    

    # test if the results between 0 and 1
    for _ in range(10):
        n = random.randint(3, 10)
        objs = [Cuboid((random.random(), random.random(), random.random()), (random.random(), random.random(), random.random()), (random.random(), random.random(), random.random()), "") for _ in range(n)]
        sym = Symmetry(objs)
        # print(sym.badness())
        assert 0 <= sym.badness() <= 1
