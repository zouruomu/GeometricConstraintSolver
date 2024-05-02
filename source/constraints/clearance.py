from .constraint import ConstraintProposition
from ..geometry import Cuboid
from typing import Iterable
from shapely import LineString, intersection

class Clearance(ConstraintProposition):
    """
    arguments[0] should not block arguments[1] and arguments[2] from each other.
    """
    def __init__(self, arguments: Iterable[Cuboid]):
        """Init method."""
        super().__init__(arguments)

    @staticmethod
    def arity():
        return 3

    def badness(self):
        """See superclass documentation for details.
        """
        obj0, obj1, obj2 = self.arguments
        rect0, rect1, rect2 = obj0.to_2d_rect(), obj1.to_2d_rect(), obj2.to_2d_rect()
        line = LineString([(rect1.cx, rect1.cy), (rect2.cx, rect2.cy)])
        intersection_line = intersection(rect0.get_contour(), line)
        return 1 - float(intersection_line.is_empty)

    
    def __str__(self) -> str:
        obj0, obj1, obj2 = self.arguments
        return f"{obj0.name} should not block {obj1.name} and {obj2.name} from each other."


    @classmethod
    def random(cls, arguments):
        raise ValueError("Random generation of Clearance constraint is not supported.")