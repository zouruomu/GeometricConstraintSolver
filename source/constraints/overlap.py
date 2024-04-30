from .constraint import ConstraintProposition, GeometricObject, Cuboid
from ..utils.scaled_sigmoid import scaled_sigmoid
import numpy as np
from typing import Iterable
from shapely.geometry import Polygon


class NoOverlap(ConstraintProposition):
    """All objects cannot overlap. THIS ONLY WORKS FOR CUBOIDS!!!

    NOTE: For 3d boxes to overlap, there must be overlap on ALL 3 dimensions. This constraint ensures that neither
          the x nor y directions have overlap, which implies that there cannot be overlap overall either.
    """
    def __init__(self, arguments: Iterable[Cuboid]):
        """Init method."""
        super().__init__(arguments)

    @staticmethod
    def arity():
        return -1

    def get_iou(self, obj1, obj2): # [[0,2,6,4],:2] get corners in correct order for Polygon
        polygon1 = Polygon(obj1.get_corners()[[0,2,6,4],:2].tolist())
        polygon2 = Polygon(obj2.get_corners()[[0,2,6,4],:2].tolist())
        intersection = polygon1.intersection(polygon2)
        iou = intersection.area / (polygon1.area + polygon2.area - intersection.area)
        return iou

    def badness(self):
        """See superclass documentation for details.
        """
        total_iou = 0
        for i in range(len(self.arguments)):
            for j in range(i, len(self.arguments)):
                if i == j:
                    continue
                total_iou += self.get_iou(self.arguments[i], self.arguments[j])
        # FIXME: if total overlap area is too high, sacled_sigmoid looks like 1 everywhere in that neighborhood
        #        and optimizer has trouble, ideally there is a reasonable way to normalize this value
        return total_iou
    
    def avg_iou(self):
        total_iou = 0
        for i in range(len(self.arguments)):
            for j in range(i, len(self.arguments)):
                if i == j:
                    continue
                total_iou += self.get_iou(self.arguments[i], self.arguments[j])
        return total_iou / (len(self.arguments) * (len(self.arguments) - 1))
    
    def all_iou(self):
        ious = []
        for i in range(len(self.arguments)):
            for j in range(i, len(self.arguments)):
                if i == j:
                    continue
                ious.append(self.get_iou(self.arguments[i], self.arguments[j]))
        return ious
    
    def __str__(self) -> str:
        """To string method.
        """
        obj_names = ""
        for obj in self.arguments[:-1]:
            obj_names += f"{str(obj)}, "
        obj_names += f"and {str(self.arguments[-1])}"
        return f"{obj_names} must not overlap."
    
    @classmethod
    def random(cls, arguments):
        raise ValueError("Random generation of NoOverlap constraint is not supported.")