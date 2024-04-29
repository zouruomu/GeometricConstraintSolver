from .constraint import ConstraintProposition

class Proximity(ConstraintProposition):
    """
    NOTE: Currently it is only defined for 2D objects.
    """
    @staticmethod
    def arity():
        return 2 # binary

    def badness(self):
        obj1, obj2 = self.arguments
        dist = obj1.to_2d_rect().distance(obj2.to_2d_rect())
        dist = min(1, dist)
        return dist
    
    def __str__(self) -> str:
        """To string method.
        """
        return f"{str(self.arguments[0])} and {str(self.arguments[1])} must be proximal."