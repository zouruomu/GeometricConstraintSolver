from ..geometry import Cuboid
from typing import Iterable

GeometricObject = Cuboid


class ConstraintProposition:
    """Abstract superclass representing a proposition asserting the satisfaction of a constraint between given objects.

    All constraints should inherit from this class. A constraint proposition is a proposition asserting
    that some constraint (defined by the badness method) is satisfied on the given set of objects
    passed at initialization (stored in self.arguments) -- for example, given a sub-class IsAligned, 
    the initialization of that sub-class might look like IsAligned((cube1, cube2), kwargs) and represent
    the proposition that cube1 and cube2 are aligned. To define a new constraint, one only needs to
    implement the badness method (how badly is the constraint violated?) and the define_arity method (what is
    the arity of this proposition?).

    Attributes:
        arity: int or None, the arity of this proposition. None denotes flexible arity.
        arguments: list (length equals arity) of objects of class GeometricObject, the arguments to constrain.
    """
    def __init__(self, arguments: Iterable[GeometricObject]):
        """Init method.
        
        Args: 
            arguments: A list-like of Object objects.   
        Returns:
            None
        Raises:
            ValueError if there is an arity mismatch.
        """
        arguments = list(arguments)
        # check to make sure list has correct arity
        if self.arity() is not None and len(arguments) != self.arity():
            raise ValueError(f"Input args has arity {len(arguments)}, but constraint was defined with arity {self.arity()}.")

        # if it does, store as attribute
        self.arguments = arguments

    @property
    def arity(self):
        """Define the arity of the constraint proposition, i.e. length of self.arguments.

        THIS METHOD SHOULD RETURN EITHER AN INTEGER OR NONE. NONE denotes flexible
        arity (the badness method will, in that case, have to be written accordingly
        to process a whole list and cannot make assumptions about its length).

        Args:
            None.
        Returns:
            An integer or None denoting arity.
        """
        raise NotImplementedError
        
    def badness(self):
        """Compute the badness of satisfaction of this constraint.
        
        Each type of constraint will have a different definition for badness of satisfaction
        and so will each need to implement this method separately. Furthermore, this method
        only operates on attributes and must take no additional arguments. Any information
        other than self.arguments should be stored as additional attributes by __init__.
        
        Args:
            None. 
        Returns:
            A scalar float value indicating badness of satisfaction of this constraint by self.arguments.
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        """To string method.
        """
        raise NotImplementedError