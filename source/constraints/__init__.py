from .proximity import Proximity
from .symmetry import Symmetry
from .alignment import TranslationalAlignment
from .target import Target
from .perpendicularity import Perpendicularity
from .parallelism import Parallelism
from .overlap import NoOverlap
from .direction import Direction
from .cover import Cover
from .clearance import Clearance


all_constraints = [Proximity, Symmetry, TranslationalAlignment, Target, Perpendicularity, Parallelism, NoOverlap, Direction, Cover]