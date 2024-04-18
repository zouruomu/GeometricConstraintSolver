from GeoConstraintSolver import *

problem = Problem()

# add objects
cube1 = Cuboid(loc=[0,0,0], rot=[0,0,0], scale=[2,2,2])
problem.add_optimizable_object(cube1, "cube1")
cube2 = Cuboid(loc=[5,5,5], rot=[0,0,0], scale=[1,1,1])
problem.add_optimizable_object(cube2, "cube2")

# plot initial state
problem.plot(fixed_axes=10)
print(f"Cube 1 initial coordinates: {cube1.loc}")
print(f"Cube 2 initial coordinates: {cube2.loc}")

# add constraint propositions
problem.add_constraint_proposition(IsAligned(arguments=(cube1,cube2),
                                             dimension="x",
                                             location="center"), weight=1)
problem.add_constraint_proposition(IsAligned(arguments=(cube1,cube2),
                                             dimension="y",
                                             location="center"), weight=1)
problem.add_constraint_proposition(IsAligned(arguments=(cube1,cube2),
                                             dimension="z",
                                             location="center"), weight=1)

# solve
problem.solve()

# plot final state
problem.plot(fixed_axes=10)
print(f"Cube 1 final coordinates: {cube1.loc}")
print(f"Cube 2 final coordinates: {cube2.loc}")