from matplotlib import collections
from collections import defaultdict
import stompy.grid.quad_laplacian as quads
from stompy.grid import exact_delaunay, unstructured_grid
import matplotlib.pyplot as plt
import six
from stompy import utils,filters
from stompy.spatial import field
import numpy as np
from scipy import sparse
from shapely import ops

from matplotlib import colors
import itertools

import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')
cmap=scmap.load_gradient('oc-sst.cpt')

##

from stompy.grid import triangulate_hole, orthogonalize
from stompy.spatial import wkb2shp, constrained_delaunay

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(constrained_delaunay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(orthogonalize)
six.moves.reload_module(quads)

# 13 is good.
# 14 is playing with non-90 angles
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v14.pkl')

#gen_src.delete_orphan_edges()
#gen_src.renumber_cells()

if 1:
    plt.figure(1).clf()
    gen_src.plot_cells(labeler='id',centroid=True)
    plt.axis('tight')
    plt.axis('equal')

quads.prepare_angles_halfedge(gen_src)

gen_src.plot_edges(mask=np.isfinite(gen_src.edges['angle']),
                   color='r',lw=2)
quads.add_bezier(gen_src)
quads.plot_gen_bezier(gen_src)

## 
grids=[]


#cells=[10, 16, 17, 28, 35, 36, 37, 44, 45, 46] # FAILS
#cells=[10, 16, 17, 28, 35, 36, 37, 45, 46] # FAILS
#cells=[10, 16, 17, 28, 35, 36, 46]  # OKAY
#cells=[10, 16, 17, 28, 35, 36, 45, 46] # FAILS
#cells=[10, 16, 17, 28, 45, 46] # FAILS
#cells=[10, 16, 17, 45, 46] # FAILS
#cells=[10, 45, 46] # FAILS
#cells=[45, 46] #FAILS, 1 extra dof
#cells=[45] # OKAY
#cells=[46] # OKAY

# Really do need a way to specify spacing at a finer scale.

# Option:
#    Allow specifying edge count (maybe as negative scale).
#      Edge counts are automatically converted to scales when
#      building the interpolated scale.
#      Scales on an edge will be locally evaluated, i.e. not
#      subject to interpolation.
#    Swaths scan the edges perpendicular to the swath
#      if an edge has a count, contour values for the span
#      of that edge will be generated according to the count.
#    So I have a swath, with its constituent patches.
# 

# In the interest of being able to finish the grid,
# just avoid ragged edges.
# Can start to think about what it would take to have
# self-contained, cell-by-cell generation.
# Say each cell has to be a rectangle.
# Edges can have a target resolution, a target number
# of cells, or nothing.
# Nodes are positioned globally along the edges in the generating
# grid.  Counts must be consistent within each rectangle, and that's
# where the real challenge is.

# Some edges in gen can be fused (when a specific count is not
# given, but set from scale).
# Then all edges have a target number of nodes, and some measure
# of how tightly that is specified.
# Each cell then implies two equations, forcing the sum of
# i-steps and j-steps to both be zero.
# Edges with two adjacent cells will appear in 2 equations.
# There is probably an assumption somewhere in here that the
# cells are convex.  I'm okay with that.

# For unknowns, what if each cell gets two scalar unknowns,
# one corresponding to a stretching of +i vs -i edges, the
# other for +j/-j edges.

# Then each edge gets a target count and some scaling of the adjustments
# (possibly from two adjustment factors).

# Forget it -- just require that any edge shared by two cells has an exact
# count, and that all cells must be internally consistent.


#for c in [46]: # gen_src.valid_cell_iter():
#try:
qg=quads.QuadGen(gen_src,
                 cells=cells, # [c],
                 execute=False,
                 triangle_method='gmsh',
                 nom_res=3.5)

g_final=qg.execute()

# This is failing with ERROR:quad_laplacian:M.shape: (18287, 18288)
# i.e. underconstrained by 1 dof.
# also there are some errors early on:
# maybe hit a dead end -- boundary maybe not closed
# edge centered at [ 553644.60355219 4123404.63396081] traversed twice
# this is during psi setup.

qg.plot_psi_phi_setup()

# So I think that this is b/c the sting is now not quite 360deg, but
# should be treated more like it is?
# But I can't just put an nf triangle in there, can I?  What if
# it had a gentle angle on both sides?
#



## 
grids.append(g_final)
# except:
#     print()
#     print("--------------------FAIL--------------------")
#     print()
#     continue

##

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
# qg.g_not_final.plot_edges(ax=ax) # DT
qg.g_final2.plot_edges(ax=ax) # patches
qg.gen.plot_edges(labeler='scale',mask=qg.gen.edges['scale']>0,lw=0.5,color='k')

# Swaths may be smaller than an edge in gen. So gen might say put 5 edges here,
# but then the swath won't necessarily line up with that.
# A: set spacing of nodes at the gen level
# B: convert everything to resolution.

# When gridding 


##     
comb=unstructured_grid.UnstructuredGrid(max_sides=4)

for g in grids:
    comb.add_grid(g)

comb.write_ugrid('combined-20201026a.nc',overwrite=True)

# HERE:
#   Start adjusting resolutions, still on the separate grids.
#   Make a QGIS interface, so I can select one or more cells, generate
#   a grid.



##     
plt.clf()
#ccoll=g_final.plot_edges(color='orange',lw=0.4)
ccoll=g.plot_edges(color='k',lw=0.4)
plt.axis( (552047.9351268414, 552230.9809219765, 4124547.643451654, 4124703.282891116) )

##

# Generate all cells indepedently
