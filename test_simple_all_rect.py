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
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v15.pkl')

# 21, lagoon: good
# 20, Lower Pescadero: good now.

# sqg=quads.SimpleQuadGen(gen_src,cells=[0,1,2,3,4,7,8,9,10,11,12,14,16,17,
#                                        19, # questionable.
#                                        20,21,22,23,24,25,26,27,28,29,30,
#                                        33,34,35,36,
#                                        37,38,39,40,41])

sqg=quads.SimpleQuadGen(gen_src,cells=[25],execute=False)
sqg.execute()

## 
plt.figure(1).clf()
colors=plt.rcParams['axes.prop_cycle']
for g,col in zip(sqg.grids,colors()):
    g.plot_edges(**col)

plt.axis('tight')
plt.axis('equal')

##

plt.figure(2).clf()
gen_src.plot_cells(labeler='id',centroid=True)
plt.axis('tight')
plt.axis('equal')

