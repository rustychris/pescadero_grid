import heapq
from itertools import chain

import six

import matplotlib.pyplot as plt
import numpy as np

from stompy.grid import unstructured_grid
from stompy.grid import front, exact_delaunay
import stompy.grid.quad_laplacian as quads
from stompy.spatial import field
from stompy import utils


##
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(front)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(quads)

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v00.pkl')

from stompy.grid import rebay

        
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v00.pkl')
    
# This is the edge scale.
scale=field.ConstantField(10.0)

# Prepare a nice input akin to what quad laplacian will provide:
qg=quads.QuadGen(gen=gen_src,execute=False,cell=0)
qg.add_bezier(qg.gen)
gsmooth=qg.create_intermediate_grid_tri_boundary(src='IJ',scale=scale)

# quad laplacian does not currently do this step:
gsmooth.modify_max_sides(gsmooth.Nnodes())
gsmooth.make_cells_from_edges()
assert gsmooth.Ncells()>0

rad=rebay.RebayAdvancingDelaunay(grid=gsmooth,scale=scale,
                                 heap_sign=1)

##

rad.execute()

# Starting point: 8s
# 4.2s in edge_to_cells
# Don't ask for full updates -- still runs okay
# and now it's 4.4s.
# 2.1s of that is in add_node().  So exact_delaunay stuff.
# still have 1.2s in edge to cells.
# must be some internal call?

##

rad.plot_progress()
