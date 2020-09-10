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
six.moves.reload_module(rebay)

        
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

g=rad.extract_result()

##

plt.figure(1).clf()
g.plot_edges(color='k',lw=0.5)
g.plot_cells(color='tab:orange',alpha=0.5)

# HERE: offer this as an option in triangulate hole.

##


