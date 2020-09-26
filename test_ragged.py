# Southeast marsh channel
from matplotlib import collections
from collections import defaultdict
import stompy.grid.quad_laplacian as quads
from stompy.grid import exact_delaunay
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import six
from stompy import utils,filters
import numpy as np
from scipy import sparse


from scipy.optimize import fmin
import stompy.plot.cmap as scmap
from shapely import ops
turbo=scmap.load_gradient('turbo.cpt')
cmap=scmap.load_gradient('oc-sst.cpt')

##

# v00 has a ragged edge.
# v01 makes that a right. angle
from stompy.grid import triangulate_hole, rebay
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)


if 0: 
    gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v00.pkl')

    qg=quads.QuadGen(gen_src,cell=0,final='anisotropic',execute=True,nom_res=5,
                     gradient_scale=1.0)

    qg.plot_result()

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v02.pkl')
## 
# Testing a grid with a 360-degree vertex, and much more complicated
# patter
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

qg=quads.QuadGen(gen_src,cell=0,final='anisotropic',execute=False,nom_res=5)

qg.add_internal_edge([23,36])
qg.add_internal_edge([20,32])

qg.execute()

# All the code below is now in quad_laplacian.py

##

g=qg.g_final

plt.figure(4).clf()
fig,ax=plt.subplots(num=4)
g.plot_edges(color='k',lw=0.5)
g.plot_cells(color='0.8',lw=0,zorder=-2)
#g.plot_nodes(sizes=40,alpha=0.2,color='r')

ax.axis('off')
ax.set_position([0,0,1,1])

# Not too bad!

##

# Is there a cleaner way to get the initial set of patches?
