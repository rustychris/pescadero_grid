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
nice_colors=itertools.cycle(colors.TABLEAU_COLORS)

import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')
cmap=scmap.load_gradient('oc-sst.cpt')

##

from stompy.grid import triangulate_hole, rebay

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

# v06 puts angles on half-edges
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v06.pkl')

gen_src.renumber_cells()

# Adapt quads to having angles on half-edges, and dealing with multiple
# cells at once.

# Later add a convenience routine to copy from nodes.
qg=quads.QuadGen(gen_src,cells=[7,8,9],final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])

qg.execute()
qg.plot_result(num=100)
qg.g_final.write_ugrid('by_angle_output_cell00.nc',overwrite=True)

##

# cell 1
qg=quads.QuadGen(gen_src,cell=1,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=101)

qg.g_final.write_ugrid('by_angle_output_cell01.nc',overwrite=True)

##

# cell 2
qg=quads.QuadGen(gen_src,cell=2,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=102)

qg.g_final.write_ugrid('by_angle_output_cell02.nc',overwrite=True)

##

qg=quads.QuadGen(gen_src,cell=3,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=103)

qg.g_final.write_ugrid('by_angle_output_cell03.nc',overwrite=True)

##

qg=quads.QuadGen(gen_src,cell=4,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=104)

qg.g_final.write_ugrid('by_angle_output_cell04.nc',overwrite=True)

## 

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v05.pkl')

qg=quads.QuadGen(gen_src,cell=5,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result()

#qg.g_final.write_ugrid('by_angle_output_cell05.nc')

##

# Old Butano channel - fails unless nom_res is smaller, and can't have
# the large ragged edge.

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v05.pkl')

# This would benefit from some internal edges, or breaking into separate
# grids.

qg=quads.QuadGen(gen_src,cell=6,final='anisotropic',execute=False,
                 nom_res=3.5,
                 scales=[field.ConstantField(3.5),
                         field.ConstantField(3.5)])
qg.execute()
qg.plot_result()

qg.g_final.write_ugrid('by_angle_output_cell06.nc')


##
g_combined=None

for fn in ["by_angle_output.nc",
           "by_angle_output_cell01.nc",
           "by_angle_output_cell02.nc",
           "by_angle_output_cell03.nc",
           "by_angle_output_cell04.nc",
           "by_angle_output_cell05.nc",
           "by_angle_output_cell06.nc" ]:
    g=unstructured_grid.UnstructuredGrid.read_ugrid(fn)
    if g_combined is None:
        g_combined=g
    else:
        g_combined.add_grid(g)
        
##
plt.figure(1).clf()
g_combined.plot_edges(color='k',lw=0.5)
g_combined.plot_cells(color='0.85',zorder=-2,lw=0)

plt.axis('tight')
plt.axis('equal')
