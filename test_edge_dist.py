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

# v06 puts angles on half-edges
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v11.pkl')
gen_src.delete_orphan_edges()

gen_src.renumber_cells()

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

for c in gen_src.valid_cell_iter():
    try:
        qg=quads.QuadGen(gen_src,
                         cells=[c],
                         execute=False,
                         triangle_method='gmsh',
                         nom_res=3.5)

        # Can I speed this up a bit?
        # 46s, with 37 in create_final_by_patches
        #   15s 2300 delaunay node insertions, from trace_and_insert_contour
        #   13s in fields_to_xy
        # 10s in construct_matrix
        # What is the real value in trace_and_insert_contour?
        # probably could speed it up, but it's not worth the distraction
        g_final=qg.execute()
        grids.append(g_final)
    except:
        print()
        print("--------------------FAIL--------------------")
        print()
        continue
    
comb=unstructured_grid.UnstructuredGrid(max_sides=4)

for g in grids:
    comb.add_grid(g)

comb.write_ugrid('combined-20201026a.nc',overwrite=True)

##     
plt.clf()
#ccoll=g_final.plot_edges(color='orange',lw=0.4)
ccoll=g.plot_edges(color='k',lw=0.4)
plt.axis( (552047.9351268414, 552230.9809219765, 4124547.643451654, 4124703.282891116) )

##

# Generate all cells indepedently
