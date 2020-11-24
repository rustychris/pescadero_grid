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

from stompy.grid import triangulate_hole, orthogonalize,shadow_cdt
from stompy.spatial import wkb2shp, constrained_delaunay

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(constrained_delaunay)
six.moves.reload_module(shadow_cdt)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(orthogonalize)
six.moves.reload_module(quads)

# 13 is good.
# 14 is playing with non-90 angles
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v15.pkl')

## 

sqg=quads.SimpleQuadGen(gen_src,cells=list(gen_src.valid_cell_iter()))
sqg.execute()

## 
g=unstructured_grid.UnstructuredGrid(max_sides=4)
for sub_g in sqg.grids:
    g.add_grid(sub_g,merge_nodes='auto',tol=0.01)

# Maybe an issue with some nodes not getting merged. In particular,
# there is a pair of nodes in the crook of the north pond split.
g.renumber()

if 0:
    plt.figure(1).clf()
    g.plot_edges(lw=0.7,color='tab:blue')

    plt.axis('tight')
    plt.axis('equal')

    zoom=(552354.2071988618, 552371.1522911632, 4124869.5447136736, 4124899.120904969)

    # This puts 9354 at the crook.
    g.plot_nodes(clip=zoom,labeler='id')

# Add the non-cell edges of gen_src back into g:
gen_src_tri=gen_src.copy()
e2c=gen_src_tri.edge_to_cells(recalc=True)
j_to_delete=np.any(e2c>=0,axis=1)

# These nodes in gen_src_tri will be matched up with nodes in the merged grid
n_to_match=[]
for n in gen_src_tri.valid_node_iter():
    js=gen_src_tri.node_to_edges(n)
    if len(js)<2:
        print("Weird node %d"%n)
        continue
    
    js_status=j_to_delete[js]

    if np.all(js_status):
        pass # Node only participates in edges that are part of cells.
    elif np.any(js_status):
        n_to_match.append(n)
    else:
        pass

for j in np.nonzero(j_to_delete)[0]:
    gen_src_tri.delete_edge_cascade(j)

gen_src_tri.delete_orphan_nodes()

if 0:
    gen_src_tri.plot_edges(color='k',lw=0.5)
    gen_src_tri.plot_nodes(color='r') 
    gen_src_tri.plot_nodes(mask=n_to_match,color='r')

merge_nodes=[]
for nB in n_to_match:
    # Might want to limit this to boundary nodes...
    nA=g.select_nodes_nearest( gen_src_tri.nodes['x'][nB] )
    merge_nodes.append( [nA,nB] )

g.add_grid(gen_src_tri,merge_nodes=merge_nodes)

##

g.renumber()

g_orig=g

##

g_orig.write_pickle('ready_for_triangles.pkl',overwrite=True)

##
g=unstructured_grid.UnstructuredGrid.read_pickle('ready_for_triangles.pkl')
g.cells['_area']=np.nan
g.orient_cells()

##

# Fill some holes!
x_north_pond=[552498., 4125123.]

from stompy.grid import triangulate_hole

g_new=triangulate_hole.triangulate_hole(g,seed_point=x_north_pond,method='gmsh',
                                        splice=True)

##

plt.figure(1).clf()
g.plot_edges(color='tab:blue',lw=3,alpha=0.5)
#g_new.plot_edges(color='tab:red')

##


##
plt.figure(1).clf()
g.plot_edges(color='k',lw=0.5)
plt.axis('tight')
plt.axis('equal')

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

