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
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v16.pkl')

## 

sqg=quads.SimpleQuadGen(gen_src,cells=list(gen_src.valid_cell_iter()),
                        nom_res=2.5,execute=False)
sqg.execute()

## 
g=sqg.g_final

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

g.renumber(reorient_edges=False)

g_orig=g

g_orig.write_pickle('ready_for_triangles.pkl',overwrite=True)

## 
g=unstructured_grid.UnstructuredGrid.read_pickle('ready_for_triangles.pkl')
g.cells['_area']=np.nan
g.orient_cells()

# Fill some holes!

from stompy.grid import triangulate_hole
g_new=g
x_north_pond=[552498., 4125123.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_north_pond,method='gmsh')

x_lagoon_shallow=[552384., 4124450.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_lagoon_shallow,method='gmsh')

x_north_marsh=[552841., 4124582.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_north_marsh,method='gmsh')

x_butano_lagoon=[552516., 4124182.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_lagoon,method='gmsh')

x_butano_marsh_w=[552607., 4123680.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_marsh_w,method='gmsh')

x_butano_marsh_s=[552905., 4123225.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_marsh_s,method='gmsh')

g_new.write_pickle('prebug.pkl',overwrite=True)
##
six.moves.reload_module(unstructured_grid)
g_new=unstructured_grid.UnstructuredGrid.read_pickle('prebug.pkl')

## 
# This has another issue: there is a 'hint' edge. So far I've focused on hint
# nodes, but if there is an edge with no cells but joining two nodes that do
# have cells, it is getting left in the grid, and causes problems.
six.moves.reload_module(triangulate_hole)
x_delta_marsh=[552771., 4124233.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_delta_marsh,method='gmsh')

## 
x_delta_marsh_s=[552844., 4123945.]

g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_delta_marsh_s,method='gmsh',
                                        max_nodes=g_new.Nnodes())

##

x_pesc_roundhill=[553257., 4123782.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_pesc_roundhill,method='gmsh')

##

x_butano_se=[553560., 4123089.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_se,method='gmsh')

## 
plt.figure(1).clf()
g_new.plot_edges(color='tab:blue',lw=1)
plt.axis('tight')
plt.axis('equal')

##

g_new.renumber()
g_new.write_ugrid('quad_tri_15.nc',overwrite=True)

