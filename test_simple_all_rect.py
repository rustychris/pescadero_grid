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
plt.figure(10).clf()
gen_src.plot_cells(labeler='id',centroid=True)
plt.axis('tight')
plt.axis('equal')

##
six.moves.reload_module(quads)

results=[]
for lowpass in [True,False]:
    quads.SimpleSingleQuadGen.lowpass_ds_dval=lowpass
    sqg=quads.SimpleQuadGen(gen_src,cells=[53],
                            nom_res=2.5,execute=False)
    sqg.execute()
    results.append(sqg.g_final)

##

plt.figure(2).clf()
fig,axs=plt.subplots(2,1,num=2)
results[0].plot_edges(color='tab:blue',ax=axs[0])
results[1].plot_edges(color='tab:blue',ax=axs[1])
for ax in axs:
    ax.axis('tight')
    ax.axis('equal')

## 
# lowpass or no doesn't make a difference.

scale=sqg.qgs[0].scales[1]
g_int=sqg.qgs[0].g_int
g_int_scale=scale( g_int.nodes['x'])

scat=g_int.contourf_node_values(g_int_scale,20,cmap='jet')
plt.colorbar(scat)

##
sqg=quads.SimpleQuadGen(gen_src,cells=list(gen_src.valid_cell_iter()),
                        nom_res=2.5,execute=False)
sqg.execute()

## 
g=sqg.g_final
g.renumber()

g.write_pickle('all-quads-v16.pkl',overwrite=True)

## 
g=unstructured_grid.UnstructuredGrid.read_pickle('all-quads-v16.pkl')

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

merge_nodes=[]
for nB in n_to_match:
    # Might want to limit this to boundary nodes...
    nA=g.select_nodes_nearest( gen_src_tri.nodes['x'][nB] )
    merge_nodes.append( [nA,nB] )

g.add_grid(gen_src_tri,merge_nodes=merge_nodes)

g.renumber(reorient_edges=False)

g_orig=g

g_orig.write_pickle('quads_and_lines-v16.pkl',overwrite=True)
##

plt.figure(1).clf()
g_orig.plot_edges(lw=0.5,color='k')

##

g=unstructured_grid.UnstructuredGrid.read_pickle('quads_and_lines-v16.pkl')

# Fill some holes!
#th_kwargs=dict(method='gmsh')
th_kwargs=dict(method='front',method_kwargs=dict(reject_cc_outside_cell=False))
##
               
from stompy.grid import triangulate_hole
g_new=g
x_north_pond=[552498., 4125123.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_north_pond,**th_kwargs)
# front okay so far.
x_lagoon_shallow=[552384., 4124450.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_lagoon_shallow,**th_kwargs)

# Front okay.
x_north_marsh=[552841., 4124582.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_north_marsh,**th_kwargs)

# okay

x_butano_lagoon=[552516., 4124182.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_lagoon,**th_kwargs)

# yes...
## 
x_butano_marsh_w=[552607., 4123680.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_marsh_w,**th_kwargs)

# yes. fixed!
## 
x_butano_marsh_s=[552905., 4123225.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_marsh_s,**th_kwargs)

g_new.write_pickle('prebug.pkl',overwrite=True)

## 
# six.moves.reload_module(unstructured_grid)
g_new=unstructured_grid.UnstructuredGrid.read_pickle('prebug.pkl')

x_delta_marsh=[552771., 4124233.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_delta_marsh,**th_kwargs)

x_delta_marsh_s=[552844., 4123945.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_delta_marsh_s,**th_kwargs)

x_pesc_roundhill=[553257., 4123782.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_pesc_roundhill,**th_kwargs)

x_butano_se=[553560., 4123089.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_butano_se,**th_kwargs)

x_nmarsh_west=[552379., 4124697.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_nmarsh_west,**th_kwargs)

x_lagoon_north=[552323., 4124492.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_lagoon_north,**th_kwargs)

x_lagoon_south=[552226., 4124428.]
g_new=triangulate_hole.triangulate_hole(g_new,seed_point=x_lagoon_south,**th_kwargs)

##
g_new.write_pickle('prebug2.pkl',overwrite=True)

##
g_new=unstructured_grid.UnstructuredGrid.read_pickle('prebug2.pkl')

# Need to add a few more seed points.
for seed in [[552547., 4123962.], # butano off-channel storage
             [552581., 4123866.], # another butano off-channel storage
             [552596., 4123395.], # various in Butano Marsh
             [552576., 4123458.], # Funny intersection place
             [552746., 4123550.]]:
    g_new=triangulate_hole.triangulate_hole(g_new,seed_point=seed,**th_kwargs)

## 
plt.figure(1).clf()
g_new.plot_edges(color='tab:blue',lw=0.7)
plt.axis('tight')
plt.axis('equal')


##

g_new.renumber()
g_new.write_ugrid('quad_tri_v16.nc',overwrite=True)

# This finishes.
# There a lot of places where gmsh has split edges.
# Several things to do:
# A The resolution smoothing in quad_laplacian should be more scale aware,
#   so that it smooths at the level of a couple grid cells, rather than
#   whatever arbitrary thing it's currently doing. There is probably something
#   more clever that can be done, but this should be okay.

# HERE. Something is off. Even nailing down the resolution strongly, it
# gets smeared out much more than I'd expect.  Isolate the output of
# patch_contour,


# B See if front does any better. => It is getting through things slowly.a
# C Adjust the resolutions to have smoother transitions
# D Could post-process the spliced grid after triangulate_hole to automatically
#   split edges.
# 

