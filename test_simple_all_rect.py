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

from stompy.grid import triangulate_hole, orthogonalize,shadow_cdt, front
from stompy.spatial import wkb2shp, constrained_delaunay

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(constrained_delaunay)
six.moves.reload_module(shadow_cdt)
six.moves.reload_module(front)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(orthogonalize)
six.moves.reload_module(quads)


ver='v19'
gen_src=unstructured_grid.UnstructuredGrid.read_pickle(f'grid_lagoon-{ver}.pkl')

##
if 0:
    plt.figure(10).clf()
    gen_src.plot_cells(labeler='id',centroid=True)
    plt.axis('tight')
    plt.axis('equal')

## 
sqg=quads.SimpleQuadGen(gen_src,cells=list(gen_src.valid_cell_iter()),
                        nom_res=2.5,execute=False)
sqg.execute()

## 
g=sqg.g_final
g.renumber()

g.write_pickle(f'all-quads-{ver}.pkl',overwrite=True)

##


g=unstructured_grid.UnstructuredGrid.read_pickle(f'all-quads-{ver}.pkl')

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

g_orig.write_pickle(f'quads_and_lines-{ver}.pkl',overwrite=True)
##

plt.figure(1).clf()
g_orig.plot_edges(lw=0.5,color='k')

##

six.moves.reload_module(unstructured_grid)

g=unstructured_grid.UnstructuredGrid.read_pickle(f'quads_and_lines-{ver}.pkl')

# Calculate a global apollonius scale field.  Otherwise some of the
# internal edges can't see nearby short edges, and we get mismatches.
# This makes a chunky field, and slows down gmsh considerably.
e2c=g.edge_to_cells(recalc=True)

# I want edges on the boundary of the quad regions
j_sel=(e2c.min(axis=1)<0) & (e2c.max(axis=1)>=0)

# Refine this by ignoring edges that won't be triangulated anyway.
# This boundary doesn't care about cells, safe to use with a partial
# grid.
boundary=g.boundary_cycle()
for a,b in zip(boundary,np.roll(boundary,-1)):
    j=g.nodes_to_edge([a,b])
    j_sel[j]=False
    
el=g.edges_length()
ec=g.edges_center()

from stompy.spatial import field
density=field.PyApolloniusField(X=ec[j_sel],F=el[j_sel],
                                redundant_factor=0.95)
                                
# Fill some holes!
# th_kwargs=dict(method='gmsh',density=density)

th_kwargs=dict(method='front',density=density,
               method_kwargs=dict(reject_cc_outside_cell=True,
                                  # 0.05: only one failure.
                                  # 0.10: one failure
                                  # 0.20: 9 failures
                                  reject_cc_distance_factor=0.15))

from stompy.grid import triangulate_hole

##
six.moves.reload_module(front)
six.moves.reload_module(triangulate_hole)

if 0: # Debugging bad slides
    seed=[552596., 4123395.] # South of nexus in Butano marsh
    AT=triangulate_hole.triangulate_hole(g,seed_point=seed,dry_run=True,
                                         return_value='front',splice=False,
                                         **th_kwargs)

##--------------------
seed_points=[
    [552596., 4123395.], # various in Butano Marsh. Fails if too late in the sequence
    [552648., 4124387.], # between N marsh and Pescadero, east side
    [552905., 4123225.], # x_butano_marsh_s # This is segfaulting!!
    [552841., 4124582.], # x_north_marsh
    [553257., 4123782.], # x_pesc_roundhill
    [552498., 4125123.], # x_north_pond
    [552384., 4124450.], # x_lagoon_shallow
    [552516., 4124182.], # x_butano_lagoon
    [552607., 4123680.], # x_butano_marsh_w
    [552771., 4124233.], # F x_delta_marsh
    [552844., 4123945.], # x_delta_marsh_s
    [553560., 4123089.], # x_butano_se
    [552379., 4124697.], # x_nmarsh_west
    [552323., 4124492.], # x_lagoon_north
    [552226., 4124428.], # seg fault! x_lagoon_south
    [552547., 4123962.], # butano off-channel storage
    [552581., 4123866.], # another butano off-channel storage
    [552576., 4123458.], # Funny intersection place
    [553116., 4123723.], # extra point in butano marsh w/ tangent
    [552746., 4123550.], # butano between old and new.
    [552331., 4124869.], # small area on west side of N Pond channel
]

##

# With reject_cc_distance_factor=0.15, skipped 1 due to segfault, and 3 others fail.
#
failures=[]

for dist_fact in [0.15, 0.10,None]:
    th_kwargs['method_kwargs']['reject_cc_distance_factor']=dist_fact
    g_new=g
    for seed in seed_points:
        print("-------------- %s ---------------"%str(seed))
        c=g_new.select_cells_nearest(seed,inside=True)
        if c is not None:
            print("Point already filled"%seed)
            continue
        print("Will fill this one")

        result=triangulate_hole.triangulate_hole(g_new,seed_point=seed,**th_kwargs)
        if isinstance(result,unstructured_grid.UnstructuredGrid):
            g_new=result
        else:
            print("-----Fail on point: %s-----"%seed)
            failures.append( result )
        
## 

plt.figure(1).clf()
for fail in failures:
    fail.grid.plot_edges(color='tab:red',lw=0.7)

if 0:
    g.cells_area()
    clear=g.edge_clearance(recalc_e2c=True,mode='double')
    coll=g_new.plot_edges(values=clear,cmap='jet',clim=[0,0.2],lw=0.8)
    plt.colorbar(coll)
else:
    coll=g_new.plot_edges(color='0.5',lw=0.7)
plt.axis('tight')
plt.axis('equal')

##         
plt.figure(1).clf()
g_new.plot_edges(color='tab:brown',lw=0.7)
plt.axis('tight')
plt.axis('equal')

## 
if len(failures)==0:
    g_new.renumber()
    g_new.write_ugrid(f'quad_tri_{ver}frontcc.nc',overwrite=True)
else:
    g_new.write_pickle(f'{ver}-successes.pkl',overwrite=True)


##

# Had to add a few cells manually at the sting. Could hard-code that
# in.  Sort of annoying.
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(orthogonalize)
    
g=unstructured_grid.UnstructuredGrid.read_ugrid('quad_tri_v19frontcc.nc')

# Automatically adjust for orthogonality:
tweaker=orthogonalize.Tweaker(g=g)
tweaker.merge_all_by_edge_clearance()
# That merges a bunch.  Probably too many.

tweaker.nudge_all_orthogonal()


## 
plt.figure(1).clf()

coll=g.plot_edges(values=bad_edges2,cmap='jet_r',clim=[0,0.25])

bad_cells=np.unique( [ g.edge_to_cells(j) for j in np.nonzero(bad_edges<0.05)[0]] )
bad_cells=bad_cells[ bad_cells>=0 ]

cc=g.cells_center()
plt.colorbar(coll)

plt.plot( cc[bad_cells,0], cc[bad_cells,1], 'g.')

plt.axis('tight')
plt.axis('equal')


## 
# Starts at 38.  It fixes all of the edges, but
# at cost of orthogonality.
bad_edges=g.edge_clearance(recalc_e2c=True)<0.05
print(f"Bad edges: {bad_edges.sum()}/{len(bad_edges)}")

for j in np.nonzero(bad_edges)[0]:
    print(j)
    tweaker.adjust_for_edge_quality(j)

##
# this works out okay, but the cells look kind of crappy.
tweaker.nudge_all_orthogonal()
    
## 


# And short edges:
# Testing..
# zoom=(552529.6383148731, 552547.6935756239, 4124310.2765580793, 4124320.431186181)
# edge_mask=np.nonzero(g.edge_clip_mask(zoom))[0]

zoom=(552484.5867645434, 552515.8451034953, 4124264.0836399, 4124287.376144087)
def figure_quality(num=2):
    edge_quality=g.edge_clearance(recalc_e2c=True)
    cell_errors=g.circumcenter_errors(radius_normalized=True)
    thresh_single_sided=0.05

    e2c=g.edge_to_cells()
    cc=g.cells_center()

    bad_edges=edge_quality<thresh_single_sided

    plt.figure(num).clf()
    g.plot_edges(color='0.7',lw=0.6)
    if np.any(bad_edges):
        g.plot_edges(mask=bad_edges,values=edge_quality,cmap='jet',clim=[0,0.1])

        bad_cells=np.unique(e2c[bad_edges>0,:].ravel())
        bad_cells=bad_cells[bad_cells>=0]
        plt.plot( cc[bad_cells,0],cc[bad_cells,1],'g.')

    g.plot_cells(values=cell_errors,clim=[0,0.03],cmap='copper_r')

    plt.axis('tight')
    plt.axis('equal')
    plt.axis(zoom)

figure_quality(2)


tweaker.adjust_for_edge_quality(j=106719,expand=True)
    

##

figure_quality(2)
g.plot_nodes(mask=nodes,color='k')
plt.axis(zoom)

##



