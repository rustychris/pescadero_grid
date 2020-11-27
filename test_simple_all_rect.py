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

sqg=quads.SimpleQuadGen(gen_src,cells=[4],
                        nom_res=2.5,execute=False)
sqg.execute()
plt.figure(1).clf()
gen_src.plot_edges(color='tab:blue',lw=0.5,alpha=0.5)
sqg.g_final.plot_edges(color='k')


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

##
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v16.pkl')
quads.SimpleSingleQuadGen.lowpass_ds_dval=False
sqg=quads.SimpleQuadGen(gen_src,cells=[60],
                        nom_res=2.5,execute=False)
sqg.execute()
qg=sqg.qgs[0]
##



##
plt.figure(1).clf()
#qg.g_final.plot_edges()

# These contours looks exactly even.
qg.g_int.contour_node_values( qg.psi, j_contours,colors='r',linestyles='-' )

# This
scale=qg.scales[1](qg.g_int.nodes['x'])
cset=qg.g_int.contourf_node_values( scale,50,cmap='jet' )
plt.colorbar(cset)
##
# This is the call in quads:2641
# This is the correct matching of j_contours and psi field.
# This is the correct scale, and varies from 2ish in the middle to 6 ish on the
# ends.

# recalculating contours:
def patch_contours(g_int,node_field,scale,count=None,Mdx=None,Mdy=None,
                   lowpass_ds_dval=True,variable_scale=True):
    """
    Given g_int, a node field (psi/phi) defined on g_int, a scale field, and 
    a count of edges, return the contour values of the node field which
    would best approximate the requested scale.

    g_int: UnstructuredGrid
    node_field: a psi or phi field defined on the nodes of g_int
    scale: length scale Field with domain include g_int
    count: if specified, the number of nodes in the resulting discretization
    
    Mdx,Mdy: matrix operators to calculate derivatives of the node field. 
      by default create from scratch

    variable_scale: if True, consider variation of scale within the patch.

    returns the contour values (one more than the number of edges)
    """
    field_min=node_field.min()
    field_max=node_field.max()

    # original swath code had to pull out a subset of node in the node
    # field, but now we can assume that g_int is congruent to the target
    # patch.
    swath_nodes=np.arange(g_int.Nnodes())
    swath_vals=node_field[swath_nodes]

    # preprocessing for contour placement
    nd=quads.NodeDiscretization(g_int)
    if Mdx is None:
        Mdx,_=nd.construct_matrix(op='dx') # could be saved between calls.
    if Mdy is None:
        Mdy,_=nd.construct_matrix(op='dy') #
        
    field_dx=Mdx.dot(node_field)
    field_dy=Mdy.dot(node_field)
    field_grad=np.sqrt( field_dx**2 + field_dy**2 ) # looks okay
    swath_grad=field_grad

    order=np.argsort(swath_vals)
    o_vals=swath_vals[order] 
    o_dval_ds=swath_grad[order] # s: coordinate perpendicular to contours
    
    local_scale=scale( g_int.nodes['x'][swath_nodes[order]] )

    # local_scale is ds/di or ds/dj
    o_ds_dval=1./(o_dval_ds*local_scale)

    # trapezoid rule integration
    d_vals=np.diff(o_vals) # some zeros at the ends.
    # Particularly near the ends there are a lot of
    # duplicate swath_vals.
    # Try a bit of lowpass to even things out.
    if lowpass_ds_dval:
        # HERE this needs to be scale-aware!!
        # could use count, but we may not have it, and
        # it ends up being evenly spread out, which isn't
        # ideal.
        # How bad is it to just drop this? Does have an effect,
        # particularly in the lateral
        if count is None:
            # There is probably something more clever to do using the actual
            # swath vals.
            winsize=int(len(o_vals)/10)
        else:
            winsize=int(len(o_vals)/count)
        if winsize>1:
            o_ds_dval=filters.lowpass_fir(o_ds_dval,winsize)
    else:
        print("No lowpass on ds_dval")

    s=np.cumsum(d_vals*0.5*(o_ds_dval[:-1]+o_ds_dval[1:]))
    s=np.r_[0,s]

    # calculate this from resolution
    # might have i/j swapped.  range of s is 77m, and field
    # is 1. to 1.08.  better now..
    if count is None:
        count=max(2,int(np.round(s.max())))
        # Before s was geographic distance, but now s is i or j index
        #n_swath_cells=int(np.round( (s.max() - s.min())/local_scale.mean()))
        #n_swath_cells=max(1,n_swath_cells)
        
    s_contours=np.linspace(s[0],s[-1],count)
    adj_contours=np.interp( s_contours,
                            s,o_vals)
    
    adj_contours[0]=field_min
    adj_contours[-1]=field_max

    assert np.all(np.diff(adj_contours)>0),"should be monotonic, right?"

    return adj_contours


self=qg
#import pdb
#pdb.run( """
j_contours=patch_contours(self.g_int,self.psi,
                          self.scales[1],
                          len(self.left_j), # 55.  okay.
                          lowpass_ds_dval=False)
#""")

