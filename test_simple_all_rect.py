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

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v18.pkl')

##
if 0:
    plt.figure(10).clf()
    gen_src.plot_cells(labeler='id',centroid=True)
    plt.axis('tight')
    plt.axis('equal')

##

sqg=quads.SimpleQuadGen(gen_src,cells=[],
                        nom_res=2.5,execute=False)
sqg.execute()

qg=sqg.process_one_cell(51,smooth_patch=False)

@utils.add_to(qg)
def nudge_boundaries_monotonic(self):
    """
    Check that boundary nodes have monotonic psi/phi, and 
    nudge any violating nodes to be linearly interpolated between
    the okay nodes.

    Updates pp and x for offending nodes.
    """
    nodes_r=self.patch_elts['nodes']

    for nlist,coord,sign in [ (nodes_r[:,0],1,-1),
                              (nodes_r[:,-1],1,-1),
                              (nodes_r[0,:],0,1),
                              (nodes_r[-1,:],0,1)]:
        # nlist: node indices into patch
        # coord: which coordinate of pp to adjust
        # sign: +1 for increasing, -1 for decreasing
        vals=self.patch.nodes['pp'][nlist,coord]
        if np.all(sign*np.diff(vals)>0): continue
        
        rigid=self.patch.nodes['rigid'][nlist]
        # At least the rigid ones better be monotonic.
        assert np.all(sign*np.diff(vals[rigid]))>0

        i=np.arange(len(vals))
        # each entry is the index of the next rigid node, self included.
        i_next=i[rigid][ np.searchsorted(i[rigid],i) ]
        # each entry is the index of the previous rigid node, self included
        i_prev=i[rigid][::-1][ np.searchsorted(-i[rigid][::-1],-i) ]
        assert np.all( i_prev[ i[rigid] ] == i[rigid] )
        assert np.all( i_next[ i[rigid] ] == i[rigid] )

        bad= (~rigid) & ( (sign*vals[i_prev] >= sign*vals) | (sign*vals[i_next]<=sign*vals))
        print(bad.sum())
        vals[bad] = np.interp(i[bad], i[~bad], vals[~bad] )
        self.patch.nodes['pp'][nlist,coord]=vals
        for n in nlist[bad]:
            x=self.g_int.fields_to_xy(self.patch.nodes['pp'][n],[self.psi,self.phi],
                                      self.patch.nodes['x'][n])
            self.patch.nodes['x'][n]=x


# This is failing. There is some overlap between the sets of rigid
# nodes.  In theory, this should be fine. The choice of resolution
# in the upper right corner might be making things worse, but
# it should still complete.
## 
# qg.nudge_boundaries_monotonic() # seems to work

# The smoothing is problematic.  It operates on displacements,
# so doesn't "know" that there is a kink.  And the anisotropy
# means that bad spacing along the boundary gets projected
# along contours.
# Hack solution is to decrease anisotropy until boundaries are
# monotonic.

qg.smooth_patch_psiphi_implicit()

pp_r=qg.patch.nodes['pp'][qg.patch_elts['nodes']]
psi_mono=np.all(np.diff(pp_r[:,:,0],axis=1)>0)
phi_mono=np.all(np.diff(pp_r[:,:,1],axis=0)<0)


## 
plt.figure(1).clf()

#qg.g_final.plot_edges(lw=3,alpha=0.3)
qg.patch.plot_edges()

#plt.plot( qg.left_j[:,0], qg.left_j[:,1], 'g-o')
#plt.plot( qg.right_j[:,0], qg.right_j[:,1], 'g-o')

# The unsmoothed patch doubles back on itself on the left.
# Try a pass between position_patch_nodes and smooth, that
# enforces monotonicity along the boundary.

plt.axis('tight')
plt.axis('equal')
plt.axis( (552557.885928849, 552622.3543178033, 4124415.7229523533, 4124463.7623002506) )

            
## 
sqg=quads.SimpleQuadGen(gen_src,cells=list(gen_src.valid_cell_iter()),
                        nom_res=2.5,execute=False)
sqg.execute()

## 
g=sqg.g_final
g.renumber()

g.write_pickle('all-quads-v18.pkl',overwrite=True)

## 
g=unstructured_grid.UnstructuredGrid.read_pickle('all-quads-v18.pkl')

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

g_orig.write_pickle('quads_and_lines-v18.pkl',overwrite=True)
##

plt.figure(1).clf()
g_orig.plot_edges(lw=0.5,color='k')

##

g=unstructured_grid.UnstructuredGrid.read_pickle('quads_and_lines-v18.pkl')

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
    
## 
el=g.edges_length()
ec=g.edges_center()

from stompy.spatial import field
density=field.PyApolloniusField(X=ec[j_sel],F=el[j_sel],
                                redundant_factor=0.95)
                                
##

# Fill some holes!
# th_kwargs=dict(method='gmsh',density=density)
th_kwargs=dict(method='front',density=density,
               method_kwargs=dict(reject_cc_outside_cell=True))

from stompy.grid import triangulate_hole

seed_points=[
    [552498., 4125123.], # x_north_pond
    [552384., 4124450.], # x_lagoon_shallow
    [552841., 4124582.], # x_north_marsh
    [552516., 4124182.], # x_butano_lagoon
    [552607., 4123680.], # x_butano_marsh_w
    [552905., 4123225.], # x_butano_marsh_s
    [552771., 4124233.], # x_delta_marsh
    [552844., 4123945.], # x_delta_marsh_s
    [553257., 4123782.], # x_pesc_roundhill
    [553560., 4123089.], # x_butano_se
    [552379., 4124697.], # x_nmarsh_west
    [552323., 4124492.], # x_lagoon_north
    [552226., 4124428.], # x_lagoon_south
    [552547., 4123962.], # butano off-channel storage
    [552581., 4123866.], # another butano off-channel storage
    [552596., 4123395.], # various in Butano Marsh
    [552576., 4123458.], # Funny intersection place
    [553116., 4123723.], # extra point in butano marsh w/ tangent
    [552746., 4123550.], # butano between old and new.
    [552331., 4124869.], # small area on west side of N Pond channel
]

failures=[]

## 
g_new=g
for seed in seed_points:
    c=g_new.select_cells_nearest(seed,inside=True)
    if c is not None:
        print("Point %s already filled"%seed)
        continue
    print("Will fill this one")

    result=triangulate_hole.triangulate_hole(g_new,seed_point=seed,**th_kwargs)
    if isinstance(result,unstructured_grid.UnstructuredGrid):
        g_new=result
    else:
        print("-----Fail on point: %s-----"%seed)
        failures.append( result )
        
## 
# 16 of 20 succeed, 4 failures.
# 0: bad telescoping leads to doubled edge.
# 1: around roundhill. It's just too thin here. Can I push
#    the quads to the boundary?
# 2: south side of the tangent join, quad resolution doesn't match
# 3: west n pond entry angle at n end too shallow.

plt.figure(1).clf()
failures[3].grid.plot_edges(lw=0.7)
g_new.plot_edges(lw=0.5,alpha=0.3,color='0.6')
plt.axis('tight')
plt.axis('equal')

##
g_new.write_pickle('v18-successes.pkl')

        
##         
plt.figure(1).clf()
g_new.plot_edges(color='tab:brown',lw=0.7)
plt.axis('tight')
plt.axis('equal')

g_new.renumber()
g_new.write_ugrid('quad_tri_v18frontcc.nc',overwrite=True)


