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

# plt.figure(2).clf()
# qg.gen.plot_edges()
# qg.gen.plot_nodes(labeler='id')

qg.add_internal_edge([23,36])
qg.add_internal_edge([20,32])

qg.execute()

# Using the tan_groups, set the values to be exact
for i_grp in qg.i_tan_groups:
    grp_psi=qg.psi[i_grp].mean()
    qg.psi[i_grp]=grp_psi
for j_grp in qg.j_tan_groups:
    grp_phi=qg.phi[j_grp].mean()
    qg.phi[j_grp]=grp_phi
    
# rebay was filling in the sting with an extra, invalid edge.
# this is maybe fixed by retaining the constraints, though
# that may bring new issues down the road.

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
ax.set_position([0,0,1,1])
ax.axis('off')

# qg.plot_psi_phi(ax=ax)
qg.g_int.plot_edges(lw=0.5,color='k',alpha=0.3)

# This code assumes that either ij are both fixed, or neither fixed.
fixed_int_to_gen={}
for n in qg.g_int.valid_node_iter():
    val=qg.g_int.nodes['ij'][n,:]
    if np.isnan(val[0] + val[1]): continue
    # does it appear in gen?
    x=qg.g_int.nodes['x'][n]
    gn=qg.gen.select_nodes_nearest(x)
    gx=qg.gen.nodes['x'][gn]
    delta=utils.dist( x-gx)
    if delta>0.01: continue
    if not np.any(qg.gen.nodes['ij_fixed'][gn]): continue
    fixed_int_to_gen[n]=gn

n_fixed=list(fixed_int_to_gen.keys())
    
qg.g_int.plot_nodes(mask=n_fixed, color='tab:red', ax=ax)

# Thoughts on how to create the quad grid --
#  Can't rely on anything in ij space.
# Visit nodes with rigid i/j in the intermediate grid.
#   - for any fixed coordinate, trace a contour.
#     currently there is not an indication of fixed.
#     When the fixed coordinate corresponds to a boundary
#     edge (can probably used g_int.nodes['ij'] to figure
#     that out), then pull the exact boundary edge.
#     Otherwise, trace the contour in 1 [or 2] directions,

# These edges then have to get intersected to find internal
# nodes.

# For contours that have global fixed nodes in multiple locations:
# two different directions?  For the moment disregard that
# issue.

## Join

g_int=qg.g_int

six.moves.reload_module(utils)
six.moves.reload_module(exact_delaunay)

g_final=exact_delaunay.Triangulation(extra_edge_fields=[('dij',np.float64,2),
                                                        ('ij',np.float64,2),
                                                        ('psiphi',np.float64,2)])
g_final.edge_defaults['dij']=np.nan
# Not great - when edges get split, this will at least leave the fields as nan
# instead of 0.
g_final.edge_defaults['ij']=np.nan
g_final.edge_defaults['psiphi']=np.nan

final_traces=[]

def trace_contour(b,dij):
    if dij[0]!=0:
        # trace constant phi
        cval=qg.phi[b] # The contour to trace
        node_field=qg.phi # the field to trace a contour of
        if dij[0]<0:
            cval_pos='left'
        else:
            cval_pos='right' # guess and check
    elif dij[1]!=0:
        cval=qg.psi[b]
        node_field=qg.psi
        if dij[1]<0:
            cval_pos='left' # guess and check
        else:
            cval_pos='right' # guess and check
    else:
        raise Exception("what?")
    trace_items=g_int.trace_node_contour(n0=b,cval=cval,
                                         node_field=node_field,
                                         pos_side=cval_pos,
                                         return_full=True)
    return trace_items

# g_final node index =>
# list of [
#   ( dij, from the perspective of leaving the node,
#     'internal' or 'boundary' )
node_exits=defaultdict(list)

def insert_contour(trace_items,dij=None,
                   psiphi0=[np.nan,np.nan],ij0=[np.nan,np.nan]):
    assert np.isfinite(ij0[0]) or np.isfinite(ij0[1])
    assert np.isfinite(psiphi0[0]) or np.isfinite(psiphi0[1])
    
    if dij is not None:
        dij=np.asarray(dij)
        
    trace_points=np.array( [pnt
                            for typ,idx,pnt in trace_items
                            if pnt is not None])

    # Check whether the ends need to be forced into the boundary
    # but here we preemptively doctor up the ends
    for i in [0,-1]:
        if trace_items[i][0]=='edge':
            # When it hits a non-cartesian edge this will fail (which is okay)
            # Feels a bit fragile:
            j_int=trace_items[i][1].j # it's a halfedge
            j_gen=g_int.edges['gen_j'][j_int] # from this original edge
            dij_gen=qg.gen.edges['dij'][j_gen]
            if (dij_gen!=0).sum()==2:
                print("Not worrying about contour hitting diagonal")
                continue
            
            # Force that point into an existing constrained edge of g_final
            pnt=trace_points[i]
            best=[None,np.inf]
            for j in np.nonzero(g_final.edges['constrained'] & (~g_final.edges['deleted']))[0]:
                d=utils.point_segment_distance( pnt,
                                                g_final.nodes['x'][g_final.edges['nodes'][j]] )
                if d<best[1]:
                    best=[j,d]
            j,d=best
            # Typ. 1e-10 when using UTM coordinates
            assert d<1e-5
            if d>0.0:
                n_new=g_final.split_constraint(x=pnt,j=j)

    trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert',
                                                               on_exists='stop')
    if dij is not None:
        g_final.edges['dij'][trace_edges]=dij
    if ij0 is not None:
        g_final.edges['ij'][trace_edges]=ij0
    if psiphi0 is not None:
        g_final.edges['psiphi'][trace_edges]=psiphi0
            
    trace_data=dict(fin_nodes=trace_nodes,
                    fin_edges=trace_edges,
                    items=trace_items,
                    dij=dij,
                    psiphi0=psiphi0)
    final_traces.append(trace_data)

    # Update node_exits:
    exit_dij=dij
    for a in trace_nodes[:-1]:
        node_exits[a].append( (exit_dij,'internal') )
    if dij is not None:
        exit_dij=-dij
    for b in trace_nodes[1:]:
        node_exits[b].append( (exit_dij,'internal') )
        

def trace_and_insert_contour(b,dij):
    # does dij_angle fall between the angles formed by the boundary, including
    # a little slop.
    print(f"{dij} looks good")
    gn=fixed_int_to_gen[b] # below we already check to see that b is in there.
    
    ij0=qg.gen.nodes['ij'][gn].copy()
    # only pass the one constant along the contour
    if dij[0]==0:
        ij0[1]=np.nan
        psiphi0=[qg.psi[b],np.nan]
    else:
        ij0[0]=np.nan
        psiphi0=[np.nan,qg.phi[b]]
    
    trace_items=trace_contour(b,dij)
    return insert_contour(trace_items,dij=dij,
                          psiphi0=psiphi0,
                          ij0=ij0)

def trace_boundary(b,dij):
    nodes=[b]
    while 1:
        nbrs=g_int.node_to_nodes(nodes[-1])
        last_ij=g_int.nodes['ij'][nodes[-1]]
        for n in nbrs:
            nbr_dij=utils.to_unit( g_int.nodes['ij'][n] - last_ij )
            if (dij*nbr_dij).sum() > 0.99:
                nodes.append(n)
                break
        else:
            # no good neighbor -- end of while loop
            break
    return nodes

def trace_and_insert_boundary(b,dij):
    if dij is not None:
        dij=np.asarray(dij)
        
    trace_int_nodes=trace_boundary(b,dij)
    trace_int_edges=[g_int.nodes_to_edge(a,b)
                     for a,b in zip(trace_int_nodes[:-1],trace_int_nodes[1:])]
    trace_int_cells=[]

    trace_points=g_int.nodes['x'][trace_int_nodes]
    trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert')
    g_final.edges['dij'][trace_edges]=dij
    # need to update ij,psiphi for these edges, too.
    gn=fixed_int_to_gen[b]
    if dij[0]==0:
        ij=[qg.gen.nodes['ij'][gn,0],np.nan]
        psiphi=[qg.psi[b],np.nan]
    elif dij[1]==0:
        ij=[np.nan, qg.gen.nodes['ij'][gn,1]]
        psiphi=[np.nan, qg.phi[b]]
    else:
        assert False
    g_final.edges['ij'][trace_edges]=ij
    g_final.edges['psiphi'][trace_edges]=psiphi
        
    trace_data=dict(int_nodes=trace_int_nodes,
                    int_cells=trace_int_cells,
                    int_edges=trace_int_edges,
                    fin_nodes=trace_nodes,
                    fin_edges=trace_edges,
                    dij=dij,
                    ij0=g_int.nodes['ij'][b])
    final_traces.append(trace_data)
    
    # Update node_exits:
    for a in trace_nodes[:-1]:
        node_exits[a].append( (dij,'boundary') )
    for b in trace_nodes[1:]:
        node_exits[b].append( (-dij,'boundary') )

# Add boundaries when they coincide with contours
cycle=g_int.boundary_cycle() # could be multiple eventually...


# Need to get all of the boundary contours in first, then
# return with internal.
for mode in ['boundary','internal']:
    for a,b,c in zip(cycle,
                     np.roll(cycle,-1),
                     np.roll(cycle,-2)):
        if b not in fixed_int_to_gen: continue
        
        # First, should the edge a--b be included as a boundary edge coincident
        # with a contour?
        ij_a=g_int.nodes['ij'][a]
        ij_b=g_int.nodes['ij'][b]
        ij_c=g_int.nodes['ij'][c]

        ij_angle_ab=np.arctan2( ij_a[1] - ij_b[1],
                                ij_a[0] - ij_b[0] )
        ij_angle_cb=np.arctan2( ij_c[1] - ij_b[1],
                                ij_c[0] - ij_b[0] )

        for dij in [ [-1,0], [1,0], [0,-1],[0,1]]:
            # is dij into the domain?
            dij_angle=np.arctan2( dij[1],dij[0] )
            trace=None
            eps=1e-5

            # If dij coincides with a boundary, trace it.
            # Only check ij_angle_cb, since each boundary edge
            # should have a fixed node at each end.
            if np.abs( (dij_angle-ij_angle_cb+np.pi)%(2*np.pi)-np.pi)<eps:
                if mode!='boundary': continue
                print("Trace boundary")
                trace_and_insert_boundary(b,dij)
            elif ( ( (dij_angle-(ij_angle_cb+eps)) % (2*np.pi) )
                   < ( (ij_angle_ab-eps-ij_angle_cb)%(2*np.pi))):
                if mode!='internal': continue
                b_final=g_final.select_nodes_nearest(g_int.nodes['x'][b],max_dist=0.0)
                dupe=False
                if b_final is not None:
                    for exit_dij,exit_type in node_exits[b_final]:
                        if np.all( exit_dij==dij ):
                            dupe=True
                            print("Duplicate exit for internal trace from %d. Skip"%b)
                            break
                if not dupe:
                    trace_and_insert_contour(b,dij)
# ---
# Start at b, trace in the dij direction.
# should have 14.  have 11 attempts.
g_final.plot_edges(color='tab:green',mask=g_final.edges['constrained'],
                   lw=3.0,zorder=2.,alpha=0.4)
# ---


def tri_to_grid(g_final):
    g_final2=g_final.copy()

    for c in g_final2.valid_cell_iter():
        g_final2.delete_cell(c)

    for j in np.nonzero( (~g_final2.edges['deleted']) & (~g_final2.edges['constrained']))[0]:
        g_final2.delete_edge(j)

    g_final2.modify_max_sides(2000)
    g_final2.make_cells_from_edges()
    return g_final2

g_final2=tri_to_grid(g_final)

import stompy.plot.cmap as scmap
cmap=scmap.load_gradient('oc-sst.cpt')

g_final2.plot_cells(values=np.linspace(0,1,g_final2.Ncells()),cmap=cmap)

ax.axis('tight')
ax.axis('equal')

# plt.figure(2).clf()
# fig,ax=plt.subplots(num=2)
# g_final.plot_edges(mask=g_final.edges['constrained'],ax=ax,color='k',alpha=0.5,lw=2.0)
# ax.axis('equal')

# ---

e2c=g_final2.edge_to_cells(recalc=True)

i_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)
j_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)

for j in g_final2.valid_edge_iter():
    c1,c2=e2c[j,:]
    if c1<0 or c2<0: continue
    
    if g_final2.edges['dij'][j,0]==0:
        i_adj[c1,c2]=i_adj[c2,c1]=True
    elif g_final2.edges['dij'][j,1]==0:
        j_adj[c1,c2]=j_adj[c2,c1]=True
    else:
        print('What?')

from scipy.sparse import csgraph
n_comp_i,labels_i=csgraph.connected_components(i_adj.astype(np.int32),directed=False)
n_comp_j,labels_j=csgraph.connected_components(j_adj,directed=False)

def add_swath_contours_old(comp_cells,node_field,coord,field_extrap):
    # Check all of the nodes to find the range ij
    comp_nodes=[ g_final2.cell_to_nodes(c) for c in comp_cells ]
    comp_nodes=np.unique( np.concatenate(comp_nodes) )
    comp_ijs=[] # Certainly could have kept this info along while building...

    field_values=[]

    comp_ij=np.array(g_final2.edges['ij'][ g_final2.cell_to_edges(comp_cells[0]) ])
    comp_pp=np.array(g_final2.edges['psiphi'][ g_final2.cell_to_edges(comp_cells[0]) ])
    
    # it's actually the other coordinate that we want to consider.
    field_min=np.nanmin( comp_pp[:,1-coord] )
    field_max=np.nanmax( comp_pp[:,1-coord] )
    
    coord_min=np.nanmin( comp_ij[:,1-coord] )
    coord_max=np.nanmax( comp_ij[:,1-coord] )

    n_swath_cells=int(np.round(coord_max-coord_min))

    new_field_contours=np.linspace(field_min,field_max,1+n_swath_cells)[1:-1]

    c=comp_cells[0] # arbitrary -- just need somebody in the swath
    nodes=g_final2.cell_to_nodes(c)
    c_x=g_final2.nodes['x'][nodes]
    c_x_vals=field_extrap(c_x)

    # Get from a point to proper grid element:
    def trace_contour_from_point(pnt,cval,node_field):
        trace_left=g_int.trace_node_contour(loc0=['point',None,pnt],
                                            cval=cval,
                                            node_field=node_field,
                                            pos_side='left',
                                            return_full=True)

        trace_right=g_int.trace_node_contour(loc0=['point',None,pnt],
                                             cval=cval,
                                             node_field=node_field,
                                             pos_side='right',
                                             return_full=True)

        trace_items=trace_right[1:][::-1] + trace_left
        return trace_items

    for cnum,cval in enumerate(new_field_contours):
        # Find starting point.
        i=np.arange(len(c_x))

        for ai,bi in zip(i, np.roll(i,-1)):
            if cval==c_x_vals[ai]:
                pnt=c_x[ai]
                break
            elif c_x_vals[ai]==c_x_vals[bi]:
                continue # avoid division by zero
            else:
                alpha=(cval - c_x_vals[ai]) / (c_x_vals[bi] - c_x_vals[ai])
                if alpha>=0 and alpha <= 1:
                    pnt=(1-alpha)*c_x[ai] + alpha*c_x[bi]
                    break
        else:
            raise Exception("Failed to find edges bracketing contour value")

        trace_items=trace_contour_from_point(pnt,cval,node_field)
        psiphi0=[np.nan,np.nan]
        psiphi0[1-coord]=cval
        ij0=[np.nan,np.nan]
        ij0[1-coord]=coord_min+1+cnum
        insert_contour(trace_items,dij=None,psiphi0=psiphi0,
                       ij0=ij0) # could figure out dij if needed


# preprocessing for contour placement
nd=quads.NodeDiscretization(g_int)
Mdx,Bdx=nd.construct_matrix(op='dx')
Mdy,Bdy=nd.construct_matrix(op='dy')
psi_dx=Mdx.dot(qg.psi)
psi_dy=Mdy.dot(qg.psi)
phi_dx=Mdx.dot(qg.phi)
phi_dy=Mdy.dot(qg.phi)

# These should be about the same.  And they are, but
# keep them separate in case the psi_phi solution procedure
# evolves.
psi_grad=np.sqrt( psi_dx**2 + psi_dy**2)
phi_grad=np.sqrt( phi_dx**2 + phi_dy**2)

pp_grad=[psi_grad,phi_grad]

## 
# Just figures out the contour values and sets them on the patches.
patch_to_contour=[{},{}] # coord, cell index=>array of contour values

def add_swath_contours_new(comp_cells,node_field,coord,field_extrap):
    # Check all of the nodes to find the range ij
    comp_nodes=[ g_final2.cell_to_nodes(c) for c in comp_cells ]
    comp_nodes=np.unique( np.concatenate(comp_nodes) )
    comp_ijs=[] # Certainly could have kept this info along while building...

    field_values=[]

    comp_ij=np.array(g_final2.edges['ij'][ g_final2.cell_to_edges(comp_cells[0]) ])
    comp_pp=np.array(g_final2.edges['psiphi'][ g_final2.cell_to_edges(comp_cells[0]) ])
    
    # it's actually the other coordinate that we want to consider.
    field_min=np.nanmin( comp_pp[:,1-coord] )
    field_max=np.nanmax( comp_pp[:,1-coord] )
    
    coord_min=np.nanmin( comp_ij[:,1-coord] )
    coord_max=np.nanmax( comp_ij[:,1-coord] )

    n_swath_cells=int(np.round(coord_max-coord_min))

    # Could do this more directly from topology if it mattered..
    swath_poly=ops.cascaded_union( [g_final2.cell_polygon(c) for c in comp_cells] )
    swath_nodes=g_int.select_nodes_intersecting(swath_poly)
    swath_vals=node_field[swath_nodes]
    swath_grad=pp_grad[1-coord][swath_nodes] # right?
    order=np.argsort(swath_vals)
    o_vals=swath_vals[order]
    o_dval_ds=swath_grad[order]
    o_ds_dval=1./o_dval_ds

    # trapezoid rule integration
    d_vals=np.diff(o_vals)
    # Particularly near the ends there are a lot of
    # duplicate swath_vals.
    # Try a bit of lowpass to even things out.
    if 1:
        winsize=int(len(o_vals)/5)
        if winsize>1:
            o_ds_dval=filters.lowpass_fir(o_ds_dval,winsize)
            
    s=np.cumsum(d_vals*0.5*(o_ds_dval[:-1]+o_ds_dval[1:]))
    s=np.r_[0,s]
    s_contours=np.linspace(s[0],s[-1],1+n_swath_cells)
    adj_contours=np.interp( s_contours,
                            s,o_vals)
    adj_contours[0]=field_min
    adj_contours[-1]=field_max

    for c in comp_cells:
        patch_to_contour[coord][c]=adj_contours

        
if 1: # Swath processing        
    for coord in [0,1]: # i/j
        print("Coord: ",coord)
        if coord==0:
            labels=labels_i
            n_comp=n_comp_i
            node_field=qg.phi # feels backwards.. it's right, just misnamed 
        else:
            labels=labels_j
            n_comp=n_comp_j
            node_field=qg.psi

        # Would be nice to just pass in the g_int triangulation.
        field_extrap=utils.LinearNDExtrapolator(g_int.nodes['x'],node_field)

        for comp in range(n_comp):
            print("Swath: ",comp)
            comp_cells=np.nonzero(labels==comp)[0]
            add_swath_contours_new(comp_cells,node_field,coord,field_extrap)

## 

# Plot the contours to check visually before working up patch-construction
# code.

plt.figure(3).clf()
fig,ax=plt.subplots(num=3)

g_final2.plot_edges(color='k',lw=0.6,ax=ax)

for c in g_final2.valid_cell_iter():
    c_poly=g_final2.cell_polygon(c)
    c_int_cells=g_int.select_cells_intersecting(c_poly)

    for coord in [0,1]:
        if coord==0:
            node_field=qg.phi
        else:
            node_field=qg.psi

        cvals=patch_to_contour[coord][c][1:-1]
        cset=g_int.contour_node_values(node_field,cvals,ax=ax,
                                       colors='k',linewidths=0.6,
                                       linestyles='-',
                                       tri_kwargs=dict(cell_mask=c_int_cells))
ax.axis('equal')

##

# Direct grid gen from contour specifications:

# rough implementation:
extraps=[ utils.LinearNDExtrapolator( g_int.nodes['x'], qg.psi),
          utils.LinearNDExtrapolator( g_int.nodes['x'], qg.phi) ]
def pp_to_xy(pp,x0):
    def cost(x,extraps):
        return ((extraps[0](x)-pp[0])**2 + (extraps[1](x)-pp[1])**2).sum()
    best=fmin(cost,x0,args=(extraps,),disp=False)
    return best

@utils.add_to(g_int)
def fields_to_xy(self,target,node_fields,x0):
    """
    target: values of node_fields to locate
    x0: starting point

    NB: edges['cells'] must be up to date before calling
    """
    c=self.select_cells_nearest(x0)

    while 1:
        c_nodes=self.cell_to_nodes(c)
        M=np.array( [ node_fields[0][c_nodes],
                        node_fields[1][c_nodes],
                        [1,1,1] ] )
        b=[target[0],target[1],1.0]

        weights=np.linalg.solve(M,b)
        if min(weights)<0: # not there yet.
            min_w=np.argmin(weights)
            c_edges=self.cell_to_edges(c,ordered=True)# nodes 0--1 is edge 0, ...
            sel_j=c_edges[ (min_w+1)%(len(c_edges)) ]
            edges=self.edges['cells'][sel_j]
            if edges[0]==c:
                next_c=edges[1]
            elif edges[1]==c:
                next_c=edges[0]
            else:
                raise Exception("Fail.")
            if next_c<0:
                if weights.min()<-1e-5:
                    print("Left triangulation (min weight: %f)"%weights.min())
                    import pdb
                    pdb.set_trace()
                # Clip the answer to be within this cell (will be on an edge
                # or node).
                weights=weights.clip(0)
                weights=weights/weights.sum()
                break
            c=next_c
            continue
        else:
            break
    x=(self.nodes['x'][c_nodes]*weights[:,None]).sum(axis=0)
    return x

patch_grids=[]

g_int.edge_to_cells()

for c in utils.progress(g_final2.valid_cell_iter()):
    psi_cvals=patch_to_contour[1][c]
    phi_cvals=patch_to_contour[0][c]
    
    g_patch=unstructured_grid.UnstructuredGrid(max_sides=4)
    g_patch.add_rectilinear( [0,0], [len(psi_cvals)-1,len(phi_cvals)-1],
                             len(psi_cvals),len(phi_cvals))
    g_patch.add_node_field( 'ij', g_patch.nodes['x'].astype(np.int32))
    pp=np.c_[ psi_cvals[g_patch.nodes['ij'][:,0]],
              phi_cvals[g_patch.nodes['ij'][:,1]] ]
    g_patch.add_node_field( 'pp', pp)

    x0=g_final2.cells_centroid([c])[0]
    for n in g_patch.valid_node_iter():
        x=g_int.fields_to_xy(g_patch.nodes['pp'][n],
                             node_fields=[qg.psi,qg.phi],
                             x0=x0)
        g_patch.nodes['x'][n]=x
        # Hmm -
        # When it works, this probably reduces the search time considerably,
        # but there is the possibility, particularly at corners, that
        # this x will be a corner, that corner will lead to the cell *around*
        # the corner, and then we get stuck.
        # Even the centroid isn't great since it might not even fall inside
        # the cell.
        # x0=x 
    patch_grids.append(g_patch)

##

# c=14, some of it looks great, some not so much.

g=patch_grids[0]
for g_next in patch_grids[1:]:
    g.add_grid(g_next,merge_nodes='auto')

##

plt.figure(4).clf()
fig,ax=plt.subplots(num=4)
g.plot_edges(color='k',lw=0.5)
g.plot_cells(color='0.8',lw=0,zorder=-2)
ax.axis('off')
ax.set_position([0,0,1,1])

## 
# This is the result for one of the bad nodes:
bad_xy=[552518.562916275, 4124197.347880902]
bad_n=g_patch.select_nodes_nearest(bad_xy)

g_patch.plot_nodes(mask=[bad_n],color='k')
bad_ij=g_patch.nodes['ij'][bad_n] # 1,1

tgt_pp=psi_cvals[bad_ij[0]],phi_cvals[bad_ij[1]]
x=g_patch.nodes['x'][bad_n]
ans_pp=extraps[0](x),extraps[1](x) # actually very close.

# Where I wanted it:
exp_x=[552479.679184539, 4124259.167987721]
exp_pp=extraps[0](exp_x),extraps[1](exp_x)

##

if 0: # Older postprocessing
    # -- 

    g_final3=tri_to_grid(g_final)
    g_final3.edge_to_cells(recalc=True)
    g_final3.delete_orphan_edges()

    # -- 

    e2c=g_final3.edge_to_cells()

    # Look for nodes 
    for n in g_final3.valid_node_iter():
        js=g_final3.node_to_edges(n)
        if len(js)>2: continue
        if len(js)==0: continue # orphan

        ij1=g_final3.edges['ij'][js[0]]
        ij2=g_final3.edges['ij'][js[1]]
        if np.allclose( ij1,ij2, equal_nan=True):
            g_final3.merge_edges(node=n)

    g_final3.delete_orphan_nodes()

    g_final3.renumber()

    n_fixed=np.isfinite(g_final3.edges['psiphi']).sum(axis=1)
    assert np.all( n_fixed==1 )

    # --

    plt.figure(1).clf()
    fig,ax=plt.subplots(num=1)

    g_final3.plot_edges(lw=0.5,ax=ax,cmap='rainbow')

    ax.set_position([0,0,1,1])
    ax.axis('off')
    ax.axis('tight')
    ax.axis('equal')

    # --

    # This isn't the most direct way to handle the adjustments, but
    # will work for the moment while I make sure that a reasonable
    # output is possible.

    # For each node, I want a psiphi, and whether either are fixed
    # Map psiphi0 values to nodes

    node_pp=np.zeros( (g_final3.Nnodes(),2), np.float64)
    node_pp[:,:]=np.nan
    node_ij=np.zeros( (g_final3.Nnodes(),2), np.float64)
    node_ij[:,:]=np.nan

    # Enumerate connected contours at the same time
    adj=[ sparse.dok_matrix( (g_final3.Nnodes(),g_final3.Nnodes()), np.int32),
          sparse.dok_matrix( (g_final3.Nnodes(),g_final3.Nnodes()), np.int32) ]

    for coord in [0,1]:
        for j in g_final3.valid_edge_iter():
            cval=g_final3.edges['psiphi'][j,coord]
            ijval=g_final3.edges['ij'][j,coord]
            if np.isnan(cval):
                continue
            assert np.isfinite(ijval)
            nodes=g_final3.edges['nodes'][j]
            node_pp[nodes,coord]=cval
            node_ij[nodes,coord]=ijval
            adj[coord][nodes[0],nodes[1]]=adj[coord][nodes[1],nodes[0]]=1

    g_final3.add_node_field('psiphi',node_pp, on_exists='overwrite')
    g_final3.add_node_field('ij',node_ij, on_exists='overwrite')

    # --

    n_node_i_comp,node_labels_i=sparse.csgraph.connected_components(adj[0], directed=False)
    n_node_j_comp,node_labels_j=sparse.csgraph.connected_components(adj[1], directed=False)

    g_final3.add_node_field('comp',np.c_[node_labels_i,node_labels_j], on_exists='overwrite')

    # --

    # Note which contours are fixed.

    # Iterate through fixed nodes of qg.gen, and
    # find matches.

    ij_fixed=np.zeros( (g_final3.Nnodes(),2), np.bool8 )
    for coord in [0,1]:
        for gn in np.nonzero( qg.gen.nodes['ij_fixed'][:,coord] )[0]:
            gx=qg.gen.nodes['x'][gn]
            n=g_final3.select_nodes_nearest(gx,max_dist=0.0)
            assert n is not None

            n_comp=g_final3.nodes['comp'][n,coord]
            match=g_final3.nodes['comp'][:,coord]==n_comp
            ij_fixed[match,coord]=True

    g_final3.add_node_field('ij_fixed',ij_fixed,on_exists='overwrite')        

    #--
    plt.figure(1).clf()
    fig,ax=plt.subplots(num=1)

    g_final3.plot_edges(lw=0.5,ax=ax)
    g_final3.plot_nodes(values=g_final3.nodes['comp'][:,0] % 7,
                        mask=g_final3.nodes['ij_fixed'][:,0],
                        ax=ax,cmap='rainbow')

    ax.set_position([0,0,1,1])
    ax.axis('off')
    ax.axis('tight')
    ax.axis('equal')

##

# How best to deal with the spacing?
# Is there a nice way to express the global problem?
#   Say, build a linear system that solves for the contour
#   values
#  1. For each pair of adjacent contours, calculate the average
#     geographic distance between, and their current contour difference
#     to get a local rate of change.
#  2. Design a cost function that minimizes change in spacing.
#     So each term is something like (d(a,b) - d(b,c))^2
#     where d(a,b) is the average geographic distance between contour a
#     and contour b.
#     Some contours of course are fixed.
#     May have to include monotonicity constraints, or use the
#     inverse of distance to apply more weight on small distances.

#  Does it make sense to go back and solve this more directly at the
#  swath level?


# At the swath level, I could either have the current, fixed swath widths
# or just a nominal spacing and calculate the count along the way.
# The objective is to closely match spacing at boundary with adjacent
# swaths, and to have either the specified number of cells or approx.
# match the target spacing elsewhere.

# As a standard optimization problem:
# The swaths are connected in an undirected graph.
#  Each swath must ultimately generate a sequence of contour values.
#  Each swath is made of patches (cells of g_final2)
#  Each patch can calculate the spacing from the contour values.
#  Cost is evaluated
#   (a) within swaths, based on how close the target
#       resolution or count is achieved
#       and [maybe] how spacing changes within the swath/patches
#   (b) between patches in adjacent swaths, based on how spacing
#       matches up.
#  Cost is optimized over the parameters for spacing within
#  each swath.

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
g_final2.plot_cells(ax=ax,values=np.arange(g_final2.Ncells()),
                    cmap='tab20')
ax.axis('equal')

##

# Methods to estimate spacing from contour values at the patch
# or swath

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
g_final2.plot_edges(ax=ax,color='k',lw=0.5)
c=0
g_final2.plot_cells(mask=[c], ax=ax,color='orange')

ax.axis('equal')

# Nodes of g_int inside c:
c_poly=g_final2.cell_polygon(c)
patch_nodes=g_int.select_nodes_intersecting(c_poly)
#g_int.plot_nodes(mask=patch_nodes)


##
coord=0
# Previous way is linear in psi/phi:
comp_cells=[0]
comp_nodes=[ g_final2.cell_to_nodes(cc) for cc in comp_cells ]
comp_nodes=np.unique( np.concatenate(comp_nodes) )
comp_ijs=[] # Certainly could have kept this info along while building...

field_values=[]
comp_ij=np.array(g_final2.edges['ij'][ g_final2.cell_to_edges(comp_cells[0]) ])
comp_pp=np.array(g_final2.edges['psiphi'][ g_final2.cell_to_edges(comp_cells[0]) ])
    
# it's actually the other coordinate that we want to consider.
field_min=np.nanmin( comp_pp[:,1-coord] )
field_max=np.nanmax( comp_pp[:,1-coord] )
coord_min=np.nanmin( comp_ij[:,1-coord] )
coord_max=np.nanmax( comp_ij[:,1-coord] )
# n_swath_cells=int(np.round(coord_max-coord_min))
n_swath_cells=20
new_field_contours=np.linspace(field_min,field_max,1+n_swath_cells)
for v in new_field_contours:
    ax.axvline(v,color='k',lw=0.5)

plt.figure(3).clf()
fig,ax=plt.subplots(num=3)
g_int.plot_edges(ax=ax,color='0.6',lw=0.5)
g_int.contour_node_values(qg.phi,new_field_contours,colors='g')

##

patch_val=qg.phi[patch_nodes]
patch_grad=phi_grad[patch_nodes]

plt.figure(7).clf()
fig,(ax_grad,ax_s)=plt.subplots(2,1,num=7)
ax_grad.plot(patch_val,patch_grad,'g.')

ax_grad.set_xlabel('val')
ax_grad.set_ylabel(r'|$\nabla$|')

# Linear in phi space:
new_field_contours=np.linspace(field_min,field_max,1+n_swath_cells)

# I want contour values such that the x spacing is roughly
# even.
# I have dphi/dx ~ fn(phi)
# What is the estimate spacing between contours of phi_a,phi_b?
# delta_phi = (phi_b - phi_a)
# delta_x = delta_phi / (dphi/dx)

# What if integrate?
# x(phi) = int_0^phi 1/(dphi/dx) d phi'

order=np.argsort(patch_val)

o_vals=patch_val[order]
o_dphi_ds=patch_grad[order]
o_ds_dphi=1./o_dphi_ds

d_vals=np.diff(o_vals)
s=np.cumsum(d_vals*0.5*(o_ds_dphi[:-1]+o_ds_dphi[1:]))
s=np.r_[0,s]
ax_s.plot(s, o_vals)

s_contours=np.linspace(s[0],s[-1],1+n_swath_cells)
adj_contours=np.interp( s_contours,
                        s,o_vals)

# Significantly better than linear in phi space.
g_int.contour_node_values(qg.phi,adj_contours,colors='orange',ax=ax)
