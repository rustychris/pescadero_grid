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

# Using the tan_groups, set the values to be exact
for i_grp in qg.i_tan_groups:
    grp_psi=qg.psi[i_grp].mean()
    qg.psi[i_grp]=grp_psi
for j_grp in qg.j_tan_groups:
    grp_phi=qg.phi[j_grp].mean()
    qg.phi[j_grp]=grp_phi
    
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

if 1: 
    # Try adding in any diagonal edges here, so that ragged edges
    # get cells in g_final2, too.
    ragged=np.nonzero( (qg.gen.edges['dij']!=0.0).sum(axis=1)==2 )[0]
    for gen_j in ragged:
        j_ints=np.nonzero( g_int.edges['gen_j']==gen_j )[0]
        for j_int in j_ints:
            nodes=[g_final2.add_or_find_node(g_int.nodes['x'][n])
                   for n in g_int.edges['nodes'][j_int]]
            j_fin2=g_final2.nodes_to_edge(nodes)
            if j_fin2 is None:
                j_fin2=g_final2.add_edge(nodes=nodes,constrained=True,
                                         # May need other sentinel values here to simplify
                                         # code below
                                         dij=[1,1],
                                         ij=[np.nan,np.nan],
                                         psiphi=[np.nan,np.nan])

    g_final2.make_cells_from_edges()


## 

plt.figure(5).clf()
fig,ax=plt.subplots(num=5)
g_final2.plot_cells(values=np.linspace(0,1,g_final2.Ncells()),cmap=cmap,ax=ax)

ax.axis('tight')
ax.axis('equal')

##

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
        print("What? Ragged edge okay, but it shouldn't have both cell neighbors")

n_comp_i,labels_i=sparse.csgraph.connected_components(i_adj.astype(np.int32),directed=False)
n_comp_j,labels_j=sparse.csgraph.connected_components(j_adj,directed=False)

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

# Direct grid gen from contour specifications:

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
                    # Either the starting cell didn't allow a simple path
                    # to the target, or the target doesn't fall inside the
                    # grid (e.g. ragged edge)
                    return [np.nan,np.nan]
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
        if np.isnan(x[0]):
            # If it's a ragged cell, probably okay.
            edge_dijs=g_final2.edges['dij'][ g_final2.cell_to_edges(c) ]
            ragged_js=(edge_dijs!=0.0).sum(axis=1)
            if np.any(ragged_js):
                print("fields_to_xy() failed, but cell is ragged.")
                g_patch.delete_node_cascade(n)
            else:
                print("ERROR: fields_to_xy() failed. Cell not ragged.")
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

g=patch_grids[0]
for g_next in patch_grids[1:]:
    g.add_grid(g_next,merge_nodes='auto',tol=1e-6)

##

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
