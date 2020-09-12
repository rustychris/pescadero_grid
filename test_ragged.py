# Southeast marsh channel
from matplotlib import collections
import stompy.grid.quad_laplacian as quads
from stompy.grid import exact_delaunay
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import six
from stompy import utils
import numpy as np

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

##

# Pretty close -
# HERE: Remaining:
#   - making the spacing more even (cubic interp?)
#   - regions that overlap in ij space.

##
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

plt.figure(2).clf()
qg.gen.plot_edges()
qg.gen.plot_nodes(labeler='id')

qg.add_internal_edge([23,36])
qg.add_internal_edge([20,32])

qg.execute()

# rebay was filling in the sting with an extra, invalid edge.
# this is maybe fixed by retaining the constraints, though
# that may bring new issues down the road.

# HERE
#  - current code can't cope with the duplicate ij values on either side
#    of the sting.
#  - the ij mapping also fails.

# The psi/phi field is perfect though!

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

# will there be a problem with contours that are common from
# two different directions?  For the moment disregard that
# issue.

g_int=qg.g_int
# g_final=unstructured_grid.UnstructuredGrid(max_sides=4,
#                                            extra_node_fields=[('gn',np.int32)])
# g_final.node_defaults['gn']=-1


six.moves.reload_module(utils)
six.moves.reload_module(exact_delaunay)

g_final=exact_delaunay.Triangulation()

final_traces=[]



# Add boundaries when they coincide with contours
cycle=g_int.boundary_cycle() # could be multiple eventually...

# Need to get all of the boundary contours in first, then
# return with internal.
for mode in ['boundary','internal']:
    for a,b,c in zip(cycle,
                     np.roll(cycle,-1),
                     np.roll(cycle,-2)):
        # First, should the edge a--b be included as a boundary edge coincident
        # with a contour?
        ij_a=g_int.nodes['ij'][a]
        ij_b=g_int.nodes['ij'][b]
        ij_c=g_int.nodes['ij'][c]

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
            # MAYBE - implement return_full, and then decode the output here.
            trace_items=g_int.trace_node_contour(b,cval=cval,
                                                 node_field=node_field,
                                                 pos_side=cval_pos,
                                                 return_full=True)
            return trace_items

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

        # Second, is there a fixed contour starting at 'b' going into the domain?
        # maybe return to this (which was the point of including a 'c' point.)
        # but I'm thinking a sweep line construction might be better.
        if b in fixed_int_to_gen:
            # gn=fixed_int_to_gen[b]
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
                    trace_int_nodes=trace_boundary(b,dij)
                    trace_int_edges=[g_int.nodes_to_edge(a,b)
                                     for a,b in zip(trace_int_nodes[:-1],trace_int_nodes[1:])]
                    trace_int_cells=[]

                    trace_points=g_int.nodes['x'][trace_int_nodes]
                    trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert')
                    trace_data=dict(int_nodes=trace_int_nodes,
                                    int_cells=trace_int_cells,
                                    int_edges=trace_int_edges,
                                    fin_nodes=trace_nodes,
                                    fin_edges=trace_edges,
                                    dij=dij,
                                    ij0=g_int.nodes['ij'][b])
                    final_traces.append(trace_data)
                elif ( ( (dij_angle-(ij_angle_cb+eps)) % (2*np.pi) )
                       < ( (ij_angle_ab-eps-ij_angle_cb)%(2*np.pi))):
                    if mode!='internal': continue
                    
                    # does dij_angle fall between the angles formed by the boundary, including
                    # a little slop.
                    print(f"{dij} looks good")
                    trace_items=trace_contour(b,dij)
                    trace_points=np.array( [pnt
                                            for typ,idx,pnt in trace_items
                                            if pnt is not None])
                    for i in [0,-1]:
                        if trace_items[i][0]=='edge':
                            # Force that point into an existing constrained edge of g_final
                            pnt=trace_points[i]
                            best=[None,np.inf]
                            for j in np.nonzero(g_final.edges['constrained'] & (~g_final.edges['deleted']))[0]:
                                d=utils.point_segment_distance( pnt,
                                                                g_final.nodes['x'][g_final.edges['nodes'][j]] )
                                if d<best[1]:
                                    best=[j,d]
                            j,d=best
                            assert d<1e-5
                            j_nodes=g_final.edges['nodes'][j].copy()
                            g_final.remove_constraint(j=j)
                            n_new=g_final.add_node(x=pnt)
                            g_final.add_constraint(j_nodes[0],n_new)
                            g_final.add_constraint(n_new,j_nodes[1])

                    trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert')
                    trace_data=dict(fin_nodes=trace_nodes,
                                    fin_edges=trace_edges,
                                    items=trace_items,
                                    dij=dij,
                                    psiphi0=[qg.psi[b],qg.phi[b]])
                    final_traces.append(trace_data)

# Start at b, trace in the dij direction.
# should have 14.  have 11 attempts.
g_final.plot_edges(color='tab:green',mask=g_final.edges['constrained'],
                   lw=1.0,zorder=2.)

# And have some issues with contours not quite matching boundaries.

# # Finding intersections:
# # Why not just a general grid function that finds all intersections and makes them
# # into nodes.
# # Will try this (it's close), but there will probably be too many roundoff issues
# # with internal contours intersecting the boundary. Next more complicated is to
# # explicitly check the endpoints of the internal contours since they will all
# # lead to an edge or node intersection with an external boundary.
# 
# def join_to_nearest_constraint(n,g):
#     # originally tried querying just the Delaunay triangles
#     # of n to find the edge.  That's not robust, as a nearby
#     # parallel contour might grab the edge instead.
#     js=g.select_edges_nearest(g.nodes['x'][n],count=10)
#     best_j=None
#     best_dist=np.inf
# 
#     for j in js:
#         if not g.edges['constrained'][j]: continue
#         nbr_a,nbr_b=g.edges['nodes'][j]
#         if n in [nbr_a,nbr_b]: continue
#         
#         d=utils.point_segment_distance( g.nodes['x'][n],
#                                         g.nodes['x'][[nbr_a,nbr_b]] )
#         if d<best_dist:
#             best_j=(j,nbr_a,nbr_b)
#             best_dist=d
#             print(f"candidate j={best_j} d={best_dist}")
#             
#     assert best_dist<1e-5
#     j,nbr_a,nbr_b=best_j
#     g.remove_constraint(j=j)
#     g.add_constraint(nbr_a,n)
#     g.add_constraint(n,nbr_b)
# 
# # there is probably a cleaner way to do this.
# for trace in final_traces:
#     # Only consider internal traces
#     if 'psiphi0' not in trace: continue
# 
#     for n in [ trace['fin_nodes'][0],
#                trace['fin_nodes'][-1]]:
#         js=g_final.node_to_edges(n)
#         con=g_final.edges['constrained'][js]
#         js_con=np.array(js)[con]
#         n_deg=len(js_con)
#         if n_deg>1:
#             continue
#         assert n_deg==1
# 
#         j=js_con[0]
#         if g_final.edges_length(j)<1e-5:
#             g_final.delete_node(n)
#             # Note that trace['fin_nodes'] is now out of date
#         else:
#             # Possible to have a little pigtail, when the constructed point goes
#             # slightly beyond the domain. In that case, 
#             join_to_nearest_constraint(n,g_final)

# Can use the DT to find the edge to intersect
g_final2=g_final.copy()

for c in g_final2.valid_cell_iter():
    g_final2.delete_cell(c)

for j in np.nonzero( (~g_final2.edges['deleted']) & (~g_final2.edges['constrained']))[0]:
    g_final2.delete_edge(j)

g_final2.modify_max_sides(2000)
g_final2.make_cells_from_edges()

import stompy.plot.cmap as scmap
cmap=scmap.load_gradient('oc-sst.cpt')

g_final2.plot_cells(values=np.linspace(0,1,g_final2.Ncells()),cmap=cmap)

ax.axis('tight')
ax.axis('equal')


##

# HERE:
#   Have to fudge the traces slightly now.
#   Option A: Scan the i_tan_groups and j_tan_groups.
#    These give indices into g_int.
#    For each one, sort by the changing psi or phi,
#    Scan the nodes, and note gaps where boundary
#    edges do not connect successive nodes
#    Then when tracing contours, check on that, blha blha blah

#   Option B: Refactor out the trace/insert contour code.
#     Include an epsilon parameter, and if a contour includes
#     a point within that parameter of an existing node on
#     a *parallel* line, then stop the trace, join to that existing
#     node, and call it done.

#  What if I just clamp them?
#  




##

# Enumerate swaths, add new contours
# Every cell is part of two swaths

# Option A:
#  iterate through cells of g_final2,
#    check for i-swath, j-swath
#      traverse swath
#  iterate through swaths:
#    calculate number of cells across, incl. 0.
#    if >1, calculate add'l contours
#    if 1, do no thing
#    if 0, HERE
#      (a) go back and re-solve psi/phi with these things lined up.
#          search along the thin sliver for gen nodes on both sides,
#          check those nodes for ij that indicate they can be
#          joined, and then add constraints to the solver.
#    For starters, manually add these constraints from the start.

## 

# __Old Notes__

# What about more of a paving approach?
# Choose a fixed vertex
# Scan in the 4 directions along contours.
#  Each direction could be
#   (a) out of the domain - ignore
#   (b) along a boundary - scan to next fixed vertex, note
#       which side (cw, ccw or both) is into the domain. 
#   (c) into the domain - scan along contour to opposite boundary.
#   (d) along existing edge - scan the segment. note which
#       side (cw,ccw, or both) is not meshed.

#  Choose a quadrant for the new quad.
#  Calculate length based on scans, matching the shortest constraint.
#  Add the cell...

#  The problem here [maybe] is that spacing will depend on the contours
#  at the node, which may be squished or stretched elsewhere in the
#  domain.

# for n in n_fixed:
#     gn=fixed_int_to_gen[n]
#     break






