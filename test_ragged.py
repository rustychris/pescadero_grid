# Southeast marsh channel
from matplotlib import collections
import stompy.grid.quad_laplacian as quads
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import six
from stompy import utils

##

# v00 has a ragged edge.
# v01 makes that a right. angle
from stompy.grid import triangulate_hole, rebay
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

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

# Testing a grid with a 360-degree vertex, and much more complicated
# patter
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v02.pkl')

qg=quads.QuadGen(gen_src,cell=0,final='anisotropic',execute=False,nom_res=5)

qg.execute()

##

# rebay was filling in the sting with an extra, invalid edge.
# this is maybe fixed by retaining the constraints, though
# that may bring new issues down the road.

# HERE
#  - current code can't cope with the duplicate ij values on either side
#    of the sting.
#  - the ij mapping also fails.

# The psi/phi field is perfect though!
##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
ax.set_position([0,0,1,1])
ax.axis('off')

qg.plot_psi_phi(ax=ax)

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
g_final=unstructured_grid.UnstructuredGrid(max_sides=4,
                                           extra_node_fields=[('gn',np.int32)])
g_final.node_defaults['gn']=-1

# Add boundaries when they coincide with contours
cycle=g_int.boundary_cycle() # could be multiple eventually...

for a,b,c in zip(cycle,
                 np.roll(cycle,-1),
                 np.roll(cycle,-2)):
    # First, should the edge a--b be included as a boundary edge coincident
    # with a contour?
    ij_a=g_int.nodes['ij'][a]
    ij_b=g_int.nodes['ij'][b]
    ij_c=g_int.nodes['ij'][c]
    if np.any( np.abs(ij_a-ij_b)<1e-10):
        nodes=[]
        for n in [a,b]:
            gn=fixed_int_to_gen.get(n,-1)
            nodes.append( g_final.add_or_find_node(x=g_int.nodes['x'][n],
                                                   gn=gn) )
        g_final.add_edge(nodes=nodes)
    # Second, is there a fixed contour starting at 'b' going into the domain?
    # maybe return to this (which was the point of including a 'c' point.)
    # but I'm thinking a sweep line construction might be better.
    if b in fixed_int_to_gen:
        # gn=fixed_int_to_gen[b]
        ij_angle_ab=np.arctan2( ij_a[1] - ij_b[1],
                                ij_a[0] - ij_b[0] )
        ij_angle_cb=np.arctan2( ij_c[1] - ij_b[1],
                                ij_c[0] - ij_b[0] )

        def trace_from_b(b,dij):
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
            return g_int.trace_node_contour(b,cval=cval,
                                            node_field=node_field,
                                            pos_side=cval_pos)
    
        for dij in [ [-1,0], [1,0], [0,-1],[0,1]]:
            # is dij into the domain?
            dij_angle=np.arctan2( dij[1],dij[0] )

            # does dij_angle fall between the angles formed by the boundary, including
            # a little slop.
            eps=1e-5
            if ( ( (dij_angle-(ij_angle_cb+eps)) % (2*np.pi) )
                 < ( (ij_angle_ab-eps-ij_angle_cb)%(2*np.pi))):
                print(f"{dij} looks good")
                #import pdb
                #pdb.run("
                trace=trace_from_b(b,dij)
                # ")
                print("Trace: ",len(trace))
                # HERE add to g_final, too.
                ax.plot(trace[:,0],trace[:,1],'r-')

# Start at b, trace in the dij direction.
# should have 14.  have 11 attempts.

# And have some issues with contours not quite matching boundaries.
    
## 
g_final.plot_edges(color='tab:green',lw=1.0)

##

# # Draw in the fixed, internal contours
# for n in n_fixed:
#     gn=fixed_int_to_gen[n]
#     break
# 
# for coord in [0,1]: 
#     break

# Use ij to figure outa cardinal ij directions which are into the domain
# g_int.nodes['ij'][n]


##

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






