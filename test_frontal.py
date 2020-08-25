from stompy.grid import unstructured_grid
from stompy.grid import front, exact_delaunay
import stompy.grid.quad_laplacian as quads
from stompy.spatial import field
import six
import heapq

import matplotlib.pyplot as plt

##
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(front)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(quads)

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v00.pkl')

DELETED=-1
EXT=0
WAITING=1
ACTIVE=2
DONE=3

class RebayAdvancingDelaunay(front.AdvancingFront):
    """ 
    Implementation of Rebay, 1993 for fast triangular mesh generation.

    The approach is quite different from AdvancingFront, so while this is 
    a subclass there is relatively little shared code.
    """
    
    scale=None
    def __init__(self,grid=None,**kw):
        """
        grid: edges will be copied into a constrained delaunay triangulation,
        and cells will be used to define what is 'inside'
        """

        self.grid=exact_delaunay.Triangulation()
        utils.set_keywords(self,kw)

        if grid is not None:
            self.init_from_grid(grid)
            
        self.instrument_grid()

    def init_from_grid(self,g):
        self.grid.init_from_grid(g,set_valid=True,valid_min_area=1e-2)

    def instrument_grid(self):
        self.grid.add_cell_field( 'stat', np.zeros( self.grid.Ncells(),np.int32 ) )
        self.grid.add_cell_field( 'radius', np.zeros( self.grid.Ncells(),np.float64 ) )

# This is the edge scale.
scale=field.ConstantField(10.0)
# radius scale, for an equilateral triangle follows this:
#rad_scale = (1/1.73) * scale
# Generally don't have equilateral, so derate that some.
rad_scale = (1/1.5) * scale

# Prepare a nice input akin to what quad laplacian will provide:
qg=quads.QuadGen(gen=gen_src,execute=False,cell=0)
qg.add_bezier(qg.gen)
gsmooth=qg.create_intermediate_grid_tri_boundary(src='IJ',scale=scale)

#plt.figure(1).clf()
#fig,ax=plt.subplots(num=1)
#gen_src.plot_edges(ax=ax,color='0.7',lw=2.5,zorder=-2)
#gsmooth.plot_edges(ax=ax)
#gsmooth.plot_nodes(ax=ax)

# quad laplacian does not currently do this step:
gsmooth.modify_max_sides(gsmooth.Nnodes())
gsmooth.make_cells_from_edges()
assert gsmooth.Ncells()>0

rad=RebayAdvancingDelaunay(grid=gsmooth,scale=scale)

##
# subdivide edges as needed:

node_pairs=[]
for j in np.nonzero(rad.grid.edges['constrained'])[0]:
    node_pairs.append( rad.grid.edges['nodes'][j] )
    rad.grid.remove_constraint(j=j)

# once an edge exists, will subdividing another edge potentially
# break it?  for now, be conservative and check for that

while 1:
    new_node_pairs=[]
    for a,b in node_pairs:
        j=rad.grid.nodes_to_edge(a,b)
        if j is None:
            # subdivide:
            print("subdivide")
            mid=0.5*(rad.grid.nodes['x'][a]+rad.grid.nodes['x'][b])
            n_mid=rad.grid.add_node(x=mid)
            new_node_pairs.append( [a,n_mid] )
            new_node_pairs.append( [n_mid,b] )
        else:
            new_node_pairs.append( [a,b] )
    assert len(new_node_pairs)>=len(node_pairs)
    if len(new_node_pairs)==len(node_pairs):
        break
    node_pairs=new_node_pairs
    
##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
# gen_src.plot_edges(ax=ax,color='0.7',lw=2.5,zorder=-2)

g=rad.grid

g.plot_edges(values=g.edges['constrained'])
g.plot_cells(values=g.cells['valid'],cmap='winter',zorder=-1)

g.cells['stat'][ ~g.cells['valid'] ] = EXT

# What do we have to listen for?
# We're tracking cells, and need
cell_log=[] # [ ['action', cell id], ...]
def on_delete_cell(_,func,cell,*a,**k):
    cell_log.append( [func,cell,k] )
def on_modify_cell(_,func,cell,**k):
    cell_log.append( [func,cell,k] )
def on_add_cell(_,func,**k):
    cell=k['return_value']
    cell_log.append( [func,cell,k] )
g.subscribe_after( 'delete_cell', on_delete_cell )
g.subscribe_after( 'modify_cell', on_modify_cell )
g.subscribe_after( 'add_cell', on_add_cell )

# Q: original algorithm doesn't use a constrained triangulation.
# If it becomes problematic, return here and add their resampling
# approach.
# cc=g.constrained_centers()
cc=g.cells_center()
centroids=g.cells_centroid()

g.cells['radius']=utils.dist( cc - g.nodes['x'][ g.cells['nodes'][:,0] ] )
target_radii=rad_scale(centroids)

alpha=g.cells['radius']/target_radii

valid=g.cells['valid']

g.cells['stat'][ valid & (alpha<1.0) ] = DONE
g.cells['stat'][ valid & (alpha>=1.0) ] = WAITING

#-- 
# And ACTIVE:

e2c=g.edge_to_cells(recalc=True)
assert e2c[:,0].min()>=0,"Was hoping that this invariant is honored"

good_ext=(e2c<0) | (g.cells['stat'][e2c]==DONE) | (g.cells['stat'][e2c]==EXT)
good_int_cells = (g.cells['stat']==WAITING) | (g.cells['stat']==ACTIVE)
good_int = good_int_cells[ e2c ]

active_c1=good_ext[:,0] & good_int[:,1]
active_c0=good_ext[:,1] & good_int[:,0]

g.cells['stat'][ e2c[active_c0,0] ] = ACTIVE
g.cells['stat'][ e2c[active_c1,1] ] = ACTIVE

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

ccoll=g.plot_cells(values=g.cells['stat'],cmap='rainbow',ax=ax)
ccoll.set_clim([0,3])
ax.axis('equal')

#----

# Algorithm:
# Choose the active cell with the largest circumradius.

# Will heapq do what I need?
#  it's just methods on a list. there's not a way to easily
#  update an element's cost 

active_heap=[] # elements are [radius, cell, valid]
active_hash={} # cell=>entry in active_heap

def push_cell_radius(c,r):
    if c in active_hash:
        remove_active(c)
    entry=[r,c,True]
    heapq.heappush(active_heap, entry)
    active_hash[c]=entry

def remove_active(cell):
    entry = active_hash.pop(cell)
    entry[-1] = False

def pop_active():
    while active_heap:
        radius, cell, valid = heapq.heappop(active_heap)
        if valid:
            del active_hash[cell]
            return cell
    return None

for c in np.nonzero( g.cells['stat']==ACTIVE )[0]:
    push_cell_radius( c, radii[c] )

##     
# One iteration:

# The constrained DT does introduce some issues.
# The circumcenter may fall outside the triangle, and specifically
# outside the domain.
# Even relaxing things a bit, it's no good.
# How about just testing whether xm-CA crosses a constrained
# edge (only have to test 2 edges...)
# And in that case, subdivide the edge instead.

# HERE
# Well...... not so fast.  I introduced the subdividing and still
# get a fail.  Might have to add a counter, and stop at the time of
# the fail to see what's going on.


c_target=pop_active()
if c_target is None:
    raise StopIteration()

# Select the edge of c_target:
j_target=None
j_target_L=np.inf

e2c=g.edge_to_cells(recalc=True)
assert e2c[:,0].min()>=0,"Was hoping that this invariant is honored"

for j in g.cell_to_edges(c_target):
    c0,c1=e2c[j]
    if c1==c_target:
        c_nbr=c0
    elif c0==c_target:
        c_nbr=c1
    else:
        assert False
    if (c_nbr<0) or (g.cells['stat'][c_nbr] in [EXT,DONE]):
        # It's a candidate -- is the shortest candidate?
        # Rebay notes this is a good, but not necessarily
        # optimal choice
        L=g.edges_length(j)
        if L<j_target_L:
            j_target=j
            j_target_L=L
j=j_target
xm=0.5*(g.nodes['x'][g.edges['nodes'][j,0]] +
        g.nodes['x'][g.edges['nodes'][j,1]])
C_A=g.cells_center(refresh=[c_target])[c_target]

rho_m=rad_scale(xm)
p=0.5*j_target_L
q=utils.dist( C_A, xm)
rho_hat_m = min( max(rho_m,p), (p**2+q**2)/(2*q))

d=rho_hat_m + np.sqrt( rho_hat_m**2 - p**2)
assert np.isfinite(d) # sanity
e_vec=utils.to_unit( C_A - xm)

new_x=xm+ d*e_vec

g.add_node(x=new_x)

changed_cells=set( [ entry[1] for entry in cell_log ] )
del cell_log[:]

#---

Nc=g.Ncells()
live_cells=[]

for c in changed_cells:
    if c in active_hash:
        remove_active(c)
    
    if c>=Nc: # deleted and cell array truncated
        continue
    
    if g.cells['deleted'][c]:
        g.cells['stat'][c]=DELETED
        continue
    
    # Either modified or added.  update radius,
    # status, and potentially requeue
    cc=g.cells_center(refresh=[c])[c]
    rad=utils.dist( cc - g.nodes['x'][ g.cells['nodes'][c,0] ] )
    g.cells['radius'][c]=rad
    g.cells['valid'][c]=True # Should be an invariant...
    live_cells.append(c)

live_cells=np.array(live_cells)

centers=g.cells_centroid(live_cells)
target_radii=rad_scale(centers)

# add 0.1 for some slop.
done=(g.cells['radius'][live_cells] / target_radii) < 1.1

g.cells['stat'][ live_cells[done] ]=DONE

from itertools import chain

def set_active(c):
    g.cells['stat'][c]=ACTIVE
    push_cell_radius(c, g.cells['radius'][c])
    
for c in live_cells[~done]:
    # ACTIVE or WAITING?
    nbrs=g.cell_to_cells(c,ordered=True) # ordered doesn't really matter I guess
    for nbr in nbrs:
        if (nbr<0) or (g.cells['stat'][nbr] in [DONE,EXT]):
            set_active(c)
            break
    else:
        g.cells['stat'][c]=WAITING

for c in live_cells[done]:
    for nbr in g.cell_to_cells(c,ordered=True):
        if nbr>=0 and g.cells['stat'][nbr]==WAITING:
            set_active(nbr)

# And loop through the neighbors of cells adjacent to live_cells[done]
# and set them to be active.
    
#--

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(color='k',lw=0.5)
ax.plot( [xm[0]], [xm[1]], 'bo')
ax.plot( [new_x[0]], [new_x[1]], 'go')
ccoll=g.plot_cells(values=g.cells['stat'],cmap='rainbow',ax=ax)
ccoll.set_clim([0,3])

ax.axis('equal')

##
