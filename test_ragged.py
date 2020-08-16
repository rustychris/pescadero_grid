# Southeast marsh channel
import stompy.grid.quad_laplacian as quads
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import six
from stompy import utils


##
six.moves.reload_module(quads)
gen=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v01.pkl')

# the grid at this point is super f'd
plt.figure(1).clf()
gen.plot_edges()
gen.plot_nodes(labeler=lambda i,r: f"{r['i']},{r['j']}")

##
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v01.pkl')

qg=quads.QuadGen(gen_src,cell=0,anisotropic=False,execute=False,nom_res=3)

# qg.gen.nodes now has IJ
# qg.execute()
# def create_intermediate_grid(self,

src='IJ'
coordinates='xy'

# target grid
g=unstructured_grid.UnstructuredGrid(max_sides=4,
                                     extra_node_fields=[('ij',np.float64,2),
                                                        ('gen_j',np.int32),
                                                        ('rigid',np.int32)])
gen=qg.gen
for c in gen.valid_cell_iter():
    break

local_edges=gen.cell_to_edges(c,ordered=True)
flip=(gen.edges['cells'][local_edges,0]!=c)

edge_nodes=gen.edges['nodes'][local_edges]
edge_nodes[flip,:] = edge_nodes[flip,::-1]

dijs=gen.edges['d'+src][local_edges] * ((-1)**flip)[:,None]
xys=gen.nodes['x'][edge_nodes[:,0]]
ij0=gen.nodes[src][edge_nodes[0,0]]
ijs=np.cumsum(np.vstack([ij0,dijs]),axis=0)

# Sanity check to be sure that all the dijs close the loop.
assert np.allclose( ijs[0],ijs[-1] )

ijs=np.array(ijs[:-1])
# Actually don't, so that g['ij'] and gen['ij'] match up.
ij0=ijs.min(axis=0)
ijN=ijs.max(axis=0)
ij_size=ijN-ij0

# Create in ij space
patch=g.add_rectilinear(p0=ij0,
                        p1=ijN,
                        nx=int(1+ij_size[0]),
                        ny=int(1+ij_size[1]))
pnodes=patch['nodes'].ravel()

g.nodes['gen_j'][pnodes]=-1

# Copy xy to ij, then optionally remap xy
g.nodes['ij'][pnodes] = g.nodes['x'][pnodes]

##

# What does this look like in IJ space?
# Is there a case where the xy triangulation would become
# invalid in IJ space?
# Could easily have a kink in xy that gets a local triangle,
# and that becomes flat/degenerate in IJ.


plt.figure(1).clf()
fig,axs=plt.subplots(2,1,num=1)
gen.plot_edges(ax=axs[0],lw=2,color='k')

axs[1].plot(ijs[:,0],ijs[:,1],'k-',lw=2,label='gen IJ')

axs[1].plot( gen.nodes['IJ'][:,0], gen.nodes['IJ'][:,1], 'r.')
# Create a triangulation in IJ space:
from stompy.grid import exact_delaunay
six.moves.reload_module(exact_delaunay)
tri_ij=exact_delaunay.Triangulation()
tri_ij.init_from_grid(gen,'IJ')

tri_ij.plot_edges(ax=axs[1])

tri_xy=exact_delaunay.Triangulation()
tri_xy.init_from_grid(gen,'x',set_valid=True)
tri_xy.plot_edges(ax=axs[0])
for ax in axs:
    ax.axis('tight')
    ax.axis('equal')
tri_xy_ij=tri_xy.copy()
tri_xy_ij.nodes['x']=gen.nodes['IJ']
to_delete=(~tri_xy_ij.cells['valid'])&(~tri_xy_ij.cells['deleted'])
for c in np.nonzero( to_delete )[0]:
    tri_xy_ij.delete_cell(c)

tri_xy_ij.delete_orphan_edges()
tri_xy_ij.delete_orphan_nodes()
tri_xy_ij.renumber()
tri_xy_ij.plot_edges(ax=axs[1],color='g')

## 
# Delete cells outside the original grid.

# Convert the xy triangulation to IJ space.

## -------------------------------

# Go back to solving on a fully unstructured triangular grid, so construction
# is easier

six.moves.reload_module(quads)
from stompy.grid import triangulate_hole

# v00 has a ragged edge.
# v01 makes that right.
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v00.pkl')

qg=quads.QuadGen(gen_src,cell=0,anisotropic=False,execute=False,
                 gradient_scale=1.0)

gen=qg.gen
qg.add_bezier(gen)

g=unstructured_grid.UnstructuredGrid(max_sides=4,
                                     extra_edge_fields=[ ('gen_j',np.int32) ],
                                     extra_node_fields=[ ('ij',np.float64,2) ])
g.nodes['ij']=np.nan

res=10.0
src='IJ'
for j in gen.valid_edge_iter():
    # Just to get the length
    points=qg.gen_bezier_linestring(j=j,samples_per_edge=10,span_fixed=False)
    dist=utils.dist_along(points)[-1]
    N=max( 4, int(dist/res))
    points=qg.gen_bezier_linestring(j=j,samples_per_edge=N,span_fixed=False)

    # Figure out what IJ to assign:
    ij0=gen.nodes[src][gen.edges['nodes'][j,0]]
    ijN=gen.nodes[src][gen.edges['nodes'][j,1]]

    nodes=[]
    for p_i,p in enumerate(points):
        n=g.add_or_find_node(x=p,tolerance=0.1)
        alpha=p_i/(len(points)-1.0)
        assert alpha>=0
        assert alpha<=1
        ij=(1-alpha)*ij0 + alpha*ijN
        g.nodes['ij'][n]=ij
        nodes.append(n)
        
    for a,b in zip(nodes[:-1],nodes[1:]):
        g.add_edge(nodes=[a,b],gen_j=j)
    
plt.figure(1).clf()

g.plot_edges(values=g.edges['gen_j'],cmap='rainbow')
g.plot_nodes(marker='.')


seed=gen.cells_centroid()[0]
# This will suffice for now.  Probably can use something
# less intense.
gnew=triangulate_hole.triangulate_hole(g,seed_point=seed,hole_rigidity='all')

plt.figure(1).clf()

g.plot_edges(values=g.edges['gen_j'],cmap='rainbow')

g.plot_nodes(mask=np.isfinite(g.nodes['ij'][:,0]),
             labeler=lambda i,r: f"{r['ij'][0]:.2f},{r['ij'][1]:.2f}")


## 
# Rather than the finite difference approach, is there a boxed up FEM
# way I could solve the laplacian?

# First, use this grid and my existing method
qg.g_int=gnew
qg.calc_psi_phi()
qg.plot_psi_phi()

# 
# Trouble -- how are my edges getting off?
# BCs or something are messed up?
# Is the problem that I'm not specify corners as rigid?
# g.nodes['rigid'][n]=RIGID
# But that's not used later...
# Maybe this provides some clues on DOF




## Revert to quad intermediate:

qg=quads.QuadGen(gen_src,cell=0,anisotropic=False,execute=False,nom_res=10,
                 smooth_iterations=5)
qg.execute()
## 
qg.plot_intermediate()

qg.g_int.plot_nodes(labeler='id')

# Somehow two nodes are stacking up at the inside corner.

nodes=[119,149]
# qg.g_int.
# ij for those:
# array([[ 3., 29.],
#        [ 4., 29.]])
# So that should be an edge

## 
# sfepy is promising.

plt.figure(1).clf()
gnew.plot_edges()
