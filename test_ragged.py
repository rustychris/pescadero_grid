# Southeast marsh channel
import stompy.grid.quad_laplacian as quads
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import six
from stompy import utils


##
six.moves.reload_module(quads)

# Go back to solving on a fully unstructured triangular grid, so construction
# is easier

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

res=6.0
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
g.node_defaults['ij']=np.nan
gnew=triangulate_hole.triangulate_hole(g,seed_point=seed,hole_rigidity='all')

plt.figure(1).clf()

g.plot_edges(values=g.edges['gen_j'],cmap='rainbow')

g.plot_nodes(mask=np.isfinite(g.nodes['ij'][:,0]),
             labeler=lambda i,r: f"{r['ij'][0]:.2f},{r['ij'][1]:.2f}")


##
# i_tan_groups has node 139 in two different groups.  bad.
# node 139 is the last in the bcycle, and also n1 on the
# first go through
six.moves.reload_module(quads)
qg=quads.QuadGen(gen_src,cell=0,anisotropic=False,execute=False,
                 gradient_scale=1.0)

gen=qg.gen
qg.add_bezier(gen)

# First, use this grid and my existing method
qg.g_int=gnew

qg.calc_psi_phi()
qg.plot_psi_phi(thinning=0.5)
plt.axis( (552313.3506335834, 552414.4919506207, 4124365.0316236005, 4124468.394946426) )


# HERE -
# Finally have a nice psi/phi field.
# 1. Move the above triangulation logic into quad_laplacian as an option.
# 2. See if that resolves the issue with north channel, and finish the interpolation.
# 3. Come back to here and see if I can finish a quad grid with a ragged edge

##

# Use nd just to see what the gradients look like
# Looks entirely reasonable
gtri=gnew

nd=quads.NodeDiscretization(gnew)

bcycle=gtri.boundary_cycle()

grad_psi=np.zeros( (len(bcycle),2), np.float64)
grad_phi=np.zeros( (len(bcycle),2), np.float64)

for ni,n in enumerate(bcycle):
    dx_nodes,dx_coeffs,rhs=nd.node_dx(n)
    dy_nodes,dy_coeffs,rhs=nd.node_dy(n)

    grad_psi[ni,0] = (qg.psi[dx_nodes]*dx_coeffs).sum()
    grad_psi[ni,1] = (qg.psi[dy_nodes]*dy_coeffs).sum()

    grad_phi[ni,0] = (qg.phi[dx_nodes]*dx_coeffs).sum()
    grad_phi[ni,1] = (qg.phi[dy_nodes]*dy_coeffs).sum()


plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
gnew.plot_edges(lw=0.3,ax=ax)

grad_psi=utils.to_unit(grad_psi)

bc_grad_psi,bc_grad_phi=qg.calc_bc_gradients(gtri)

if 1: # show psi, internal and BC
    ax.quiver( gnew.nodes['x'][bcycle,0], gnew.nodes['x'][bcycle,1],
               grad_psi[:,0], grad_psi[:,1],color='b')

    bc_grad=np.array( [bc_grad_psi[n] for n in bcycle] )
        
    ax.quiver( gnew.nodes['x'][bcycle,0], gnew.nodes['x'][bcycle,1],
               bc_grad[:,0],bc_grad[:,1],
               color='tab:green')
if 0:
    ax.quiver( gnew.nodes['x'][:,0], gnew.nodes['x'][:,1],
               grad_phi[:,0], grad_phi[:,1],color='red')
    ax.quiver(xys[:,0],xys[:,1], bc_grad_phi[:,0], bc_grad_phi[:,1], color='tab:orange')

##
# Maybe the failures of lagoon-v01 provide some clues on DOF

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
# but scikit-fem is more my speed.  pure python, and should be enough for a simple problem
# like this.

plt.figure(1).clf()
gnew.plot_edges()

## 
