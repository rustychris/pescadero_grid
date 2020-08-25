# Southeast marsh channel
import stompy.grid.quad_laplacian as quads
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import six
from stompy import utils

##

# v00 has a ragged edge.
# v01 makes that a right. angle
six.moves.reload_module(quads)

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v00.pkl')

# hmm - failed??  try nom res 5
# Okay, but one node is corrupted.
qg=quads.QuadGen(gen_src,cell=0,final='anisotropic',execute=True,nom_res=5,
                 gradient_scale=1.0)

# gen=qg.gen
# qg.add_bezier(gen)
# qg.g_int=qg.create_intermediate_grid_tri(src='IJ')
# 
# qg.calc_psi_phi()
# qg.plot_psi_phi(thinning=0.5)
# plt.axis( (552313.3506335834, 552414.4919506207, 4124365.0316236005, 4124468.394946426) )
qg.plot_result()

##

# HERE
# Derive a 'proper' IJ field by looking at psi/phi.  I.e. IJ=some scaling of psi/phi
# at the fixed nodes.
# Try cubic interpolation?



##
# Updating the logic when g_int is triangular, and we adjust a quad output
# For the moment, that's an isotropic quad output
self=qg
self.g_final=self.create_intermediate_grid_quad(src='IJ',coordinates='ij')
# this looks fine -- it's in IJ space, 16 units wide, 81 long, has the
# taper.  nice.
# I: [0,16]
# J: [0,81]

plt.figure(3).clf()
self.g_final.plot_edges(color='k',lw=0.5)
plt.axis('equal')

g=self.g_final
src='IJ'
#-------------------
# def adjust_by_psi_phi(self,g,update=True,src='ij'):
#     """
#     Move internal nodes of g according to phi and psi fields
# 
#     update: if True, actually update g, otherwise return the new values
# 
#     g: The grid to be adjusted. Must have nodes['ij'] filled in fully.
# 
#     src: the ij coordinate field in self.gen to use.  Note that this needs to be
#       compatible with the ij coordinate field used to create g.
#     """
# Check to be sure that src and g['ij'] are approximately compatible.
assert np.allclose( g.nodes['ij'].min(), self.gen.nodes[src].min() )
assert np.allclose( g.nodes['ij'].max(), self.gen.nodes[src].max() )

map_ij_to_pp = self.psiphi_to_ij(self.gen,self.g_int,inverse=True,src=src)

# Calculate the psi/phi values on the nodes of the target grid
g_psiphi=map_ij_to_pp( g.nodes['ij'] ) # had a bug there..


# Use self.g_int to go from phi/psi to x,y
# I think this is where it goes askew.
# This maps {psi,phi} space onto {x,y} space.
# But psi,phi is close to rectilinear, and defined on a rectilinear
# grid.  Whenever some g_psi or g_phi is close to the boundary,
# the Delaunay triangulation is going to make things difficult.
interp_xy=utils.LinearNDExtrapolator( np.c_[self.psi,self.phi],
                                      self.g_int.nodes['x'],
                                      eps=None)
# Save all the pieces for debugging:
self.interp_xy=interp_xy
self.interp_domain=np.c_[self.psi,self.phi]
self.interp_image=self.g_int.nodes['x']
self.interp_tgt=g_psiphi

new_xy=interp_xy( g_psiphi )

new_xy[np.isnan(new_xy)]=0.0 # DEV

#if update:
if True:
    g.nodes['x']=new_xy
    g.refresh_metadata()
#else:
#    return new_xy
#--------------------


# adjust_by_psi_phi(self,self.g_final,src='IJ')
## 
plt.figure(2).clf()
self.g_final.plot_edges(color='k',lw=0.5)

i_psi=map_ij_to_pp.__defaults__[0]
j_phi=map_ij_to_pp.__defaults__[1]

g.contour_node_values(g_psiphi[:,0],i_psi[:,1],colors='tab:red',linestyles='solid')
g.contour_node_values(g_psiphi[:,1],j_phi[::-1,1],colors='tab:green',linestyles='solid')

# Plot the nodes of gen that are providing these:
coord=0 # i/psi
gen=self.gen
gen_valid=(~gen.nodes['deleted'])&(gen.nodes[src+'_fixed'][:,coord])
gen.plot_nodes(mask=gen_valid,color='k',
               labeler=lambda i,r: f"{gen.nodes['IJ'][i,coord]}")
gen.plot_edges(color='k',lw=0.4)

# One issue is that IJ has the ragged part as half of the total
# width, while it "should" be smaller.
# but this 

all_coord=gen.nodes[src][gen_valid,coord]

plt.plot( new_xy[:,0],new_xy[:,1], 'g.')

plt.axis('equal')


## 
map_ij_to_pp = self.psiphi_to_ij(self.gen,self.g_int,inverse=True)


##
self.adjust_by_psi_phi(self.g_final, src='IJ')
ij=self.remap_ij(self.g_final,src='ij')
self.g_final.nodes['ij']=ij


#   self.g_final=self.g_int.copy()
#   self.adjust_by_psi_phi(self.g_final, src='IJ')
#   # but update ij to reflect the 'ij' in the original input.
#   ij=self.remap_ij(self.g_final,src='ij')
#   self.g_final.nodes['ij']=ij

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
