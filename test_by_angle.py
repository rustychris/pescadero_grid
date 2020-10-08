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
nice_colors=itertools.cycle(colors.TABLEAU_COLORS)

import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')
cmap=scmap.load_gradient('oc-sst.cpt')

##

from stompy.grid import triangulate_hole, rebay

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

# v06 puts angles on half-edges
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v07.pkl')

gen_src.renumber_cells()

##

from stompy.spatial import wkb2shp, constrained_delaunay

CXYZ=constrained_delaunay.ConstrainedXYZField

i_tele=field.ApolloniusField.read_shps(['scale.shp'],value_field='i_tele')
j_tele=field.ApolloniusField.read_shps(['scale.shp'],value_field='j_tele')

i_linear=CXYZ.read_shps(['scale.shp'],value_field='i_linear')
j_linear=CXYZ.read_shps(['scale.shp'],value_field='j_linear')

i_scale=field.BinopField( i_tele, np.minimum, i_linear)
j_scale=field.BinopField( j_tele, np.minimum, j_linear)

## 
plt.figure(1).clf()
gen_src.plot_cells(labeler='id',centroid=True)
plt.axis('tight')
plt.axis('equal')

##
# Make sure that each individual cell works:
# good.
for c in gen_src.valid_cell_iter():
    qg=quads.QuadGen(gen_src,cells=[c],final='anisotropic',execute=False,
                     nom_res=3.5,
                     scales=[i_scale,j_scale])
    qg.execute()
    qg.plot_result(num=100+c)

    qg.g_final.write_ugrid('pieces_cell%03d.nc'%c,
                           overwrite=True)
    
##

g=None
for c in gen_src.valid_cell_iter():
    g_sub=unstructured_grid.UnstructuredGrid.read_ugrid('pieces_cell%03d.nc'%c)
    if g is None:
        g=g_sub
    else:
        g.add_grid(g_sub)
g.write_ugrid('combined-pieces.nc',overwrite=True)        

##

# Find the smallest set of cells that still has issues
# all cells: issues in cell 3 (butano marsh),
# junction of 11 and 9 (butano marsh to ck)
# actually, lots of places.
# but 1,0,6 is okay.
# 1,0,6,9 is okay.
# +10,11 is okay
# +3 and its bad.
# 3,9,10,11 is bad
# 3,10,11 is bad.
# 3,10 is bad.
# 3 alone is bad.
# if i use rebay grid, is it still bad? nope.
qg=quads.QuadGen(gen_src,cells=[3],
                 final='anisotropic',execute=False,
                 nom_res=3.5,triangle_method='rebay',
                 scales=[i_scale,j_scale])
# qg.execute()
qg.prepare_angles()
qg.add_bezier(qg.gen)
qg.g_int=qg.create_intermediate_grid_tri()
qg.calc_psi_phi()
qg.plot_psi_phi(thinning=0.15)

##
qg.plot_psi_phi_setup()
plt.axis('tight')
plt.axis('equal')
##

M=qg.Mblocks[0][0]
print(f"psi block: {M.shape}.  Deficient by {M.shape[1]-M.shape[0]}")

# 8 dofs shy (regardless of src grid)
# probably the number of corners?
# 2 dirichlet nodes
# 9 tangent groups (10 for j)
# 11 gradient nodes
# there are 20 corners
# 2 are along a diagonal.
# 

# gmsh gave M.shape 4117x4125
# rebay is 2441 x 2449

##

# What do I get if I solve just psi?
# bad.
psi1=sparse.linalg.lsqr( qg.Mblocks[0][0], qg.Bblocks[0])

##

zoom=(552547.5580429004, 552586.687028341, 4123426.3512172685, 4123455.5086225485)
plt.figure(5).clf()
fig,ax=plt.subplots(num=5)
qg.g_int.contourf_node_values( psi2,200, cmap='flag',alpha=0.1)
# qg.g_int.plot_nodes(labeler='id',clip=zoom)
qg.g_int.plot_edges(lw=0.4, alpha=0.5,color='k')

ax.axis('equal')
ax.axis(zoom)

##

# a problem node, 270:
bad_x=[552563.9669722788, 4123444.90592972]
bad_n=qg.g_int.select_nodes_nearest(bad_x)

# # What sorts of constraints is this involved in?
# M=qg.Mblocks[0][0]
# 
# # Which rows have a nonzero entry for bad_n?
# bad_n_hits=np.nonzero( M[:,bad_n].todense() )[0]
# 
# print(f"There are {len(bad_n_hits)} rows involving the bad node {bad_n}")
# 
# for row in bad_n_hits:
#     print(f"{row}:  ",end="")
#     row_members=np.nonzero( M[row,:].todense() )[1]
#     print( row_members)
#     print()


##

# What is the gradient here?
# Definitely bad.
nd=qg.nd
bad_dx=nd.node_dx(bad_n)
bad_dy=nd.node_dy(bad_n)

f=psi2[0]

bad_grad= [ (f[bad_dx[0]] * bad_dx[1]).sum(),
            (f[bad_dy[0]] * bad_dy[1]).sum() ]
            
ax.quiver( [qg.g_int.nodes['x'][bad_n,0]],
           [qg.g_int.nodes['x'][bad_n,1]],
           [bad_grad[0]],[bad_grad[1]])

## 
# What if I add a gradient BC for each internal corner?
# First, recreate

psi_gradients,phi_gradients=qg.calc_bc_gradients(qg.g_int)

i_grad_nodes=dict(qg.i_grad_nodes)

for n in np.nonzero( qg.g_int.nodes['rigid'])[0]:
    gen_n=qg.g_int.nodes['gen_n'][n]
    assert n>=0
    turn=qg.gen.nodes['turn'][gen_n]
    if turn!=90:
        if n not in qg.i_grad_nodes:
            # Add it!
            print(n,turn)
            i_grad_nodes[n] = psi_gradients[n]

M_psi_Lap,B_psi_Lap=nd.construct_matrix(op='laplacian',
                                        dirichlet_nodes=qg.i_dirichlet_nodes,
                                        zero_tangential_nodes=qg.i_tan_groups,
                                        gradient_nodes=i_grad_nodes)

# Looks better, but still not great.
# Is it a tolerance issue?
# I still have on extra dof.
psi2,istop,itn,r1norm,r2norm,anorm,arnorm,acond,xnorm,v=sparse.linalg.lsqr( M_psi_Lap, B_psi_Lap,
                                                                            atol=1e-10,
                                                                            btol=1e-10)
qg.i_grad_nodes=i_grad_nodes

##


# Is there something about putting BCs on nodes vs. edges that
# would explain how I end up 1 short?

# am I missing a global conservation of mass?
# like all of the inflows should balance the outflows?

# For a rectangle:
#  each non-corner node on closed boundaries solves
#  the laplacian, along with each internal node.
#  each set of open boundary nodes implies a constant
#  value.

qg=quads.QuadGen(gen_src,cells=[3],
                 final='anisotropic',execute=False,
                 nom_res=3.5,triangle_method='rebay',
                 scales=[i_scale,j_scale])
# qg.execute()
qg.prepare_angles()
qg.add_bezier(qg.gen)
qg.g_int=qg.create_intermediate_grid_tri()
qg.internal_edges=[] # While testing, drop the internal edges
qg.calc_psi_phi()
qg.plot_psi_phi(thinning=0.2)

##

# This domain has 4 tangential gradient BCs, and 4 normal gradient BCs

i_grad_nodes=dict(qg.i_grad_nodes)

# This is bad whether I include these normals or not.
nd=qg.nd

M_psi_Lap,B_psi_Lap=nd.construct_matrix(op='laplacian',
                                        dirichlet_nodes=qg.i_dirichlet_nodes,
                                        skip_dirichlet=False,
                                        zero_tangential_nodes=qg.i_tan_groups,
                                        gradient_nodes=i_grad_nodes)

print(f"{M_psi_Lap.shape}")

## 
# Manually add in the normal constraints
# qg.g_int.plot_nodes(labeler='id')
coord=0 # signify we're working on psi

# Find these automatically.
noflux_tris=[]
for n in np.nonzero(qg.g_int.nodes['rigid'])[0]:
    gen_n=qg.g_int.nodes['gen_n'][n]
    assert gen_n>=0
    gen_angle=qg.gen.nodes['turn'][gen_n]
    # For now, ignore non-cartesian, and 90
    # degree doesn't count
    if gen_angle not in [270,360]: continue
    if gen_angle==270:
        print(f"n {n}: angle=270")
    elif gen_angle==360:
        print(f"n {n}: angle=360")

    js=qg.g_int.node_to_edges(n)
    e2c=qg.g_int.edge_to_cells()

    for j in js:
        if (e2c[j,0]>=0) and (e2c[j,1]>=0): continue
        gen_j=qg.g_int.edges['gen_j'][j]
        angle=qg.gen.edges['angle'][gen_j]
        if qg.g_int.edges['nodes'][j,0]==n:
            nbr=qg.g_int.edges['nodes'][j,1]
        else:
            nbr=qg.g_int.edges['nodes'][j,0]
        print(f"j={j}  {n} -- {nbr}  angle={angle}")
        # Does the angle 
        if (angle + 90*coord)%180. == 90.:
            print("YES")
            c=e2c[j,:].max()
            tri=qg.g_int.cells['nodes'][c]
            while tri[2] in [n,nbr]:
                tri=np.roll(tri,1)
            noflux_tris.append( tri )
        
nf_block=sparse.dok_matrix( (len(noflux_tris),qg.g_int.Nnodes()), np.float64)
nf_rhs=np.zeros( len(noflux_tris) )
node_xy=qg.g_int.nodes['x'][:,:]

for idx,tri in enumerate(noflux_tris):
    target_dof=idx # just controls where the row is written
    d01=node_xy[tri[1],:] - node_xy[tri[0],:]
    d02=node_xy[tri[2],:] - node_xy[tri[0],:]
    # Derivation in sympy below
    nf_block[target_dof,:]=0 # clear old
    nf_block[target_dof,tri[0]]= -d01[0]**2 + d01[0]*d02[0] - d01[1]**2 + d01[1]*d02[1]
    nf_block[target_dof,tri[1]]= -d01[0]*d02[0] - d01[1]*d02[1]
    nf_block[target_dof,tri[2]]= d01[0]**2 + d01[1]**2
    nf_rhs[target_dof]=0

M=sparse.bmat( [ [M_psi_Lap],[nf_block]] )
B=np.concatenate( [B_psi_Lap,nf_rhs] )

# Almost Good! But one little corner goes nuts.  Fixed that
# by not dropping rows that are also dirichlet.  
# qg.psi,istop,itn,r1norm,r2norm,anorm,arnorm,acond,xnorm,v=sparse.linalg.lsqr( M, B )
qg.phi[:]=0
# Direct solve is reasonably fast and gave better result.
qg.psi=sparse.linalg.spsolve(M.tocsr(),B)

qg.plot_psi_phi(thinning=0.2)
qg.plot_psi_phi_setup()

##

from scipy.linalg.interpolative import estimate_rank

r=estimate_rank(M_psi_Lap.todense(),eps=1e-10)
print(r)

##

qg.plot_psi_phi_setup()

##

# Revisiting the discussion of degrees of freedom
#   N nodes
#    all internal nodes 

## 
# HERE:
#  1. Matrix solve is dicey in larger domain
#     => try scikit fem.  Tried, but its pretty opaque.
#  2. Intermediate grid gen is slow
#     => add gmsh option for triangulate_hole. DONE.
#  3. Scale specification is error prone:
#     => Use orientation of linestring to set i/j axes

##

# looks like I need to back up and understand the forms better.
# For the interior, I'm solving
# del dot del u = 0
# the laplace eqn

# somehow this ends up as int  grad u dot grad v

# del dot del u = 0
# (del dot del u) * v = 0 * v

# int del dot del u * v dx = int 0 dx

# there is some requirement that v satisfy the BCs.
# not sure why.  moving on

# integrate by parts:
# following the 1D example:

# int u'' v dx  ==  u' (v | 0,1) - int u' v' dx

# that's another way of saying
# int u'' v dx  + int u' v' dx  ==  u' (v | 0,1)

# or imagine taking the derivative of both sides:

# (u'' v)   +  (u'v') == d/dx ( u' v )

# so the integration by parts is roughly the integral version
# of the product rule.  Take the original integrand with u'' v
# turns into two terms:
#  int u'' int v  - int u' v'

#  back to my case:
# int del dot del u * v dx

# => (int del dot del u) * (int v) - int( del u * del v )
# => int del u * int v  -  int( del u * del v)

# Somehow green's thm lets that first bit go away


## 
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

qg=quads.QuadGen(gen_src,cells=[0],final='anisotropic',execute=False,
                 nom_res=3.5,triangle_method='gmsh',
                 scales=[i_scale,j_scale])

qg.execute()
#qg.plot_psi_phi_setup()

## 
# How well does this translate scikit-fem?
from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import laplace, unit_load
from skfem.visuals.matplotlib import plot

## 
m = MeshTri(p=qg.g_int.nodes['x'].T,
            t=qg.g_int.cells['nodes'].T,
            validate=True)

# plot(m,qg.psi,shading='gouraud')
pnts=None
def choose_top(xi):
    global pnts
    pnts=xi.copy()
    print(xi.shape)
    return False

# define_boundary, at least for this element type,
# pass [2,Nedges] midpoints.
m.define_boundary('test',choose_top)

##

plt.figure(10).clf()
qg.g_int.plot_edges()
plt.plot(pnts[0,:],pnts[1,:],'r.')

##

# Test things out on an L-shaped mesh

# The goal here is to have skfem tell us what the
# dirichlet BC for the middle edge should be.

m=MeshTri.init_lshaped()
m.refine(2)

m.define_boundary('top', lambda xi: xi[1] == 1.)
m.define_boundary('mid', lambda xi: (xi[1] == 0.) & (xi[0]>0.0) )
m.define_boundary('bottom', lambda xi: xi[1] == -1)

# These will get a zero normal by some sort of magic.
# I think it's similar to the NodeDiscretization, where
# a zero normal just happens by default
#m.define_boundary('left', lambda xi: xi[0] == 0.)
#m.define_boundary('right', lambda xi: xi[0] == 1.)

e = ElementTriP1()
basis = InteriorBasis(m, e)

##

A = asm(laplace, basis)

# 65x65 sparse matrix
#  m.t.shape => (3,96) for 96 triangular elements
#  m.p.shape => (2,65) for 65 2D nodes.

boundary_dofs = basis.find_dofs()
interior_dofs = basis.complement_dofs(boundary_dofs)
all_boundary_dofs=np.concatenate( (boundary_dofs['top'].all(),
                                   boundary_dofs['bottom'].all(),
                                   boundary_dofs['mid'].all() ) )
# 19 in all_boundary_dofs
# 46 in interior_dofs
# That sums to the 65 nodes
u = np.zeros(basis.N) # number of nodes

u[boundary_dofs['top'].all()] = 7.
u[boundary_dofs['bottom'].all()] = 1.

# So if I guess the right value here, it's fine.
# but what I want is to just say that there is no tangential
# gradient along these facets
u[boundary_dofs['mid'].all()] = np.pi

b=0.*u # rhs

# Could I handle my BCs external to the skfem machinery?
# cond_b above has some dependence on the BC for mid.
# Some entries in cond_b are equal to or half of the BC value.
# A (65x65) is full rank.

# But I could come in and say that any reference to node x should instead
# use node y

# Say there are 5 nodes that are part of the mid BC.
#
Ad=A.todense()

## 
n=46

rows=np.nonzero( Ad[:,n] )[0]
cols=np.nonzero( np.any( Ad[rows,:]!=0.0, axis=0) )[1]
block=np.concatenate( (cols[None,:], Ad[rows,:][:,cols]), axis=0)
block=np.concatenate( ( np.r_[np.nan, rows][:,None], block), axis=1)

print()
print(block)

## 
# A is square, with x and b both in order of nodes.
# So the Laplacian for n=8 is the 8th row.
# What if I omit that row, and say that 8 is equal to

# u_soln = solve(*condense(A, 0.*u, u, interior_dofs))

b=0*u
for n in boundary_dofs['top'].all():
    Ad[n,:]=0
    Ad[n,n]=1
    b[n]=7.

for n in boundary_dofs['bottom'].all():
    Ad[n,:]=0
    Ad[n,n]=1
    b[n]=1.

# This is what I want to relax
# Of the group, all but one dof are used to set the
# equality.  the last dof gets a single-element normal
# BC
tan_grp=boundary_dofs['mid'].all()
for n1,n2 in zip(tan_grp[:-1],tan_grp[1:]):
    Ad[n2,:]=0
    Ad[n2,n2]=1
    Ad[n2,n1]=-1
    b[n2]=0

# I think that what I lose is the no-flux condition around the
# corner.
# Manually evaluate the gradient in [0,22,26]
grad_tri=[0,22,26] # with the zero-flux edge along the first two of these
target_dof=tan_grp[0] # install in this row of the matrix

d01=m.p[:,grad_tri[1]] - m.p[:,grad_tri[0]]
d02=m.p[:,grad_tri[2]] - m.p[:,grad_tri[0]]
# Derivation in sympy below
Ad[target_dof,:]=0 # clear old
Ad[target_dof,grad_tri[0]]= -d01[0]**2 + d01[0]*d02[0] - d01[1]**2 + d01[1]*d02[1]
Ad[target_dof,grad_tri[1]]= -d01[0]*d02[0] - d01[1]*d02[1]
Ad[target_dof,grad_tri[2]]= d01[0]**2 + d01[1]**2
b[target_dof]=0

u_soln=np.linalg.solve(Ad,b)

# This is what I want to encode:
# [dx01 dy01] [ tgt_grad_x ] = c * [ dpsi01 ]
# [dx02 dy02] [ tgt_grad_y ]       [ dpsi02 ]

# And I don't care what c is (well, not 0)
# Instead, write this for the vector perpendicular to the
# normal, and set that to 0.

# Solve this for nx, ny:
# [dx01 dy01] [ nx ] = [ dpsi01 ]
# [dx02 dy02] [ ny ]   [ dpsi02 ]

# det=(dx01*dy02) - (dy01*dx02)

#  from sympy import *
#  dx01,dy01,dx02,dy02 = symbols("dx01 dy01 dx02 dy02")
#  nx,ny=symbols("nx ny")
#  psi0,psi1,psi2 = symbols("psi0 psi1 psi2")
#  tgt_px,tgt_py=symbols('tgt_px tgt_py')
#  
#  eq=Matrix( [[dx01, dy01],[dx02,dy02]] ) * Matrix([[nx],[ny]]) - Matrix([ [psi1-psi0],
#                                                                           [psi2-psi0]])
#  solns=solve(eq,[nx,ny])
#  # => 
#  # nx: (dy01*(psi0 - psi2) - dy02*(psi0 - psi1))/(dx01*dy02 - dx02*dy01),
#  # ny: (-dx01*(psi0 - psi2) + dx02*(psi0 - psi1))/(dx01*dy02 - dx02*dy01)
#  eq2=solns[nx]*tgt_px + solns[ny]*tgt_py
#  # A bit gross:
#  # tgt_px*(dy01*(psi0 - psi2) - dy02*(psi0 - psi1))/(dx01*dy02 - dx02*dy01)
#  #     + tgt_py*(-dx01*(psi0 - psi2) + dx02*(psi0 - psi1))/(dx01*dy02 - dx02*dy01)
#  # 
#  
#  # But I know a bit more.  Can assume that nodes 0 and 1 are along the no-flux
#  # edge.  So px,py is perpendicular to that edge
#  eq3=eq2.subs( tgt_px, -dy01).subs(tgt_py,dx01).expand()
#  # dx01*(dy01*(psi0 - psi2) - dy02*(psi0 - psi1))/(dx01*dy02 - dx02*dy01)
#  # + dy01*(-dx01*(psi0 - psi2) + dx02*(psi0 - psi1))/(dx01*dy02 - dx02*dy01)
#  
#  # Factor out the determinant
#  det_val=(dx01*dy02 - dx02*dy01)
#  det=symbols('det')
#  eq4=eq3.subs(det_val,det)
#  
#  for t in [psi0,psi1,psi2]:
#      # And since this is homogeneous, can actually factor out det
#      print(t,(eq4.coeff(t) * det).simplify())
#  
#  #  psi0:  -dx01**2 + dx01*dx02 - dy01**2 + dy01*dy02
#  #  psi1:  -dx01*dx02 - dy01*dy02
#  #  psi2:  dx01**2 + dy01**2

# This appears to work okay.
# HERE: next step is to think through how it will apply to a more complicated
#   domain (i.e. can I do this at every inside corner?  will it play nice with
#   ragged edges?).
#    Seems that every inside corner will work out.  Ragged edges when joining
#    parallel edges are maybe okay??
#  Not sure what to do about internal edges. For the moment ignoring ragged or
#   or internal edges...
# Then test it on the [10,11] subdomain above.
#  => good.
# And then the [3] subdomain.
# And finally the whole domain.

# Solving for the gradient:
g01=m.p[:,grad_tri[1]] - m.p[:,grad_tri[0]]
g02=m.p[:,grad_tri[2]] - m.p[:,grad_tri[0]]
# HERE - this seems like a good BC.
# just need to translate into a BC
# u_soln[grad_tri[1]] - u_soln[
mat=np.array( [[ g01[0], g01[1] ],
               [ g02[0], g02[1] ]])
btmp=np.array( [u_soln[grad_tri[1]]-u_soln[grad_tri[0]],
             u_soln[grad_tri[2]]-u_soln[grad_tri[0]]] )
u_grad=np.linalg.solve(mat,btmp)

cc=m.p[:,grad_tri].mean(axis=1)

# ax=plot(m,u_soln,shading='gouraud')
fig=plt.figure(12).clf()
fig,ax=plt.subplots(num=12)
plt.triplot(m.p[0],m.p[1],triangles=m.t.T,color='0.5',alpha=0.4)
coll=plt.tripcolor(m.p[0],m.p[1],u_soln,triangles=m.t.T,shading='gouraud',cmap='jet')
plt.tricontour(m.p[0],m.p[1],u_soln,20,triangles=m.t.T,colors='k',linewidths=0.5,alpha=0.5)

ax.quiver( [cc[0]],[cc[1]],[u_grad[0]],[u_grad[1]],scale=30)

                                
plt.colorbar(coll)

ax=plt.gca()
for i in range(m.p.shape[1]):
    ax.text(m.p[0,i],m.p[1,i],f"{i}")
ax.plot(m.p[0],m.p[1],'k.')

## 

plt.figure(1).clf()
g_int_gmsh.plot_edges(color='blue',lw=3.,alpha=0.3)
g_int_gmsh.plot_nodes(color='orange',sizes=10,alpha=0.3)
plt.axis('tight')
plt.axis('equal')

        
## 
# is this any faster?  yes.
# results are worse, though
psi_phi2,*rest=sparse.linalg.lsmr(bigM,rhs)
N=qg.g_int.Nnodes()
psi2=psi_phi2[:N]
phi2=psi_phi2[N:]

##

# Should see if scipy fem can be used instead of my code.


# Is it possible to find the rank of bigM?
from scipy.linalg import interpolative
from scipy import sparse

# Too slow..
bigM_op=sparse.linalg.aslinearoperator(bigM)
est_rank=interpolative.estimate_rank(bigM_op,1e-6)
        
## 
# Adapt quads to having angles on half-edges, and dealing with multiple
# cells at once.

# Later add a convenience routine to copy from nodes.
qg=quads.QuadGen(gen_src,cells=[7,8,9],final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])

qg.execute()
qg.plot_result(num=100)
qg.g_final.write_ugrid('by_angle_output_cell00.nc',overwrite=True)

##

# cell 1
qg=quads.QuadGen(gen_src,cell=1,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=101)

qg.g_final.write_ugrid('by_angle_output_cell01.nc',overwrite=True)

##

# cell 2
qg=quads.QuadGen(gen_src,cell=2,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=102)

qg.g_final.write_ugrid('by_angle_output_cell02.nc',overwrite=True)

##

qg=quads.QuadGen(gen_src,cell=3,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=103)

qg.g_final.write_ugrid('by_angle_output_cell03.nc',overwrite=True)

##

qg=quads.QuadGen(gen_src,cell=4,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result(num=104)

qg.g_final.write_ugrid('by_angle_output_cell04.nc',overwrite=True)

## 

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v05.pkl')

qg=quads.QuadGen(gen_src,cell=5,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])
qg.execute()
qg.plot_result()

#qg.g_final.write_ugrid('by_angle_output_cell05.nc')

##

# Old Butano channel - fails unless nom_res is smaller, and can't have
# the large ragged edge.

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v05.pkl')

# This would benefit from some internal edges, or breaking into separate
# grids.

qg=quads.QuadGen(gen_src,cell=6,final='anisotropic',execute=False,
                 nom_res=3.5,
                 scales=[field.ConstantField(3.5),
                         field.ConstantField(3.5)])
qg.execute()
qg.plot_result()

qg.g_final.write_ugrid('by_angle_output_cell06.nc')


##
g_combined=None

for fn in ["by_angle_output.nc",
           "by_angle_output_cell01.nc",
           "by_angle_output_cell02.nc",
           "by_angle_output_cell03.nc",
           "by_angle_output_cell04.nc",
           "by_angle_output_cell05.nc",
           "by_angle_output_cell06.nc" ]:
    g=unstructured_grid.UnstructuredGrid.read_ugrid(fn)
    if g_combined is None:
        g_combined=g
    else:
        g_combined.add_grid(g)
        
##
plt.figure(1).clf()
g_combined.plot_edges(color='k',lw=0.5)
g_combined.plot_cells(color='0.85',zorder=-2,lw=0)

plt.axis('tight')
plt.axis('equal')
