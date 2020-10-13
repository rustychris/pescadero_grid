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
from stompy.spatial import wkb2shp, constrained_delaunay

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(constrained_delaunay)
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

# v06 puts angles on half-edges
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v07.pkl')

gen_src.renumber_cells()


# How to get scale in there...

# CXYZ=constrained_delaunay.ConstrainedXYZField
# 
# i_tele=field.ApolloniusField.read_shps(['scale.shp'],value_field='i_tele')
# j_tele=field.ApolloniusField.read_shps(['scale.shp'],value_field='j_tele')
# 
# i_linear=CXYZ.read_shps(['scale.shp'],value_field='i_linear')
# j_linear=CXYZ.read_shps(['scale.shp'],value_field='j_linear')
# 
# i_scale=field.BinopField( i_tele, np.minimum, i_linear)
# j_scale=field.BinopField( j_tele, np.minimum, j_linear)


# This way of specifying scale is rough because i/j is not easy to determine,
# and worse it changes when changing the set of cells.
# One possibility is to set scale on edges of gen. Then when constructing
# the swaths, each end would hit those edges, and pick up scale.

# It's a departure from how scale is usually specified for triangles, although it's
# not terrible. Those edges might have triangles on the other side, and the
# scale on the gen side is exactly that scale needed to match up with the triangles.
# Ragged edges are slightly tricky -- they may not work well with triangles -- but
# the specification is probably okay, just taking the angle of the edge to then get
# an i and j scale from the single original scale.

if 1:
    plt.figure(1).clf()
    gen_src.plot_cells(labeler='id',centroid=True)
    plt.axis('tight')
    plt.axis('equal')

six.moves.reload_module(quads)

# Any chance a full domain will work? close..
qg=quads.QuadGen(gen_src,
                 # cells=[0,1,5,6,7,8],
                 # cells=[3],
                 # cells=[0,1,2,4,5,6,7,8,9,10,11],
                 # cells=[10,11],
                 final='anisotropic',execute=False,
                 triangle_method='gmsh',
                 nom_res=3.5)

qg.execute()
qg.plot_result()

##


qg.set_scales()

qg.g_final=qg.create_final_by_patches()

qg.plot_result(num=100+c)

# With the full domain, if include internal edges, I'm over-constrained
# by one degree M.shape: (26548, 26547)
# If I remove all internal edges, then it works.
# But there are multiple internal edges.
# Is it something about the dirichlet BC falling on an internal edge?

##

# Show where scale was specified on gen edges:
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
ax.set_position([0,0,1,1])
ax.axis('off')
scale=qg.gen.edges['scale']
sel=np.isfinite(scale) & (scale>0)
qg.g_final.plot_edges(lw=0.3,alpha=0.3,color='k')

qg.gen.plot_edges(mask=sel,values=scale[sel],cmap='jet',labeler='scale')

from stompy.plot import plot_utils
plot_utils.scalebar([0.1,0.8],L=10,fractions=[0,0.1,0.25,0.5,1.0],
                    dy=0.03,style='ticks',
                     xy_transform=ax.transAxes,ax=ax)


##
# Wondering if something special should be done at the sting?
#  Dropping the no-flux BC at a sting maybe fixed psi, but now
#  I'm one short on phi.
# Best guess is that a sting implies an extra, single-node, tangent
# group (to avoid having a no-flux BC at the sting), and adding
# a gradient BC at the sting for the other field.

#qg.plot_psi_phi_setup()

##

# plt.figure(1).clf()
# scale=qg.gen.edges['scale']
# sel=np.isfinite(scale) & (scale>0)
# qg.g_int.plot_edges(lw=0.3,alpha=0.3,color='k')
# qg.gen.plot_edges(mask=sel,values=scale[sel],cmap='jet')
# 
# qg.g_int.contourf_node_values(j_scale,30,cmap='jet')


## 
# Make sure that each individual cell works:
# c=1 is failing... that's the lagoon.  Maybe working now.
for c in gen_src.valid_cell_iter():
    qg=quads.QuadGen(gen_src,cells=[c],final='anisotropic',execute=False,
                     triangle_method='gmsh',
                     nom_res=3.5,
                     scales=[i_scale,j_scale])
    qg.execute()
    qg.plot_result(num=100+c)

    qg.g_final.write_ugrid('pieces_cell%03d.nc'%c,
                           overwrite=True)
    
g=None
for c in gen_src.valid_cell_iter():
    g_sub=unstructured_grid.UnstructuredGrid.read_ugrid('pieces_cell%03d.nc'%c)
    if g is None:
        g=g_sub
    else:
        g.add_grid(g_sub)
g.write_ugrid('combined-pieces.nc',overwrite=True)        


##

# For a rectangle:
#  each non-corner node on closed boundaries solves
#  the laplacian, along with each internal node.
#  each set of open boundary nodes implies a constant
#  value.
six.moves.reload_module(quads)

qg=quads.QuadGen(gen_src,cells=[3],
                 final='anisotropic',execute=True,
                 nom_res=3.5,triangle_method='gmsh',
                 scales=[i_scale,j_scale])

## 
# HERE:

#  above.
#  1. Matrix solve is dicey in larger domain
#     => try scikit fem.  Tried, but its pretty opaque.
#        Instead, augmented BCs, somewhat ad-hoc, but so far
#        yields square matrix (watch for ragged!), and solve
#        is faster and more robust with spsolve.
#  3. Scale specification is error prone:
#     => Use orientation of linestring to set i/j axes

##


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


