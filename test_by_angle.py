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

plt.figure(1).clf()
gen_src.plot_cells(labeler='id',centroid=True)
plt.axis('tight')
plt.axis('equal')

##
six.moves.reload_module(field)
from stompy.spatial import wkb2shp, constrained_delaunay
six.moves.reload_module(constrained_delaunay)
CXYZ=constrained_delaunay.ConstrainedXYZField

i_tele=field.ApolloniusField.read_shps(['scale.shp'],value_field='i_tele')
j_tele=field.ApolloniusField.read_shps(['scale.shp'],value_field='j_tele')

i_linear=CXYZ.read_shps(['scale.shp'],value_field='i_linear')
j_linear=CXYZ.read_shps(['scale.shp'],value_field='j_linear')

##

i_scale=field.BinopField( i_tele, np.minimum, i_linear)
j_scale=field.BinopField( j_tele, np.minimum, j_linear)

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

# HERE:
#  1. Matrix solve is dicey in larger domain
#     => try scikit fem
#  2. Intermediate grid gen is slow
#     => add gmsh option for triangulate_hole. DONE.
#  3. Scale specification is error prone:
#     => Use orientation of linestring to set i/j axes

## 
# Try to generate the whole thing at once:

qg=quads.QuadGen(gen_src,final='anisotropic',execute=False,
                 nom_res=3.5,
                 scales=[field.ConstantField(3),
                         field.ConstantField(3)])
qg.execute()
qg.plot_result(num=100)
##

# Taking forever, and psi/phi is not good anyway.
qg.plot_psi_phi_setup() ; plt.axis('tight') ; plt.axis('equal')
ax=plt.gca()
self=qg
for grps in [self.i_tan_groups,self.j_tan_groups]:
    for n,i_grp in enumerate(grps):
        idx=len(i_grp)//2
        ax.text( self.g_int.nodes['x'][i_grp[idx],0],
                 self.g_int.nodes['x'][i_grp[idx],1],str(n))

qg.plot_psi_phi(thinning=0.15) ; plt.axis('tight') ; plt.axis('equal')
qg.g_int.contourf_node_values(psi2,250,cmap='prism')
#qg.g_int.contourf_node_values(qg.phi,250,cmap='prism')

# Shows that psi flips around at the connector channel,

##

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(rebay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(quads)

qg=quads.QuadGen(gen_src,cells=[0],final='anisotropic',execute=False,
                 nom_res=3.5,triangle_method='gmsh',
                 scales=[i_scale,j_scale])

qg.execute()

## 
qg.plot_psi_phi_setup()

# How well does this translate scikit-fem?
## 
m = MeshTri(p=qg.g_int.nodes['x'].T,
            t=qg.g_int.cells['nodes'].T,
            validate=True)

# plot(m,qg.psi,shading='gouraud')

## 
from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import laplace, unit_load
from skfem.visuals.matplotlib import plot

##

e = ElementTriP1()
basis = InteriorBasis(m, e)

A = asm(laplace, basis)
b = 0*asm(unit_load, basis)

# BCs!
boundary_basis = FacetBasis(m, e)
# Seems to be the indices of the boundary nodes, but it could be
# the boundary edges.  Probably the boundary nodes, tho.
# Without the .all(), then we get some nodal_ix stuff & facet_ix.
boundary_dofs = boundary_basis.find_dofs()['all'].all()
# u[boundary_dofs] = project(dirichlet, basis_to=boundary_basis, I=boundary_dofs)

HERE -- need more work on scikit-fem usage.
u = np.zeros(basis.N) # number of nodes

if 0:
    # This is probably too direct.  There is some machinery I'm skipping.
    # dirichlet:
    boundary_dofs=[]
    for n in qg.i_dirichlet_nodes:
        u[n]=qg.i_dirichlet_nodes[n]
        boundary_dofs.append(n)
    boundary_dofs=np.array(boundary_dofs)

u = solve(*condense(A, np.zeros_like(u), u, D=boundary_dofs))

plot(m, u, shading='gouraud', colorbar=True)

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
