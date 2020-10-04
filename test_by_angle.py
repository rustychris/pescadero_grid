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
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v06.pkl')

gen_src.renumber_cells()

# Did the half-edge angles come in okay?
#plt.figure(2).clf()

#gen_src.plot_edges(color='k',lw=0.5)

#gen_src.plot_halfedges(labeler=lambda j,side: [ gen_src.edges['turn_fwd'][j],
#                                                gen_src.edges['turn_rev'][j] ][side])

#gen_src.plot_cells(labeler='id',centroid=True)


# Adapt quads to having angles on half-edges, and dealing with multiple
# cells at once.

# Later add a convenience routine to copy from nodes.
qg=quads.QuadGen(gen_src,cells=[7,8,9],final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])

# New half-edge code obviates the need for these lines:
# qg.add_internal_edge([23,36])
# qg.add_internal_edge([20,32])

@utils.add_to(qg)
def prepare_angles_halfedge(self):
    """
    Move turn angles from half edges to absolute angles of edges.
    Internal edges will get an angle, then be removed from
    gen and recorded instead in self.internal_edges
    """
    # Might get smarter in the future, but for now we save some internal
    # edge info, sum turns to nodes, have prepare_angles_nodes() do its
    # thing, then return to complete the internal edge info

    gen=self.gen
    e2c=gen.edge_to_cells()

    internals=[]
    
    gen.add_node_field('turn',np.nan*np.zeros(gen.Nnodes(),np.float32), on_exists='overwrite')

    valid_fwd=(e2c[:,0]>=0) & np.isfinite(gen.edges['turn_fwd']) & (gen.edges['turn_fwd']!=0)
    valid_rev=(e2c[:,1]>=0) & np.isfinite(gen.edges['turn_rev']) & (gen.edges['turn_rev']!=0)

    # iterate over nodes, so that edges can use default values 
    fixed_nodes=np.unique( np.concatenate( (gen.edges['nodes'][valid_fwd,1],
                                            gen.edges['nodes'][valid_rev,0]) ) )

    j_int={}
    for n in fixed_nodes:
        turn=0
        nbrs=gen.node_to_nodes(n)
        he=he0=gen.nodes_to_halfedge(nbrs[0],n)
        # Start on the CCW-most external edge:
        while he.cell_opp()>=0:
            he=he.fwd().opposite()
            assert he!=he0
        he0=he
        
        while 1:
            if he.cell()>=0:
                if he.orient==0:
                    sub_turn=gen.edges['turn_fwd'][he.j]
                else:
                    sub_turn=gen.edges['turn_rev'][he.j]
                if np.isnan(sub_turn): sub_turn=180
                elif sub_turn==0.0: sub_turn=180
                turn+=sub_turn
            else:
                sub_turn=np.nan
                
            if (e2c[he.j,0]>=0) and (e2c[he.j,1]>=0):
                if he.j not in j_int:
                    j_int[he.j]=1
                    print(f"Adding j={he.j} as an internal edge")
                    internals.append( dict(j=he.j,
                                           nodes=[he.node_fwd(),he.node_rev()],
                                           turn=turn,
                                           j0=he0.j) )

            he=he.fwd().opposite()
            if he==he0: break
        gen.nodes['turn'][n]=turn
        
    # Come back for handling of internal edges
    for internal in internals:
        print("Internal edge: ",internal['nodes'])
        gen.merge_cells(j=internal['j'])

    self.prepare_angles_nodes()

    for internal in internals:
        self.add_internal_edge(internal['nodes'],
                               gen.edges['angle'][internal['j0']]+internal['turn'])


qg.prepare_angles()
qg.add_bezier(qg.gen)
qg.plot_gen_bezier() # HERE -- one of those edges is wrong.
qg.gen.plot_edges(labeler='angle')

##
        
qg.execute()
##

## 
qg.plot_result(num=100)

## 
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
