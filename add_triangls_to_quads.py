from stompy.grid import unstructured_grid, triangulate_hole
import matplotlib.pyplot as plt
from stompy import utils

import six
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(triangulate_hole)

##

g=unstructured_grid.UnstructuredGrid.read_pickle('all_quads-edit06.pkl')
g.edge_to_cells(recalc=True,on_missing='delete')
g.orient_cells()
g.renumber(reorient_edges=False)
g.build_node_to_cells()
g.merge_duplicate_nodes()

g.write_pickle('all_quads-edit07.pkl',overwrite=True)

##

# pnt=[552450.4846184037, 4125089.577276607]
pnt=[552506.4621021512, 4124016.750187299]

g_new=triangulate_hole.triangulate_hole(g,seed_point=pnt,method='gmsh')
at=g_new

# That fails with 'GridException: Edge 34754 has cell neighbors'

# g.edges['cells'][34754] =>
#     array([19432,    -1], dtype=int32)

##
zoom=(552482.0699463683, 552494.0122478812, 4123959.2770656957, 4123974.5366424527)

plt.figure(1).clf()
#at.grid.plot_edges(lw=0.4,color='k')
g.plot_edges(lw=0.7,color='k',alpha=0.5)

g.plot_cells(clip=zoom,labeler='id',alpha=0.2,centroid=True)
g.plot_edges(clip=zoom,labeler='id',label_jitter=0.5)
g.plot_nodes(clip=zoom,labeler='id',label_jitter=0.5)
plt.axis(zoom)
