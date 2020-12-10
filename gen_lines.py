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

import stompy.plot.cmap as scmap

ver='v19'

gen_src=unstructured_grid.UnstructuredGrid.read_pickle(f'grid_lagoon-{ver}.pkl')
g=unstructured_grid.UnstructuredGrid.read_pickle(f'all-quads-{ver}.pkl')

# Add the non-cell edges of gen_src back into g:
gen_src_tri=gen_src.copy()
e2c=gen_src_tri.edge_to_cells(recalc=True)
j_to_delete=np.any(e2c>=0,axis=1)

# These nodes in gen_src_tri will be matched up with nodes in the merged grid
n_to_match=[]
for n in gen_src_tri.valid_node_iter():
    js=gen_src_tri.node_to_edges(n)
    if len(js)<2:
        print("Weird node %d"%n)
        continue
    
    js_status=j_to_delete[js]

    if np.all(js_status):
        pass # Node only participates in edges that are part of cells.
    elif np.any(js_status):
        n_to_match.append(n)
    else:
        pass

for j in np.nonzero(j_to_delete)[0]:
    gen_src_tri.delete_edge_cascade(j)

gen_src_tri.delete_orphan_nodes()

merge_nodes=[]
for nB in n_to_match:
    # Might want to limit this to boundary nodes...
    nA=g.select_nodes_nearest( gen_src_tri.nodes['x'][nB] )
    merge_nodes.append( [nA,nB] )

g.add_grid(gen_src_tri,merge_nodes=merge_nodes)

g.renumber(reorient_edges=False)

g_orig=g

g_orig.write_pickle(f'quads_and_lines-{ver}.pkl',overwrite=True)

## 
plt.figure(1).clf()
g_orig.plot_edges(lw=0.6,color='k')
g_orig.plot_cells(alpha=0.4)
plt.axis('tight')
plt.axis('equal')
