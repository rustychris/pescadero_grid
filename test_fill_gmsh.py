from stompy.grid import unstructured_grid
import numpy as np
import matplotlib.pyplot as plt
## 


g=unstructured_grid.UnstructuredGrid()

theta=np.linspace(0,2*np.pi,500)[:-1]

pnts=np.c_[ 1000*np.cos(theta), 750*np.sin(theta) ]

g.add_linestring(pnts,closed=True)

##

plt.figure(1).clf()
g.plot_edges(lw=0.5)
plt.axis('tight')
plt.axis('equal')

##

# First, how to do telescoping in gmsh
# automaticMeshSizeField would be nice, but not compiled in.
# what about the external field?
from stompy.spatial import field
el=g.edges_length()
ec=g.edges_center()
scale=field.ApolloniusField(X=ec,F=el)
import pickle
scale_file='scale.pkl'
with open(scale_file,'wb') as fp:
    pickle.dump(scale,fp)

##     

fn='tmp.geo'
g.write_gmsh_geo(fn)

with open(fn,'at') as fp:
    fp.write("""
Field[1] = ExternalProcess;
Field[1].CommandLine = "python -m stompy.grid.gmsh_scale_helper %s";

Background Field = 1;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
"""%(scale_file) )

##

import subprocess
subprocess.run(["gmsh",fn,'-2'])
g_gmsh=unstructured_grid.UnstructuredGrid.read_gmsh('tmp.msh')

plt.figure(1).clf()
g_gmsh.plot_edges(lw=0.4)

plt.axis('tight')
plt.axis('equal')


