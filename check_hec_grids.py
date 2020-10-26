from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import six
six.moves.reload_module(unstructured_grid)

## 

grid_file="../data/waterways/pescaderoassessment2d.g11.hdf"
g=unstructured_grid.UnstructuredGrid.read_ras2d(grid_file)

##


# That gets the geometry, but no elevation data
# Poking around in the hdf file, I don't think there is any elevation data in there.


##
import h5py # import here, since most of stompy does not rely on h5py
h = h5py.File(grid_file, 'r')

##

# 'Geometry'
#  '2D Flow Area Break Lines'
#  '2D Flow Areas'
#      'Attributes'
#      'BOUNDARY'
#         'Cells Center Coordinate'
#         'Cells Face and Orientation Info'  (23528,2) integers
#         'Cells Face and Orientation Values' (114558, 2) i4 topology and orientation
#         'Cells FacePoint Indexes' 
#         'FacePoints Cell Index Values'
#         'FacePoints Cell Info'
#         'FacePoints Coordinate'
#         'FacePoints Face and Orientation Info'
#         'FacePoints Face and Orientation Values'
#         'FacePoints Is Perimeter'
#         'Faces Cell Indexes'
#         'Faces FacePoint Indexes'
#         'Faces NormalUnitVector and Length'
#         'Faces Perimeter Info'
#         'Faces Perimeter Values'
#         'Perimeter'
#      'Cell Points' (22636,2) presumably cell centersa
#      'Polygon Info' [  0, 516,   0,   1]
#      'Polygon Parts'
#      'Polygon Points'

#  'Boundary Condition Lines'
#  "Land Cover (Manning's n)"
#  'Structures'


##

fn='../data/waterways/pescaderoassessment2d.p24.hdf'
g=unstructured_grid.UnstructuredGrid.read_ras2d(fn)

#import h5py # import here, since most of stompy does not rely on h5py
#h = h5py.File(fn, 'r')

##

# Here we go.  this has grid info, but also some elevation info.
# 'Geometry/2D Flow Areas/BOUNDARY/Cells Minimum Elevation'
# ...  'Cells Volume Elevation Info'
# ...  'Cells Volume Elevation Values'
# ...  'Faces Area Elevation Info'
# ...  'Faces Area Elevation Values'
# ...  'Faces Minimum Elevation'

# This has 24571 elements, but the grid has 23678 cells. 893 extra entries??
# h['Geometry/2D Flow Areas/BOUNDARY/Cells Minimum Elevation']
# # ok - there are ghost cells.  Not sure how there are more ghost cells than
# nodes in the perimeter, but carry on...
twod_area_name='BOUNDARY'

Nc=g.Ncells() # trim ghost cells
g.add_cell_field('cell_z_min',
                 h['Geometry/2D Flow Areas/' + twod_area_name + '/Cells Minimum Elevation'][:Nc])
g.add_edge_field('edge_z_min',
                 h['Geometry/2D Flow Areas/' + twod_area_name + '/Faces Minimum Elevation'][:])

##

plt.figure(1).clf()
g.plot_edges(color='k',lw=0.5)
ccoll=g.plot_cells(values='cell_z_min',cmap='jet')
ccoll.set_clim(5,25.0)
plt.colorbar(ccoll)

##

g.write_ugrid('hec-grid.nc',dialect='mdal')
