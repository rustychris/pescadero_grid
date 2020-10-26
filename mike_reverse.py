import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from stompy import utils
##


fn_bathy="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/Butano_Bathy_Scenario_Post_Project_grid_dub_16_i1.dfs2"
#fn="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/Butano_Bathy_Scenario_Pre_Project_grid_dub_16_i1.dfs2"
fn_rough="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/grid_dub_16_Post_Projects_roughness_w_ELJ.dfs2"
fn_iwse="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/IWSE_from_HS_Post_Project_IWSE_Stabilization.dfs2"
fn_grid="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/grid_dub_16.dfs2"

## 


##     
# 3.4MB
def dhi_read_string(buff):
    length,=struct.unpack('I',buff[:4])
    data=buff[4:4+length]
    assert data[-1]==0
    data=data[:-1]
    return data,buff[4+length:]

def read_dfs2(fn,
              label_pos=0x1c7, # 0x1c2 for some files, 0x1c7 for others
              dtype=np.float32): # grid has 2 float64, 2 float32
    ds=xr.Dataset()
    
    with open(fn_bathy,'rb') as fp:
        data=fp.read()
    
    # header, probably up to the ^Z, ascii 26
    head_end=data.index(26)
    head_txt=data[:head_end]
    rest=data[head_end+1:]

    # Timestamps look like year month date hour minute second "104 0 206 0"
    timestamp0=np.frombuffer(rest[:20],np.int16)
    rest=rest[20:]
    print(f"Timestamp 0: {timestamp0}")
    ds['timestamp0']=('ts_parts',),timestamp0

    timestamp1=np.frombuffer(rest[:20],np.int16)
    rest=rest[20:]
    print(f"Timestamp 1: {timestamp1}")
    ds['timestamp1']=('ts_parts',),timestamp1

    dat00=rest[:27]
    rest=rest[27:]

    name00,rest=dhi_read_string(rest)

    # Data2 (bathy), Data3 (roughness)
    print(f"Name 0: {name00}")
    ds['name']=(),name00

    dat01=rest[:5]
    rest=rest[5:]
    data_creator,rest=dhi_read_string(rest)
    print(f"Data creator: {data_creator}")
    ds['creator']=(),data_creator

    # There are also strings in there "UTM-30"
    # A date in ascii: 2002-01-01 00:00:00
    # "grid_dub_16_bathy_edit2"
    # And M21_Misc

    # In bath and roughness, this is where the file label
    # sits.
    # In IWSE, its 1c2.
    # sed files seem to be 1c7

    file_label,rest=dhi_read_string(data[label_pos:])
    print(f"File label: {file_label}")
    ds['file_label']=(),file_label

    ncol,nrow=struct.unpack('II',rest[107:107+8])

    # # bathy file:
    # ncol=struct.unpack('I',rest[410:414])
    # nrow=struct.unpack('I',rest[414:418])
    # # roughness file:
    # ncol=struct.unpack('I',rest[408:412])
    # anrow=struct.unpack('I',rest[412:416])

    print(f"{nrow} x {ncol}")

    #print(np.frombuffer(rest[3:43],np.int32))
    meta_stop=rest.find(b"M21_Misc") + 56
    # or...
    # meta_stop=145+56
    # just to get to the first 'z' 'D' 0 0
    # 

    # Skip to the good stuff.
    # clearly won't work for file with multiple fields
    raw=rest[meta_stop:]

    # raw is 3429576 bytes.
    # datatype is 4 bytes
    # 'z' 'D' 0 0   as int32 is 17530, as float32 is 2.4e-41

    nums=np.frombuffer(raw,dtype)

    # 326 rows => b'F\x01\x00\x00'
    # 2630 cols => b'F\n\x00\x00'

    #nrow=326
    #ncol=2630

    mat=nums.reshape( (nrow,ncol) )

    ds['data']=('row','col'), mat
    return ds

bathy_ds=read_dfs2(fn_bathy,
                   label_pos=0x1c7,
                   dtype=np.float32)

iwse_ds=read_dfs2(fn_iwse,
                  label_pos=0x1c7,
                  dtype=np.float32)
## 
plt.figure(1).clf()
ax=plt.gca()
ax.set_position([0,0,1,1])
img=plt.imshow(mat,cmap='jet',origin='bottom')
img.set_clim([0,30])
# plt.colorbar(img)
plt.axis('tight')
plt.axis('equal')
#plt.draw()
ax.text(0.1,0.9,f"{nrow} x {ncol}",transform=ax.transAxes)
#plt.pause(0.15)

##

# Okay..
# But the grid itself is in grid_dub_16.dfs2
# and that appears to have 6 separate fields of similar size?

# Probably the thing to do is manually read the grid geometry,
# put together a shapefile with the bathy and roughness, then
# not worry about making this general.

with open(fn_grid,'rb') as fp:
    data=fp.read()


##

# Read first field of the grid file:
# from hexl, looks like it starts around 0x378
# assume similar rows,cols as above
f1_byte_start=0x371
nrows=327 # nodal values, so 1 more than cell-centered
ncols=2631 # 
dtype=np.float64

f1_nbytes=dtype(1).nbytes*nrows*ncols

f1_data=np.frombuffer(data[f1_byte_start:f1_byte_start+f1_nbytes],
                      np.float64).reshape( (nrows,ncols) )

easting=field1_data

if 0:
    print(f1_data[:10,:10])

    plt.figure(1).clf()
    ax=plt.gca()
    ax.set_position([0,0,1,1])
    img=plt.imshow(f1_data,cmap='jet',origin='bottom')

    plt.colorbar(img)
    plt.axis('tight')
    plt.axis('equal')
    ax.text(0.1,0.9,f"{nrow} x {ncol}",transform=ax.transAxes)

    
##
f2_byte_start=f1_byte_start+f1_nbytes+165
dtype=np.float64
f2_nbytes=dtype(1).nbytes*nrows*ncols
f2_data=np.frombuffer(data[f2_byte_start:f2_byte_start+f2_nbytes],
                      np.float64).reshape( (nrows,ncols) )
northing=f2_data

print(f2_data)
##
if 0:
    print(f2_data[:10,:10])

    plt.figure(1).clf()
    ax=plt.gca()
    ax.set_position([0,0,1,1])
    img=plt.imshow(f2_data,cmap='jet',origin='bottom')

    plt.colorbar(img)
    plt.axis('tight')
    plt.axis('equal')
    ax.text(0.1,0.9,f"{nrow} x {ncol}",transform=ax.transAxes)

##

# So what is the 3rd field? Might be two 4-byte fields.
# metric terms?
f3_byte_start=f2_byte_start+f2_nbytes+14

dtype=np.float32
f3_nbytes=dtype(1).nbytes*nrows*ncols
f3_data=np.frombuffer(data[f3_byte_start:f3_byte_start+f3_nbytes],
                      dtype).reshape( (nrows,ncols) )

print(f3_data)
##
if 1:
    plt.figure(1).clf()
    ax=plt.gca()
    ax.set_position([0,0,1,1])
    img=plt.imshow(f3_data,cmap='jet',origin='bottom')

    plt.colorbar(img)
    plt.axis('tight')
    plt.axis('equal')
    ax.text(0.1,0.9,f"{nrow} x {ncol}",transform=ax.transAxes)

# Mostly varies E-W.
# typ value: 1835834.5
# ranges +-4k.

##     
f4_byte_start=f3_byte_start+f3_nbytes+5

dtype=np.float32
f4_nbytes=dtype(1).nbytes*nrows*ncols
f4_data=np.frombuffer(data[f4_byte_start:f4_byte_start+f4_nbytes],
                      dtype).reshape( (nrows,ncols) )

print(f4_data)
# typ value: 580897.06

##

if 1:
    plt.figure(1).clf()
    ax=plt.gca()
    ax.set_position([0,0,1,1])
    img=plt.imshow(f4_data,cmap='jet',origin='bottom')

    plt.colorbar(img)
    plt.axis('tight')
    plt.axis('equal')
    ax.text(0.1,0.9,f"{nrow} x {ncol}",transform=ax.transAxes)

##
import six
from stompy.grid import unstructured_grid
six.moves.reload_module(unstructured_grid)

g=unstructured_grid.UnstructuredGrid()

# cells not filled in, just nodes at this point
result=g.add_rectilinear([0,0],[nrows-1,ncols-1],nrows,ncols)

##
g.nodes['x'][result['nodes'],0]=easting
g.nodes['x'][result['nodes'],1]=northing

# What does it look like with the other two fields?
# Identical, just less precision

# pretty chunky grid:
# 327 x 2631
#  860k nodes

plt.figure(1).clf()
g.plot_edges()
plt.axis('tight')
plt.axis('equal')

# amazingly, that plots okay.
##

zoom=(1830579.4165077808, 1832262.865387332, 585920.0917458754, 586731.0169555763)

plt.figure(1).clf()
ccoll=g.plot_cells(values=bathy.ravel(),cmap='jet')
ccoll.set_clim([0,20])
plt.axis('tight')
plt.axis('equal')

##

g.add_cell_field('cell_z_bed',bathy_ds.data.values.ravel(),on_exists='replace')
g.add_cell_field('cell_iwse',iwse_ds.data.values.ravel(),on_exists='replace')

##

# z_bed appears to be in meters, based on gsf file
# map projection is claimed as UTM-30
# easting_northing_in_thousands is true.  what does that mean?
g.write_ugrid('mike_grid-v00.nc',dialect='mdal',overwrite=True)
##
# Slow!
g.write_cells_shp('mike21_grid.shp')


##

from stompy.spatial import proj_utils

# It seems that the whole project was done in UTM zone 30, even though that's
# nowhere close.  So what is the real projection?
src_srs='PROJCS["UTM-30",GEOGCS["Unused",DATUM["UTM Projections",SPHEROID["WGS 1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",-3],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0],UNIT["Meter",1]]'



print( proj_utils.mapper(src_srs,'EPSG:26910')( g.nodes['x'][0] ) )
## 

# This is roughly centerline of hwy 1, south bridge abuttment
xy_bridge=[1830457.758391805, 586723.3394563532]
utm_bridge=[552172, 4124523]

test=['EPSG:26910']

best_dist=np.inf
best_srs=None

for srs in ["EPSG:%d"%code for code in range(30000)]:
    try:
        trans=proj_utils.mapper(srs,'EPSG:26910')(xy_bridge)
    except NotImplementedError:
        continue
    
    dist=utils.dist(trans,utm_bridge)
    print(f"Best: {best_dist}  {srs}:   {dist}")
    if dist<best_dist:
        best_dist=dist
        best_srs=srs
        
# EPSG:2768 was surprisingly close.

