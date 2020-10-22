import numpy as np
import matplotlib.pyplot as plt
##


fn_bathy="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/Butano_Bathy_Scenario_Post_Project_grid_dub_16_i1.dfs2"
#fn="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/Butano_Bathy_Scenario_Pre_Project_grid_dub_16_i1.dfs2"
fn_rough="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/grid_dub_16_Post_Projects_roughness_w_ELJ.dfs2"
fn_iwse="/home/rusty/src/pescadero/data/OneDrive/Data and References/Marsh Topo and Model Surface Data/cbec_Mike21_Model/MIKE_21C_Models/Inputs/IWSE_from_HS_Post_Project_IWSE_Stabilization.dfs2"

## 

with open(fn_bathy,'rb') as fp:
    data_bathy=fp.read(512)

with open(fn_rough,'rb') as fp:
    data_rough=fp.read(512)

##

# The first 470 bytes are basically the same other than a few characters different.
delta=np.frombuffer(data_bathy,np.uint8) - np.frombuffer(data_rough,np.uint8)
# 470 in hex is ... 0x01d7
# So the difference in length comes from the label of the file.
# roughness: grid_dub_16_roughness
# bathy: grid_dub_16_bathy_edit2
# can I just check the label at offset 0x1c7?


##
with open(fn_iwse,'rb') as fp:
    data=fp.read()

# 3.4MB

# header, probably up to the ^Z, ascii 26
head_end=data.index(26)
head_txt=data[:head_end]
rest=data[head_end+1:]

# Timestamps look like year month date hour minute second "104 0 206 0"
timestamp0=np.frombuffer(rest[:20],np.int16)
rest=rest[20:]
print(f"Timestamp 0: {timestamp0}")

timestamp1=np.frombuffer(rest[:20],np.int16)
rest=rest[20:]
print(f"Timestamp 1: {timestamp1}")

# [    0     0     0     0  4350  -217  5374  1575     1     0     1  -257
#  10001  1539     0 17408 29793 12897  -256  4862]

dat00=rest[:27]
rest=rest[27:]

def dhi_read_string(buff):
    length,=struct.unpack('I',buff[:4])
    data=buff[4:4+length]
    assert data[-1]==0
    data=data[:-1]
    return data,buff[4+length:]

name00,rest=dhi_read_string(rest)

# Data2 (bathy), Data3 (roughness)
print(f"Name 0: {name00}")

dat01=rest[:5]
rest=rest[5:]
data_creator,rest=dhi_read_string(rest)
print(f"Data creator: {data_creator}") 

# There are also strings in there "UTM-30"
# A date in ascii: 2002-01-01 00:00:00
# "grid_dub_16_bathy_edit2"
# And M21_Misc

# In bath and roughness, this is where the file label
# sits.
# In IWSE, its 1c2.
# sed files seem to be 1c7
file_label,rest=dhi_read_string(data[0x1c2:])
print(f"File label: {file_label}")

np.frombuffer(data[0x1c7:0x1c7+4],np.int32)

ncol,nrow=struct.unpack('II',rest[107:107+8])

# # bathy file:
# ncol=struct.unpack('I',rest[410:414])
# nrow=struct.unpack('I',rest[414:418])
# # roughness file:
# ncol=struct.unpack('I',rest[408:412])
# anrow=struct.unpack('I',rest[412:416])


print(f"{nrows} x {ncols}")

#print(np.frombuffer(rest[3:43],np.int32))
meta_stop=rest.find(b"M21_Misc") + 56
# or...
# meta_stop=145+56
# just to get to the first 'z' 'D' 0 0
# 

# Skip to the good stuff.
raw=rest[meta_stop:]

# raw is 3429576 bytes.
# datatype is 4 bytes
# 'z' 'D' 0 0   as int32 is 17530, as float32 is 2.4e-41

nums=np.frombuffer(raw,np.float32)

# 326 rows => b'F\x01\x00\x00'
# 2630 cols => b'F\n\x00\x00'

#nrow=326
#ncol=2630

#ncol=int( len(nums)/nrow)
#mat=nums[:nrow*ncol].reshape( (nrow,ncol) )
mat=nums.reshape( (nrow,ncol) )

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

