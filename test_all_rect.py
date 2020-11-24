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
from stompy.spatial import linestring_utils

turbo=scmap.load_gradient('turbo.cpt')
cmap=scmap.load_gradient('oc-sst.cpt')

##

from stompy.grid import triangulate_hole, orthogonalize
from stompy.spatial import wkb2shp, constrained_delaunay

six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(constrained_delaunay)
six.moves.reload_module(triangulate_hole)
six.moves.reload_module(orthogonalize)
six.moves.reload_module(quads)

# 13 is good.
# 14 is playing with non-90 angles
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v15.pkl')

if 1:
    plt.figure(1).clf()
    gen_src.plot_cells(labeler='id',centroid=True)
    plt.axis('tight')
    plt.axis('equal')

##

# Snap angles -- back to a world of only 90 degree increments

# Naive approach:
#  In each cell, iterate over half-edges, filling in angles as we go.
#  Slightly less naive: if each cell is a simple quad, then just choose the
#    four smallest angles for the corners, everybody else is straight.

# snap_angles(gen_src)
quads.prepare_angles_halfedge(gen_src)

gen_src.plot_edges(mask=np.isfinite(gen_src.edges['angle']),
                   color='r',lw=2)
quads.add_bezier(gen_src)
quads.plot_gen_bezier(gen_src)

##

grids=[]

# Option:
#    Allow specifying edge count (maybe as negative scale).
#      Edge counts are automatically converted to scales when
#      building the interpolated scale.
#      Scales on an edge will be locally evaluated, i.e. not
#      subject to interpolation.
#    Swaths scan the edges perpendicular to the swath
#      if an edge has a count, contour values for the span
#      of that edge will be generated according to the count.
#    So I have a swath, with its constituent patches.
# 

# each cell has to be a rectangle.
# Edges can have a target resolution, a target number
# of cells, or nothing.
# Nodes are positioned globally along the edges in the generating
# grid.  Counts must be consistent within each rectangle, and that's
# where the real challenge is.

# Some edges in gen can be fused (when a specific count is not
# given, but set from scale).
# Then all edges have a target number of nodes, and some measure
# of how tightly that is specified.
# Each cell then implies two equations, forcing the sum of
# i-steps and j-steps to both be zero.
# Edges with two adjacent cells will appear in 2 equations.
# There is probably an assumption somewhere in here that the
# cells are convex.  I'm okay with that.

# For unknowns, what if each cell gets two scalar unknowns,
# one corresponding to a stretching of +i vs -i edges, the
# other for +j/-j edges.

# Then each edge gets a target count and some scaling of the adjustments
# (possibly from two adjustment factors).

# Forget it -- just require that any edge shared by two cells has an exact
# count, and that all cells must be internally consistent.

# Need a method that will look at a single cell, and the data on its
# edges, and distribute nodes on the perimeter.

c=21 # lagoon

plt.figure(1).clf()
gen_src.plot_cells(mask=[c],alpha=0.3)
plt.axis('tight')
plt.axis('equal')

js=gen_src.cell_to_edges(c)
gen_src.plot_edges(mask=js,labeler=lambda i,r: r['scale'])

# Still not there..
# There is a disconnect between how I'm tryin to treat the "short" ends,
# where I want to enforce the number of edges in specific segments, but then
# on the opposite side I want it to be evenly distributed. These won't match up
# in psi/phi, but I'm assuming that over the long dimension, it's not too far out
# from orthogonal. But then the long edges, I will definitely get into trouble
# if I force the spacing independently on the two opposite sides. That was the
# downfall of the earliest attempts at this.

# [A] Forget all of this, go back to directly constructing quads and then relaxing
#     to orthogonal.
# [B] Create an optimization problem for the location of nodes.  Nodes on shared edges
#     must be set directly.  Nodes on unshared edges can be moved around to satisfy
#     some mixture of a density field and psi/phi contours.
# [C] Just go back to how I was doing it, and be more careful about ragged edges, and
#     deal with the annoyance of global resolution dependence.
# [D] Scrap the whole, and use Janet.

# I have to relax the grid regardless, so what I want here is the shortest path to
# getting a quad grid that has the right discrete properties and is close enough
# to orthogonal that relaxation will converge.

# What about something more along the lines of generating quad patches, set some
# subset of nodes as fixed, and optimize the rest?
# That tosses out tons of the complexity.
# For the moment, I take the same sort of input

## 
qg=quads.QuadGen(gen_src,
                 cells=[c],
                 execute=False,
                 angle_source='existing',
                 triangle_method='gmsh',
                 nom_res=3.5)

# Alternate execution -- don't do the patch processing.
# That's the part I want to tweak
qg.process_internal_edges(qg.gen) # N.B. this flips angles
qg.g_int=qg.create_intermediate_grid_tri()
qg.calc_psi_phi()

##

# Here -- I've got a nice psi/phi field.
# I have some edges with negative scale.  Other edges
# will get just the ambient scale.

# Solve for the count of nodes
# First get the full perimeter at 10 point per bezier segment
# Group edges by angle.
# Within each group, consecutive edges with non-negative scale
# are treated as a unit.
# af is an asymmetry factor.
# It starts at 0.  We go through the grouped edges, count up
# the number of nodes, and see if opposite edges agree.  If they
# don't agree, af is adjusted.

self=qg

def he_angle(he):
    # return (he.grid.edges['angle'][he.j] + 180*he.orient)%360.0
    # Since this is being used after the internal edges handling,
    # angles are oriented to the cycle of the cell, not the natural
    # edge orientation.
    return he.grid.edges['angle'][he.j]

# Start at a corner
he=self.gen.cell_to_halfedge(0,0)
while 1:
    he_fwd=he.fwd()
    corner= he_angle(he) != he_angle(he_fwd)
    he=he_fwd
    if corner:
        break

he0=he
idx=0 # current location into list of perimeter samples
perimeter=[]
node_to_idx={}
angle_to_segments={0:[],
                   90:[],
                   180:[],
                   270:[]}

last_fixed_node=he.node_rev()

while 1:
    pnts=self.gen_bezier_linestring(he.j,span_fixed=False)
    if he.orient:
        pnts=pnts[::-1]
    perimeter.append(pnts[:-1])
    node_to_idx[he.node_rev()]=idx
    idx+=len(pnts)-1
    he_fwd=he.fwd()
    angle=he_angle(he)
    angle_fwd=he_angle(he_fwd)
    if  ( (angle!=angle_fwd) # a corner
          or (self.gen.edges['scale'][he.j]<0)
          or (self.gen.edges['scale'][he_fwd.j]<0) ):
        if self.gen.edges['scale'][he.j]<0:
            count=-int( self.gen.edges['scale'][he.j] )
        else:
            count=0
        angle_to_segments[angle].append( [last_fixed_node,he.node_fwd(),count] )
        last_fixed_node=he.node_fwd()
    
    he=he_fwd
    if he==he0:
        break
perimeter=np.concatenate(perimeter)


plt.cla()
self.gen.plot_edges()
plt.plot(perimeter[:,0],perimeter[:,1],'g-')
plt.axis('tight')
plt.axis('equal')

for n in node_to_idx:
    idx=node_to_idx[n]
    plt.plot( [perimeter[idx,0]],
              [perimeter[idx,1]],
              'ro')

self.gen.plot_nodes(labeler='id')
self.gen.plot_edges(labeler='angle')

def discretize_string(string,density):
    """
    string: a node string with counts, 
       [ (start node, end node, count), ... ]
       where a count of 0 means use the provided density
    density: a density (scale) field
    returns: (N,2) discretized linestring and (N,) bool array of
     rigid-ness.
    """
    result=[]
    rigids=[]
    
    for a,b,count in string:
        if count==0:
            idx_a=node_to_idx[a]
            idx_b=node_to_idx[b]
            if idx_a<idx_b:
                pnts=perimeter[idx_a:idx_b+1]
            else:
                pnts=np.concatenate( [perimeter[idx_a:],
                                      perimeter[:idx_b+1]] )
            assert len(pnts)>0
            seg=linestring_utils.resample_linearring(pnts,density,closed_ring=0)
            rigid=np.zeros(len(seg),np.bool8)
            rigid[0]=rigid[-1]=True
        else:
            pnt_a=self.gen.nodes['x'][a]
            pnt_b=self.gen.nodes['x'][b]
            seg=np.c_[ np.linspace(pnt_a[0], pnt_b[0], count+1),
                       np.linspace(pnt_a[1], pnt_b[1], count+1) ]
            rigid=np.ones(len(seg),np.bool8)
        result.append(seg[:-1])
        rigids.append(rigid[:-1])
    result.append( seg[-1:] )
    rigids.append( rigid[-1:] )
    result=np.concatenate(result)
    rigids=np.concatenate(rigids)
    return result,rigids

def calculate_coord_count(left,right,density):
    # 0,180:
    # positive af makes c1 larger
    af_low=-5
    af_high=5
    while 1:
        af=(af_low+af_high)/2
        assert abs(af)<4.9
        pnts0,rigid0=discretize_string(left,(2**af)*density)
        pnts1,rigid1=discretize_string(right,(0.5**af)*density)
        c0=len(pnts0)
        c1=len(pnts1)
        if c0==c1:
            break
        if c0>c1: #  af should be larger
            af_low=af
            continue
        if c0<c1:
            af_high=af
            continue
    return (pnts0,rigid0),(pnts1[::-1],rigid1[::-1])

(left_i,left_i_rigid),(right_i,right_i_rigid)=calculate_coord_count(angle_to_segments[0],
                                                                    angle_to_segments[180],
                                                                    qg.scales[0])
(left_j,left_j_rigid),(right_j,right_j_rigid)=calculate_coord_count(angle_to_segments[90],
                                                                    angle_to_segments[270],
                                                                    qg.scales[1])

# Necessary to have order match grid order below
left_i=left_i[::-1]
left_i_rigid=left_i_rigid[::-1]
right_i=right_i[::-1]
right_i_rigid=right_i_rigid[::-1]

# option 3: new patch code. Deterministically place the boundary nodes, find their
#  psi/phi coords, and linearly interpolate in psi/phi space.  I think this is the 
#  way to go.  We get nicely behaved curves in the interior and boundary.  Should
#  avoid any self-intersections.  And most of it is quite fast. Slow step is mapping
#  back from psi/phi to x/y, though in this simplified domain, that can be done quickly
#  by going back to how I did it in the past.

for s,r in [ (left_i,left_i_rigid),
             (right_i,right_i_rigid),
             (left_j,left_j_rigid),
             (right_j,right_j_rigid) ]:
    plt.plot( s[~r,0], s[~r,1], 'mo')
    plt.plot( s[r,0], s[r,1], 'ko')
    
# good.

def patch_contours(g_int,node_field,scale,count=None):
    """
    Given g_int, a node field (psi/phi) defined on g_int, a scale field, and 
    a count of edges, return the contour values of the node field which
    would best approximate the requested scale.
    
    g_int: UnstructuredGrid
    node_field: a psi or phi field defined on the nodes of g_int
    scale: length scale Field with domain include g_int
    count: if specified, the number of nodes in the resulting discretization

    returns the contour values (one more than the number of edges)
    """
    field_min=node_field.min()
    field_max=node_field.max()

    # original swath code had to pull out a subset of node in the node
    # field, but now we can assume that g_int is congruent to the target
    # patch.
    swath_nodes=np.arange(g_int.Nnodes())
    swath_vals=node_field[swath_nodes]

    # preprocessing for contour placement
    nd=quads.NodeDiscretization(g_int)
    Mdx,Bdx=nd.construct_matrix(op='dx') # could be saved between calls.
    Mdy,Bdy=nd.construct_matrix(op='dy') # 
    field_dx=Mdx.dot(node_field)
    field_dy=Mdy.dot(node_field)
    field_grad=np.sqrt( field_dx**2 + field_dy**2 )
    swath_grad=field_grad

    order=np.argsort(swath_vals)
    o_vals=swath_vals[order]
    o_dval_ds=swath_grad[order]
    o_ds_dval=1./o_dval_ds

    # trapezoid rule integration
    d_vals=np.diff(o_vals)
    # Particularly near the ends there are a lot of
    # duplicate swath_vals.
    # Try a bit of lowpass to even things out.
    if 1:
        winsize=int(len(o_vals)/5)
        if winsize>1:
            o_ds_dval=filters.lowpass_fir(o_ds_dval,winsize)

    s=np.cumsum(d_vals*0.5*(o_ds_dval[:-1]+o_ds_dval[1:]))
    s=np.r_[0,s]

    # calculate this from resolution
    # might have i/j swapped.  range of s is 77m, and field
    # is 1. to 1.08.  better now..
    local_scale=scale( g_int.nodes['x'][swath_nodes] ).mean(axis=0)
    if count is None:
        n_swath_cells=int(np.round( (s.max() - s.min())/local_scale))
        n_swath_cells=max(1,n_swath_cells)
    else:
        n_swath_cells=count-1

    s_contours=np.linspace(s[0],s[-1],1+n_swath_cells)
    adj_contours=np.interp( s_contours,
                            s,o_vals)
    
    adj_contours[0]=field_min
    adj_contours[-1]=field_max

    assert np.all(np.diff(adj_contours)>0),"should be monotonic, right?"

    return adj_contours


i_contours=patch_contours(qg.g_int,qg.phi,qg.scales[0], len(left_i))
qg.g_int.contour_node_values(qg.phi,i_contours,colors='orange')
j_contours=patch_contours(qg.g_int,qg.psi,qg.scales[1], len(left_j))
qg.g_int.contour_node_values(qg.psi,j_contours,colors='red')

# Now I have the target psi/phi contours
# I know which nodes should be rigid, and their locations.

patch=unstructured_grid.UnstructuredGrid(max_sides=4,
                                         extra_node_fields=[('rigid',np.bool8),
                                                            ('pp',np.float64,2)],
                                         extra_edge_fields=[('orient',np.float32)])
elts=patch.add_rectilinear( [0,0],
                            [len(left_i)-1, len(left_j)-1],
                            len(left_i), len(left_j) )
# Fill in orientation
segs=patch.nodes['x'][ patch.edges['nodes'] ]
deltas=segs[:,1,:] - segs[:,0,:]

patch.edges['orient'][deltas[:,0]==0]=90 # should be consistent with gen.edges['angle']
patch.edges['orient'][deltas[:,1]==0]=0

## 

# Mappings from X=>PP
# This gets a little rough.  Queries on the border of mp_tri
# return twitchy values.
# the trifinder does return -1 in these cases.  So what does
# LinearTriInterpolator do in that case?

from matplotlib.tri import LinearTriInterpolator,TriFinder,TrapezoidMapTriFinder
class PermissiveFinder(TrapezoidMapTriFinder):
    def __init__(self,grid):
        self.grid=grid
        mp_tri=grid.mpl_triangulation()
        super(PermissiveFinder,self).__init__(mp_tri)
    def __call__(self, x, y):
        base=super(PermissiveFinder,self).__call__(x,y)
        missing=np.nonzero(base==-1)[0]
        for i in missing:
            base[i]=qg.g_int.select_cells_nearest( [x[i],y[i]] )
        return base
        
finder=PermissiveFinder(qg.g_int)
psi_interp=LinearTriInterpolator(mp_tri,self.psi,finder)
phi_interp=LinearTriInterpolator(mp_tri,self.phi,finder)
psi_field=lambda x: psi_interp(x[...,0],x[...,1]).filled(np.nan)
phi_field=lambda x: phi_interp(x[...,0],x[...,1]).filled(np.nan)

# psi_field=field.XYZField(X=self.g_int.nodes['x'],F=self.psi)
# psi_field._tri = mp_tri
# phi_field=field.XYZField(X=self.g_int.nodes['x'],F=self.phi)
# phi_field._tri = mp_tri

## 

#  look up psi/phi for all boundary nodes.

# Now that we operate on smaller areas, the mapping is 1:1
# Am I going to run into out-of-domain problems?
# so far no nan, but there are some places wher it seems
# that it has left the domain and is getting some wacky results.
# could force the known coordinates
# Depending on what sort of relaxing I do, this may not matter.

# left_i_pp =np.c_[ psi_field(left_i), phi_field(left_i)]
# right_i_pp=np.c_[ psi_field(right_i), phi_field(right_i)]
# left_j_pp =np.c_[ psi_field(left_j), phi_field(left_j)]
# right_j_pp=np.c_[ psi_field(right_j), phi_field(right_j)]
# 
# # if 1: # Slip in the contour values from below for some testing
# #     left_i_pp[:,0]=-1
# #     right_i_pp[:,0]=1
# #     left_i_pp[:,1]=right_i_pp[:,1]=i_contours[::-1]
# #     left_j_pp[:,1]=1
# #     right_j_pp[:,1]=-1
# #     left_j_pp[:,0]=right_j_pp[:,0]=j_contours
# 
# assert np.all( np.isfinite(left_i_pp) )
# assert np.all( np.isfinite(right_i_pp) )
# assert np.all( np.isfinite(left_j_pp) )
# assert np.all( np.isfinite(right_j_pp) )


# Fill in nodes['pp'] and ['x']. Non-rigid nodes get a target pp, from which
# we calculate x.  Rigid nodes get a prescribed x, from which we calculate pp.

for i in range(len(left_i)):
    for j in range(len(left_j)):
        n=elts['nodes'][i,j]
        rigid=True

        if i==0 and left_j_rigid[j]:
            x=left_j[j]
        elif i+1==len(left_i) and right_j_rigid[j]:
            x=right_j[j]
        elif j==0 and left_i_rigid[i]:
            x=left_i[i]
        elif j+1==len(left_j) and right_i_rigid[i]:
            x=right_i[i]
        else:
            rigid=False
            pp=[j_contours[j],
                i_contours[-i-1]] # I think I have to reverse i
            
            x=self.g_int.fields_to_xy(pp,[self.psi,self.phi],x)

        if rigid:
            pp=[min(1,max(-1,psi_field(x))),
                min(1,max(-1,phi_field(x)))]

        patch.nodes['x'][n]=x
        patch.nodes['pp'][n]=pp
        patch.nodes['rigid'][n]=rigid

# Smooth out the deviations from target

target_pp=np.zeros( (patch.Nnodes(),2),np.float64)
target_pp[elts['nodes'],0]=j_contours[None,:]
target_pp[elts['nodes'],1]=i_contours[::-1,None]
dpp=patch.nodes['pp']-target_pp
patch.plot_nodes(values=dpp[:,1],cmap='jet')

# Simple smoothing. Can use elts['nodes'] rather than having to ask about
# neighbors.
dpp_r=dpp[elts['nodes']].copy()
rigid_r=patch.nodes['rigid'][elts['nodes']]

from scipy import signal

for it in range(10):
    smooth=0*dpp_r
    win=np.array([0.5,0,0.5])
    # Smoothing is only along the respective coordinate. I.e. phi
    # anomalies are smoothed along contours of phi, and psi anomalies
    # are smoothed along contours of psi.
    smooth[...,0]=signal.fftconvolve(dpp_r[...,0],win[:,None],mode='same')
    smooth[...,1]=signal.fftconvolve(dpp_r[...,1],win[None,:],mode='same')
        
    # Just update the non-rigid nodes:
    dpp_r[~rigid_r]=smooth[~rigid_r]

dpp=0*patch.nodes['pp']
dpp[elts['nodes']] = dpp_r

# Copy back to pp
sel=~patch.nodes['rigid']
patch.nodes['pp'][sel] = target_pp[sel] + dpp[sel]

# And remap those nodes:
for n in np.nonzero(~patch.nodes['rigid'])[0]:
    x_orig=patch.nodes['x'][n]
    patch.nodes['x'][n]=self.g_int.fields_to_xy(patch.nodes['pp'][n],
                                                [self.psi,self.phi],
                                                x_orig)

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
fig.subplots_adjust(left=0,right=1,bottom=0.01,top=1)

segs=patch.nodes['x'][ patch.edges['nodes'] ]
valid=np.all(np.isfinite(segs), axis=1 )[:,0]
#patch.plot_nodes(mask=patch.nodes['rigid'],color='k',sizes=30)

ncoll=patch.plot_nodes(values=dpp[:,1],cmap='jet')
patch.plot_edges(mask=valid)
plt.colorbar(ncoll)

ax.axis((552440.8693847182, 552491.4300357571, 4124240.871300229, 4124293.9328129073))

##

# HERE: try some relaxation approaches.
#   First, pin the known fixed nodes, and don't worry about
#   trying to keep everybody on the bezier boundary.

# rigid-ness is carried through from the discretized nodestrings,
# with nodes with negative scale and corner nodes set as rigid

from stompy.grid import orthogonalize
tweaker=orthogonalize.Tweaker(patch)

# First, just nudge everybody towards orthogonal:
# BAD.  Too far out of orthogonal.
for n in patch.valid_node_iter():
    if patch.nodes['rigid'][n]: continue
    tweaker.nudge_node_orthogonal(n)

plt.figure(1)
plt.cla()
patch.plot_nodes(mask=patch.nodes['rigid'],color='r',sizes=30)
patch.plot_edges()
#plt.axis( (552066.9646997608, 552207.1805374735, 4124548.347825134, 4124660.504092434) )
plt.axis( (552447.0573990112, 552507.7547532236, 4124244.839335523, 4124293.3901183335) )

## 
from stompy.grid import orthogonalize
tweaker=orthogonalize.Tweaker(patch)

n_free=np.nonzero(~patch.nodes['rigid'])[0]
edge_scales=np.zeros(patch.Nedges(),np.float64)
ec=patch.edges_center()

for orient,scale in zip( [0,90], qg.scales):
    sel=patch.edges['orient']==orient
    edge_scales[sel] = scale(ec[sel])

##

# This produces okay results, but it's going to be super slow
# to converge.
tweaker.smooth_to_scale( n_free, edge_scales,
                         smooth_iters=1,nudge_iters=1)
    
plt.figure(1)
plt.cla()
patch.plot_nodes(mask=patch.nodes['rigid'],color='r',sizes=30)
patch.plot_edges()
plt.axis( (552066.9646997608, 552207.1805374735, 4124548.347825134, 4124660.504092434) )

##

# Status:
# have a topologically good grid
# know which nodes are fixed, which can be moved.
# there is still a fundamental tradeoff: I want to forget "rigid-ness"
# over some length scale
# If I did have some target psi/phi contours, then I could smoothly
# move from locally rigid psi/phi values to the target values.
# I can probably re-use some of the patch code to get target psi/phi
# contours.

