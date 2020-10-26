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
gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v09.pkl')
gen_src.delete_orphan_edges()

gen_src.renumber_cells()

if 1:
    plt.figure(1).clf()
    gen_src.plot_cells(labeler='id',centroid=True)
    plt.axis('tight')
    plt.axis('equal')

## 
qg=quads.QuadGen(gen_src,
                 cells=[13],
                 final='anisotropic',execute=False,
                 triangle_method='gmsh',
                 nom_res=3.5)

# Can I speed this up a bit?
# 46s, with 37 in create_final_by_patches
#   15s 2300 delaunay node insertions, from trace_and_insert_contour
#   13s in fields_to_xy
# 10s in construct_matrix
# What is the real value in trace_and_insert_contour?
%prun -s cumulative qg.execute()

##

# So I can probably save a good bit of time by finding intersections in
# a different way .

class IntersectingLines(unstructured_grid.UnstructuredGrid):
    """ 
    Lightweight alternative to exact_delaunay when the goal is to 
    just add nodes at intersecting constraints
    """
    edge_dtype=(unstructured_grid.UnstructuredGrid.edge_dtype +
                [ ('constrained',np.bool8) ] )

    def __init__(self,*a,**k):
        super(IntersectingLines,self).__init__(*a,**k)
        self.edge_defaults['constrained']=True
        
    def split_constraint(self,x,j):
        j_new,n_new=self.split_edge_basic(j=j,x=x)
        return n_new
    
    def add_constrained_linestring(self,trace_points,
                                   on_intersection='insert', 
                                   on_exists='stop', # 'stop','exception',
                                   closed=False):
        # => trace_nodes,trace_edges
        nodes=[self.add_or_find_node(x=x)
               for x in trace_points]
        result_nodes=[nodes[0]]
        result_edges=[]

        assert on_intersection=='insert',"I'm not as general as exact_delaunay"
        
        if not closed:
            ab_list=zip(nodes[:-1],nodes[1:])
        else:
            ab_list=zip(nodes,np.roll(nodes,-1))
            
        for a,b in ab_list:
            sub_nodes,sub_edges=self.add_constraint_and_intersections(a,b,
                                                                      on_exists=on_exists)
            result_nodes+=sub_nodes[1:]
            result_edges+=sub_edges

            if (on_exists=='stop') and (sub_nodes[-1]!=b):
                print("Stopping early")
                break
        return result_nodes,result_edges

    def add_constraint_and_intersections(self,nA,nB,on_exists='stop'):
        all_segs=[ [nA,nB] ]
        result_nodes=[nA]
        result_edges=[]

        # Each iteration pull a segment, test for intersections, and
        # either add as edge, or split intersecting constraints and
        # push the two pieces onto the stack
        while all_segs:
            nA,nB=all_segs.pop(0)

            xys=self.nodes['x'][ [nA,nB], :]
            xxyy=[ xys[:,0].min(), xys[:,0].max(),
                   xys[:,1].min(), xys[:,1].max()]

            j=self.nodes_to_edge(nA,nB)
            if j is not None:
                # Already exists:
                if on_exists=='exception':
                    raise Exception("Constraint already exists")
                elif on_exists=='ignore':
                    pass
                elif on_exists=='stop':
                    break
                else:
                    assert False,"Bad value %s for on_exists"%on_exists
                
            j_sel=np.nonzero( self.edge_clip_mask(xxyy,ends=True) )[0]

            segAB=xys
            seg_other=self.nodes['x'][self.edges['nodes'][j_sel]]
            
            # Would AB intersect any of j_sel?
            for j_cross,seg in zip(j_sel,seg_other):

                x_int,alphas=utils.segment_segment_intersection(segA,seg)

                if x_int is None:
                    continue
                
                # Handle interecting case
                if alphas[1]==0:
                    # segA crosses first node of seg
                    coll_node=self.edges['nodes'][j_cross,0]
                    all_segs.insert(0, [nA,coll_node] )
                    all_segs.insert(1, [coll_node,nB] )
                    continue
                elif alphas[1]==1:
                    # segA crosses second node of seg
                    coll_node=self.edges['nodes'][j_cross,1]
                    all_segs.insert(0, [nA,coll_node] )
                    all_segs.insert(1, [coll_node,nB] )
                    continue
                else:
                    n_new=self.split_constraint(j=j_cross,x=x_int)

                    # Why are these here?
                    if nB!=n_new:
                        all_segs.insert(0,[n_new,nB])
                    if nA!=n_new:
                        all_segs.insert(0,[nA,n_new])
                    continue
            result_nodes.append(nB)
            assert j is not None
            result_edges.append(j)
                
        return result_nodes,result_edges
        
    def bulk_init(self,points):
        self.nodes=np.zeros(len(points),self.node_dtype)
        self.nodes[:]=self.node_defaults
        self.nodes['x']=points


g_src=unstructured_grid.UnstructuredGrid(max_sides=4)
g_src.add_rectilinear([0,0],[10,10],11,11)

g_final=IntersectingLines(extra_edge_fields=[
    ('angle',np.float64),
    ('psiphi',np.float64,2)])

g_final.bulk_init(g_src.nodes['x'])

plt.figure(1).clf()
g_final.plot_nodes()

for j in g_src.valid_edge_iter():
    g_final.add_constrained_linestring( g_src.nodes['x'][g_src.edges['nodes'][j]] )
    

##




## Crap that doesn't work
# Start with fractional, then come back to coerce to
# integers.
# Hmmm...
#   If I do this per edge, I'm going to get some distortion
#   Whatever resolution that comes out of this step will not
#   quite be the same distribution as what comes out of the
#   psi/phi field. I want psi/phi to dictate distributions as
#   much as possible
#   Say I nail down the counts along edge-sets joining
#   rigid nodes.  That's okay, but then to actually be able
#   to generate the grid one patch at a time, I need the distribution
#   of nodes on those edges. Assuming that these are generally short
#   sections across a channel, maybe it's okay to just set them.

# plan:
#   What about going back to just setting ij coordinates on nodes?
#   But that puts a lot of pressure on the user to come up with
#   good spacings.  i,j could be used just to get general scales,
#   and to get specific spacing only on internal edges.

# all of these suck.

# Is it better just to got with something considerably more basic?

ij_counts=np.zeros( (gen.Nedges(),2), np.float32 )

# Make a parent grid that has edges only between fixed nodes.

def end_p(n,js):
    if len(js)!=2: return True
    if gen.edges['angle'][js[0]] != gen.edges['angle'][js[1]]:
        return True
    return False

strings=gen.extract_linear_strings(end_func=end_p)
    
## 
gen_parent=unstructured_grid.UnstructuredGrid(max_sides=100)

gen_to_parent={} # node indices

for s in strings:
    nparents=[]
    
    for n in [s[0],s[-1]]:
        if n not in gen_to_parent:
            gen_to_parent[n]=gen_parent.add_node( x=gen.nodes['x'][n] )
        nparents.append( gen_to_parent[n] )
    gen_parent.add_edge(nodes=nparents)

##

gen_parent.plot_edges(color='g',lw=1.)
plt.axis('tight')
plt.axis('equal')

## 
quads.plot_gen_bezier(gen)
gen.plot_nodes(mask=[0],color='r',sizes=30)
plt.axis( (552079.815212804, 552424.5384625637, 4124455.1432898846, 4124643.825499759) )


## 
plt.figure(1).clf()
gen.plot_cells(labeler='id',centroid=True)
gen.plot_edges(labeler='angle')
gen.plot_nodes(labeler='id',mask=[n],color='r')
plt.axis('tight')
plt.axis('equal')
plt.axis( (552079.815212804, 552424.5384625637, 4124455.1432898846, 4124643.825499759) )
    
    
##     
if 0: # old gen bezier code    
    cycles=gen.find_cycles(max_cycle_len=gen.Nnodes()
    assert len(cycles)==1
    cycle=cycles[0]

    for a,b,c in zip( np.roll(cycle,1),
                      cycle,
                      np.roll(cycle,-1) ):
        ab=gen.nodes['x'][b] - gen.nodes['x'][a]
        bc=gen.nodes['x'][c] - gen.nodes['x'][b]

        j_ab=gen.nodes_to_edge(a,b)
        j_bc=gen.nodes_to_edge(b,c)

        # This makes use of angle being defined relative to a CCW
        # cycle, not the order of edge['nodes']
        dtheta_ij=(gen.edges['angle'][j_bc] - gen.edges['angle'][j_ab])*np.pi/180.
        dtheta_ij=(dtheta_ij+np.pi)%(2*np.pi) - np.pi

        # Angle of A->B
        theta0=np.arctan2(ab[1],ab[0])
        theta1=np.arctan2(bc[1],bc[0])
        dtheta=(theta1 - theta0 + np.pi) % (2*np.pi) - np.pi

        theta_err=dtheta-dtheta_ij
        # Make sure we're calculating error in the shorter direction
        theta_err=(theta_err+np.pi)%(2*np.pi) - np.pi

        cp0 = gen.nodes['x'][b] + utils.rot(  theta_err/2, 1./3 * -ab )
        if gen.edges['nodes'][j_ab,0]==b:
            cp_i=1
        else:
            cp_i=2
        gen.edges['bez'][j_ab,cp_i] = cp0

        cp1 = gen.nodes['x'][b] + utils.rot( -theta_err/2, 1./3 * bc )
        if gen.edges['nodes'][j_bc,0]==b:
            cp_i=1
        else:
            cp_i=2
        gen.edges['bez'][j_bc,cp_i] = cp1



##
qg.execute()
qg.plot_result()

##

# Maybe it's better to deal with the spacing issues as part of relaxation.
# Need to annotate the edges in g_final with their orientation, then
# calculate their target length, then we can do some relaxing.

def label_edge_orientation(g):
    j_orients=np.zeros( g.Nedges(), np.int32) - 1
    jns=g.edges['nodes']

    psi_match=(g.nodes['pp'][jns[:,0],0]==g.nodes['pp'][jns[:,1],0])
    phi_match=(g.nodes['pp'][jns[:,0],1]==g.nodes['pp'][jns[:,1],1])
    assert np.all( psi_match | phi_match )

    j_orients[psi_match]=0
    j_orients[phi_match]=90

    g.add_edge_field('orient',j_orients,on_exists='overwrite')

def target_edge_lengths(g,scales):
    ec=g.edges_center()
    target_scales=np.zeros(g.Nedges(),np.float64)
    target_scales[psi_match]=scales[0]( ec[psi_match] )
    target_scales[phi_match]=scales[0]( ec[phi_match] )
    return target_scales


def classify_nodes(g,gen):
    # Find rigid nodes by matching up with gen
    n_fixed=[]
    for gen_n in np.nonzero( gen.nodes['fixed'] )[0]:
        n=g.select_nodes_nearest( gen.nodes['x'][gen_n], max_dist=0.001)
        if n is not None:
            n_fixed.append(n)

    n_free=[n for n in g.valid_node_iter() if n not in n_fixed]
    return n_fixed, n_free


g_final=qg.g_final
label_edge_orientation(g_final)
target_scales=target_edge_lengths(g_final,qg.scales)
n_fixed,n_free=classify_nodes(g_final,qg.gen)

##
from stompy.grid import orthogonalize

# Here's what umbra does:
g=g_final.copy()
tweaker=orthogonalize.Tweaker(g)

plt.clf()
ccoll=g.plot_edges(color='k',lw=0.4)

##

# The sting really exposes the issue in tweaker about not honoring
# lack of local connection

# nudge_orthogonal: node operation. calculate the ideal position of the node
#   w.r.t. each cell, use the average
# local_smooth: fit x,y ~ a*i + b*j + c
#  using a small-ish stencil around the node, adjust node.

# A more flexible approach to the stencil,

# test case has 2012 free nodes
# stencil default r=1, for 9 nodes
# so I could build a (2012,3,3) matrix


# 0.8s.  hrrm.
@utils.add_to(tweaker)
def precalc_stencils(self,n_free):
    g=self.g
    stencil_radius=1
    
    stencils=np.zeros( (len(n_free),1+2*stencil_radius,1+2*stencil_radius), np.int32) - 1

    ij0=np.array([stencil_radius,stencil_radius])

    all_Nsides=np.array([g.cell_Nsides(c) for c in range(g.Ncells())])
    dij=np.array([1,0])
    rot=np.array([[0,1],[-1,0]])

    for ni,n in enumerate(n_free):
        # this is a bit more restrictive than it needs to be
        # but it's too much to make it general right now.
        cells=g.node_to_cells(n)
        if len(cells)!=4: continue
        if any( all_Nsides[cells] != 4):
            continue

        stencils[ni,ij0[0],ij0[1]]=n

        nbrs=g.node_to_nodes(n)

        he=g.nodes_to_halfedge(n,nbrs[0])

        for nbr in nbrs:
            he=g.nodes_to_halfedge(n,nbr)
            stencils[ni,ij0[0]+dij[0],ij0[1]+dij[1]]=he.node_fwd()
            he_fwd=he.fwd()
            ij_corner=ij0+dij+rot.dot(dij)
            stencils[ni,ij_corner[0],ij_corner[1]]=he_fwd.node_fwd()
            dij=rot.dot(dij)
    return stencils


# so a node
@utils.add_to(tweaker)
def local_smooth_flex(self,node_idxs,n_iter=3,free_nodes=None,
                      min_halo):
    """
    Fit regular grid patches iteratively within the subset of nodes given
    by node_idxs.
    Currently requires that node_idxs has a sufficiently large footprint
    to have some extra nodes on the periphery.

    node_idxs: list of node indices
    n_iter: count of how many iterations of smoothing are applied.
    free_subset: node indexes (i.e. indices of g.nodes) that are allowed 
     to move.  Defaults to all of node_idxs subject to the halo.
    """
    g=self.g
    stencil_radius=1
    
    node_stencils=self.precalc_stencils(node_idxs)
    node_stencils=node_stencils.reshape([-1,3*3])
    
    pad=1+stencil_radius
    
    stencil_rows=[]
    for i in range(-stencil_radius,stencil_radius+1):
        for j in range(-stencil_radius,stencil_radius+1):
            stencil_rows.append([i,j])
    design=np.array(stencil_rows)

    # And fit a surface to the X and Y components
    #  Want to fit an equation
    #   x= a*i + b*j + c
    M=np.c_[design,np.ones(len(design))]

    XY=g.nodes['x']
    new_XY=XY.copy()

    if free_nodes is not None:
        # use dict for faster tests
        free_nodes={n:True for n in free_nodes}

    moved_nodes={}
    stencil_ctr=stencil_radius*(2*stencil_radius+1) + stencil_radius
    
    for count in range(n_iter):
        new_XY[...]=XY
        for ni,n in enumerate(node_idxs):
            if node_stencils[ni,stencil_ctr]<0:
                continue
            if (free_nodes is not None) and (n not in free_nodes): continue

            # Query XY to estimate where n "should" be.
            # [9,{x,y}] rhs
            XY_sten=XY[node_stencils[ni],:] - XY[n]

            valid=np.isfinite(XY_sten[:,0])

            xcoefs,resid,rank,sing=np.linalg.lstsq(M[valid],XY_sten[valid,0],rcond=-1)
            ycoefs,resid,rank,sing=np.linalg.lstsq(M[valid],XY_sten[valid,1],rcond=-1)

            delta=np.array( [xcoefs[2],
                             ycoefs[2]])

            new_x=XY[n] + delta
            if np.isfinite(new_x[0]):
                new_XY[n]=new_x
                moved_nodes[n]=True
            else:
                pass # print("Hit nans.")
        # Update all at once to avoid adding variance due to the order of nodes.
        XY[...]=new_XY

    # Update grid
    count=0
    for ni,n in enumerate(node_idxs):
        if n not in moved_nodes: continue

        dist=utils.mag(XY[n] - g.nodes['x'][n])
        if dist>1e-6:
            g.modify_node(n,x=XY[n])
            count+=1

    for n in list(moved_nodes.keys()):
        for nbr in g.node_to_nodes(n):
            if nbr not in moved_nodes:
                moved_nodes[nbr]=True
    for n in moved_nodes.keys():
        if (free_nodes is not None) and (n not in free_nodes): continue
        self.nudge_node_orthogonal(n)

## 
tweaker.local_smooth_flex(n_free,n_iter=1)
plt.clf()
ccoll=g.plot_edges(color='k',lw=0.4)

##

# What about using the psi/phi field to get local gradient, then
# adjust for spacing based on local gradients?
# do I even need the original field?  Can I just fit a quadratic
# bezier to the neighboring nodes?
# zoom=(552151.4461024263, 552173.7461286188, 4124570.960167043, 4124589.92071104)
# zoom=(552283.3560604592, 552305.021722466, 4124334.2076858296, 4124352.6288631936)
# Sample node:
# n=g_final.select_nodes_nearest([552160.92, 4124579.33])
# n=389

plt.clf()
ccoll=g.plot_edges(color='k',lw=0.4)
g.plot_nodes(clip=zoom,labeler='id')
g.plot_nodes(mask=[n],color='r')

##
from scipy import interpolate

g=g_final.copy()

## 
node_moves=np.zeros( (len(n_free),2), np.float64)
el=g.edges_length()

for ni,n in enumerate(n_free):
    nbrs=g.angle_sort_adjacent_nodes(n)

    j_nbrs=[g.nodes_to_edge(n,nbr) for nbr in nbrs]

    for orient in [0,90]:
        pair=[(j,nbr) for j,nbr in zip(j_nbrs,nbrs)
              if g.edges['orient'][j]==orient]
        if len(pair)!=2:
            continue

        nodes=[pair[0][1],n,pair[1][1]]
        js=   [pair[0][0], pair[1][0]]

        node_xy=g.nodes['x'][nodes]
        s=[-1,0,1]

        x_tck=interpolate.splrep( s, node_xy[:,0], k=2 )
        y_tck=interpolate.splrep( s, node_xy[:,1], k=2 )

        jls=el[js] # lengths of those
        jts=target_scales[js]

        # What I want is
        # (jls[0]+dl)/jts[0] ~ (jls[1]-dl)/jts[1]
        # with dl the move towards nodes[2]
        #  (jls[0]+dl)/jts[0] - (jls[1]-dl)/jts[1] = 0
        #  jls[0]/jts[0] + dl/jts[0] - ( jls[1]/jts[1] - dl/jts[1]) = 0
        #  jls[0]/jts[0] + dl/jts[0] - jls[1]/jts[1] + dl/jts[1] = 0
        #  dl/jts[0] + dl/jts[1] = jls[1]/jts[1] - jls[0]/jts[0]
        #  dl= (jls[1]/jts[1] - jls[0]/jts[0]) / ( 1/jts[0] + 1/jts[1])
        dl=(jls[1]/jts[1] - jls[0]/jts[0]) / ( 1/jts[0] + 1/jts[1])
        if dl>0:
            ds=dl/jls[1]
        else:
            ds=dl/jls[0]

        new_xy=np.array( [interpolate.splev(ds, x_tck),
                          interpolate.splev(ds, y_tck)] )
        node_moves[ni]+=new_xy-node_xy[1]

for ni,n in enumerate(n_free):
    g.modify_node(n,x=g.nodes['x'][n] + 0.5*node_moves[ni])

tweaker=orthogonalize.Tweaker(g)
for n in n_free:
    tweaker.nudge_node_orthogonal(n)

##     
plt.clf()
#ccoll=g_final.plot_edges(color='orange',lw=0.4)
ccoll=g.plot_edges(color='k',lw=0.4)
plt.axis( (552047.9351268414, 552230.9809219765, 4124547.643451654, 4124703.282891116) )

##

# That's working reasonably well.
# Not super fast, but okay.
# Next step: decide how to deal with patches, swaths, and
# avoiding the global dependence of resolution
