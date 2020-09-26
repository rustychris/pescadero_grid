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

gen_src=unstructured_grid.UnstructuredGrid.read_pickle('grid_lagoon-v03.pkl')

qg=quads.QuadGen(gen_src,cell=0,final='anisotropic',execute=False,nom_res=5)

# Need a better way of reading these in:
qg.add_internal_edge([23,36])
qg.add_internal_edge([20,32])
#qg.execute()

##

no_turn=qg.gen.nodes['turn']==0.0
qg.gen.nodes['turn'][no_turn]=180.0

##

plt.figure(12).clf()
fig,ax=plt.subplots(1,1,num=12)

# Do the angles add up okay?
net_turn=(180-qg.gen.nodes['turn']).sum()
assert np.abs(net_turn-360.0)<1e-10

# What does quad_laplacians need in order to get some starter IJ info?

# I think I want to mimic coalesce_ij_nominal, but use the node angles.

# qg.turns_to_ij_nominal(qg.gen,dest='ij')

self=qg
gen=qg.gen
dest='ij'
nom_res=None
min_steps=2
max_cycle_len=1000
#@utils.add_to(qg)
#def turns_to_ij_nominal(self,gen,dest='IJ',nom_res=None,min_steps=None,
#                         max_cycle_len=1000):
#     """ 
#     Similar to coalesce_ij_nominal(), but use turn angles to track
#     orientation.
#     nom_res: TODO -- figure out good default.  This is the nominal
#      spacing for i and j in geographic units.
#     min_steps: edges should not be shorter than this in IJ space.
# 
#     max_cycle_len: only change for large problems.  Purpose here
#       is to abort on bad inputs/bugs instead of getting into an
#       infinite loop
#     """

self.g_int.plot_edges(ax=ax,color='k',alpha=0.3,lw=0.5)
ax.axis('tight')
ax.axis('equal')

if nom_res is None:
    nom_res=self.nom_res

if not isinstance(nom_res,field.Field):
    nom_res=field.ConstantField(nom_res)

if min_steps is None:
    min_steps=self.min_steps

ij=np.zeros( (gen.Nnodes(),2), np.float64)
ij[...]=np.nan


# Very similar to fill_ij_interp, but we go straight to
# assigning dIJ
cycles=gen.find_cycles(max_cycle_len=1000)
assert len(cycles)==1,"For now, cannot handle multiple cycles"
s=cycles[0]

ij_fixed=(gen.nodes['turn']!=180.0)

gen.add_edge_field('angle',np.nan*np.zeros(gen.Nedges()),
                   on_exists='overwrite')

# Collect the steps so that we can close the sum at the end
# for idx in [0,1]: # i, j
steps=[] # [ [node a, node b, delta], ... ]

# it's a cycle, so we can roll
is_fixed=np.nonzero( ij_fixed[s] )[0]
assert len(is_fixed),"There are no nodes with fixed i,j!"

s=np.roll(s,-is_fixed[0])
s=np.r_[s,s[0]] # repeat first node at the end
# Get the new indices for fixed nodes
is_fixed=np.nonzero( ij_fixed[s] )[0]

dists=utils.dist_along( gen.nodes['x'][s] )

# edge parallel to x axis. This is for each edge,
# relative to the orientation of the first edge
# that's encountered, and relative to a CCW traversal
# of the cell (so not necessarily the orientation of
# the individual edges)
orientation=0 
ec=gen.edges_center()

for a,b in zip( is_fixed[:-1],is_fixed[1:] ):
    d_ab=dists[b]-dists[a]
    ab_nodes=s[a:b+1]
    for n1,n2 in zip(ab_nodes[:-1],ab_nodes[1:]):
        j=gen.nodes_to_edge(n1,n2)
        assert j is not None
        gen.edges['angle'][j]=orientation

    res=nom_res(ec[j])
    n_steps=max( d_ab/res, self.min_steps )
    n_steps_i=n_steps * np.cos(orientation*np.pi/180.0)
    n_steps_j=n_steps * np.sin(orientation*np.pi/180.0)

    # fixed-to-fixed segments
    # [start node, end node, i_steps, j_steps]
    steps.append( [s[a],s[b],round(n_steps_i), round(n_steps_j)] )

    orientation=(orientation + (180-gen.nodes['turn'][s[b]])) % 360.0
    
# not elegant, but ...
# what about just finding the smallest cycles, find the set of nonzero edges
# within each, and figuring out adjusted steps from there?

steps=np.array(steps,np.int32) # [ [node start,node end, num i steps, num j steps], ... ]

# Assign internal edges an orientation:

const_edges=[ [], [] ]

for int_1,int_2 in self.internal_edges:
    interval=[np.nonzero( steps[:,0]==int_1)[0][0],
              np.nonzero( steps[:,0]==int_2)[0][0]]
    if interval[0]>interval[1]:
        interval=interval[::-1]
        
    i_diff=steps[interval[0]:interval[1],2].sum()
    j_diff=steps[interval[0]:interval[1],3].sum()
    # Is this an i-constant or j-constant edge?
    if (np.abs(i_diff) <= np.abs(j_diff)):
        print("Internal edge %d-%d is assumed i-constant"%(int_1,int_2))
        const_edges[0].append( [int_1,int_2] )
    else:
        print("Internal edge %d-%d is assumed j-constant"%(int_1,int_2))
        const_edges[1].append( [int_1,int_2] )


ax.plot(gen.nodes['x'][s,0],
        gen.nodes['x'][s,1],
        'r--')
ax.plot(gen.nodes['x'][s[is_fixed],0],
        gen.nodes['x'][s[is_fixed],1],
        'ro')

ax.plot(gen.nodes['x'][s[is_fixed[0]],0],
        gen.nodes['x'][s[is_fixed[-1]],1],
        'go',ms=4)

for ces,col in zip(const_edges,['orange','tab:green']):
    for nodes in ces:
        ax.plot( gen.nodes['x'][nodes,0],
                 gen.nodes['x'][nodes,1],
                 color=col)

gen.plot_nodes(mask=ij_fixed,labeler='id')

# So the error distribution approach is not robust.
# narrow features, where we can't afford an adjustment,
# can arise without any short edges. long edges can
# have some small adjustments, but then a narrow
# feature's width may be defined by the difference
# of two long features...

# What do we really need, though, for the remainder of
# the algorithm?  can just the edge angles and the internal
# edges provide enough information?
#  - add_bezier

## 

for idx in [1,0]: # [0,1]:
    cycles=[steps]
    for n1,n2 in const_edges[idx]:
        new_cycles=[]
        for cycle in cycles:
            # split the cycle on this pair of nodes.
            if (n1 not in cycle[:,0]) and (n2 not in cycle[:,0]):
                new_cycles.append(cycle)
                continue
            if (n1 not in cycle[:,0]) or (n2 not in cycle[:,0]):
                raise Exception("How is only one node of the internal edge in the cycle?")
            n1i,n2i=[ np.nonzero(cycle[:,0]==n)[0][0]
                      for n in [n1,n2]]
            if n1i>n2i:
                n1i,n2i=n2i,n1i
            cycleA=np.concatenate( [cycle[:n1i],
                                    [ [cycle[n1i,0],
                                       cycle[n2i,0],
                                       (idx==1),(idx==0)] ],
                                    cycle[n2i:]], axis=0 )
            cycleB=np.concatenate( [ cycle[n1i:n2i],
                                     [ [cycle[n2i,0],
                                        cycle[n1i,0],
                                        (idx==1),(idx==0)] ] ],
                                   axis=0)
            new_cycles.append(cycleA)
            new_cycles.append(cycleB)
        cycles=new_cycles

    for cyc in cycles:
        col=next(nice_colors)
        cyc_nodes=np.r_[ cyc[:,0], cyc[-1,1] ]
        cyc_xy=gen.nodes['x'][cyc_nodes]
        ax.plot(cyc_xy[:,0],cyc_xy[:,1],color=col,lw=3.0,alpha=0.4)

        # And apply an adjustment per cycle:
        adj_steps=cyc[:,2:].copy()
        errs=adj_steps[:,idx].sum() # cumulative error
        stepsizes=np.abs(adj_steps[:,idx])

        err_dist=np.round(errs*np.cumsum(np.r_[0,stepsizes])/stepsizes.sum())
        err_per_step = np.diff(err_dist).astype(np.int32)
        # DBG
        # adj_steps[:,idx] -= err_per_step

        # Have to take into account that some ij may already be set
        # from an earlier cycle
        coord_tmp=np.r_[0,np.cumsum(adj_steps[:-1,idx])]
        coord_nodes=np.r_[cyc[0,0], cyc[:-1,1]]
        existing=np.isfinite( ij[coord_nodes,idx] )
        if np.any(existing):
            offsets=ij[coord_nodes[existing],idx] - coord_tmp[existing]
            assert offsets.min()==offsets.max()
            coord_tmp+=offsets[0].astype(np.int32)
        ij[coord_nodes,idx]=coord_tmp
    break

# i is okay, but j is losing the side channel.
# is this a bug, or a limitation of the adjustment?

gen.add_node_field(dest,ij,on_exists='overwrite')
ij_fixed=np.c_[ij_fixed,ij_fixed]
gen.add_node_field(dest+'_fixed',ij_fixed,on_exists='overwrite')

# qg.turns_to_ij_nominal(qg.gen,dest='ij')

gen.plot_nodes(mask=ij_fixed[:,0],labeler='ij')

##

qg.fill_ij_interp(qg.gen)
qg.node_ij_to_edge(qg.gen)

qg.gen.add_node_field('IJ',qg.gen.nodes['ij'],on_exists='overwrite')
qg.gen.add_node_field('IJ_fixed',qg.gen.nodes['ij_fixed'],on_exists='overwrite')

qg.node_ij_to_edge(qg.gen,dest='ij')
qg.node_ij_to_edge(qg.gen,dest='IJ')

# Seems to get stuck, coord=1, swath=2(?)
# psi_phi field is bad.
# Goes awry at the side channel.
# Ah - it's because the ij don't match, and that disables the internal
# edge (that's probably the problem...)

qg.execute()

##

qg.plot_gen_bezier()
##

qg.plot_psi_phi_setup()


##
g=qg.g_final

plt.figure(4).clf()
fig,ax=plt.subplots(num=4)
g.plot_edges(color='k',lw=0.5)
g.plot_cells(color='0.8',lw=0,zorder=-2)


ax.axis('off')
ax.set_position([0,0,1,1])
