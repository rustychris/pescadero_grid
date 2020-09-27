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

@utils.add_to(qg)
def prepare_angles(self):
    # Allow missing angles to either be 0 or nan
    gen=self.gen
    
    missing=np.isnan(gen.nodes['turn'])
    gen.nodes['turn'][missing]=0.0
    no_turn=gen.nodes['turn']==0.0
    gen.nodes['turn'][no_turn]=180.0
    gen.add_node_field('fixed',~no_turn,on_exists='pass')
    
    # Do the angles add up okay?
    net_turn=(180-gen.nodes['turn']).sum()
    assert np.abs(net_turn-360.0)<1e-10
    
    gen.add_edge_field('angle',np.nan*np.zeros(gen.Nedges()),
                       on_exists='overwrite')

    # relative to the orientation of the first edge
    # that's encountered, and relative to a CCW traversal
    # of the cell (so not necessarily the orientation of
    # the individual edges)
    orientation=0 

    cycles=gen.find_cycles(max_cycle_len=1000)
    assert len(cycles)==1,"For now, cannot handle multiple cycles"
    cycle=cycles[0]

    for a,b in zip( cycle, np.roll(cycle,-1) ):
        j=gen.nodes_to_edge(a,b)
        assert j is not None
        gen.edges['angle'][j]=orientation

        orientation=(orientation + (180-gen.nodes['turn'][b])) % 360.0

qg.prepare_angles()

qg.add_bezier(qg.gen)
qg.plot_gen_bezier()
# qg.gen.plot_edges(labeler='angle') # yep
qg.g_int=qg.create_intermediate_grid_tri()

##

plt.clf()
qg.plot_intermediate() # good.

NodeDiscretization=quads.NodeDiscretization

@utils.add_to(qg)
def calc_bc_gradients(self,gtri):
    """
    Calculate gradient vectors for psi and phi along
    the boundary.
    """
    bcycle=gtri.boundary_cycle()

    # First calculate psi gradient per edge:
    j_grad_psi=np.zeros( (len(bcycle),2), np.float64)
    j_angles=self.gen.edges['angle'][ gtri.edges['gen_j'] ] * np.pi/180.
    
    for ji,(n1,n2) in enumerate( zip(bcycle[:-1],bcycle[1:]) ):
        j=gtri.nodes_to_edge(n1,n2)
        tang_xy=utils.to_unit( gtri.nodes['x'][n2] - gtri.nodes['x'][n1] )
        
        tang_ij=np.r_[ np.cos(j_angles[j]), np.sin(j_angles[j])]

        # Construct a rotation R such that R.dot(tang_ij)=[1,0],
        # then apply to tang_xy
        Rpsi=np.array([[tang_ij[0], tang_ij[1]],
                       [-tang_ij[1], tang_ij[0]] ] )
        j_grad_psi[ji,:]=Rpsi.dot(tang_xy)

    # Interpolate to nodes
    bc_grad_psi=np.zeros( (len(bcycle),2), np.float64)

    N=len(bcycle)
    for ni in range(N):
        bc_grad_psi[ni,:]=0.5*( j_grad_psi[ni,:] +
                                j_grad_psi[(ni-1)%N,:] )

    bc_grad_phi=np.zeros( (len(bcycle),2), np.float64)

    # 90 CW from psi
    bc_grad_phi[:,0]=bc_grad_psi[:,1]
    bc_grad_phi[:,1]=-bc_grad_psi[:,0]

    # Convert to dicts:
    grad_psi={}
    grad_phi={}
    for ni,n in enumerate(bcycle):
        grad_psi[n]=bc_grad_psi[ni,:]
        grad_phi[n]=bc_grad_phi[ni,:]
    return grad_psi,grad_phi

grad_psi,grad_phi=qg.calc_bc_gradients(qg.g_int)

nodes=list(grad_psi.keys())
grads=np.array( [grad_psi[n] for n in nodes] )

# okay!
plt.quiver( qg.g_int.nodes['x'][nodes,0],
            qg.g_int.nodes['x'][nodes,1],
            grads[:,0],grads[:,1])

@utils.add_to(qg)
def calc_psi_phi(self):
    gtri=self.g_int
    self.nd=nd=NodeDiscretization(gtri)

    e2c=gtri.edge_to_cells()

    # check boundaries and determine where Laplacian BCs go
    boundary=e2c.min(axis=1)<0
    i_dirichlet_nodes={} # for psi
    j_dirichlet_nodes={} # for phi

    # Block of nodes with a zero-tangential-gradient BC
    i_tan_groups=[]
    j_tan_groups=[]
    # i_tan_groups_i=[] # the input i value
    # j_tan_groups_j=[] # the input j value

    # Try zero-tangential-gradient nodes.  Current code will be under-determined
    # without the derivative constraints.
    bcycle=gtri.boundary_cycle()
    n1=bcycle[-1]
    i_grp=None
    j_grp=None

    psi_gradients,phi_gradients=self.calc_bc_gradients(gtri)
    psi_gradient_nodes={} # node => unit vector of gradient direction
    phi_gradient_nodes={} # node => unit vector of gradient direction

    j_angles=self.gen.edges['angle'][ gtri.edges['gen_j'] ]

    for n2 in bcycle:
        j=gtri.nodes_to_edge(n1,n2)

        imatch=j_angles[j] % 180==0
        jmatch=j_angles[j] % 180==90:
        
        if imatch: 
            if i_grp is None:
                i_grp=[n1]
                i_tan_groups.append(i_grp)
                # i_tan_groups_i.append(i1)
            i_grp.append(n2)
        else:
            i_grp=None

        if jmatch:
            if j_grp is None:
                j_grp=[n1]
                j_tan_groups.append(j_grp)
                # j_tan_groups_j.append(j1)
            j_grp.append(n2)
        else:
            j_grp=None

        if not (imatch or jmatch):
            # Register gradient BC for n1
            psi_gradient_nodes[n1]=psi_gradients[n1]
            psi_gradient_nodes[n2]=psi_gradients[n2]
            phi_gradient_nodes[n1]=phi_gradients[n1]
            phi_gradient_nodes[n2]=phi_gradients[n2]
        n1=n2

    # HERE: have to check the closing edge instead of matching
    # values
    
    # bcycle likely starts in the middle of either a j_tan_group or i_tan_group.
    # see if first and last need to be merged
    if i_tan_groups[0][0]==i_tan_groups[-1][-1]:
        i_tan_groups[0].extend( i_tan_groups.pop()[:-1] )
    if j_tan_groups[0][0]==j_tan_groups[-1][-1]:
        j_tan_groups[0].extend( j_tan_groups.pop()[:-1] )

    # Set the range of psi to [-1,1], and pin some j to 1.0
    low_i=np.argmin(i_tan_groups_i)
    high_i=np.argmax(i_tan_groups_i)

    i_dirichlet_nodes[i_tan_groups[low_i][0]]=-1
    i_dirichlet_nodes[i_tan_groups[high_i][0]]=1
    j_dirichlet_nodes[j_tan_groups[1][0]]=1

    # Extra degrees of freedom:
    # Each tangent group leaves an extra dof (a zero row)
    # and the above BCs constrain 3 of those
    dofs=len(i_tan_groups) + len(j_tan_groups) - 3
    assert dofs>0

    # Use the internal_edges to combine tangential groups
    def join_groups(groups,nA,nB):
        grp_result=[]
        grpA=grpB=None
        for grp in groups:
            if nA in grp:
                assert grpA is None
                grpA=grp
            elif nB in grp:
                assert grpB is None
                grpB=grp
            else:
                grp_result.append(grp)
        assert grpA is not None
        assert grpB is not None
        grp_result.append( list(grpA) + list(grpB) )
        return grp_result

    for gen_edge in self.internal_edges:
        edge=[self.g_int.select_nodes_nearest(x)
              for x in self.gen.nodes['x'][gen_edge]]
        edge_ij=self.gen.nodes['ij'][gen_edge]
        dij=np.abs( edge_ij[1] - edge_ij[0] )

        if dij[0]<1e-10: # join on i
            print("Joining two i_tan_groups")
            i_tan_groups=join_groups(i_tan_groups,edge[0],edge[1])
        elif dij[1]<1e-10: # join on j
            print("Joining two j_tan_groups")
            j_tan_groups=join_groups(j_tan_groups,edge[0],edge[1])
        else:
            import pdb
            pdb.set_trace()
            print("Internal edge doesn't appear to join same-valued contours")

    self.i_dirichlet_nodes=i_dirichlet_nodes
    self.i_tan_groups=i_tan_groups
    self.i_grad_nodes=psi_gradient_nodes
    self.j_dirichlet_nodes=j_dirichlet_nodes
    self.j_tan_groups=j_tan_groups
    self.j_grad_nodes=phi_gradient_nodes

    Mblocks=[]
    Bblocks=[]
    if 1: # PSI
        M_psi_Lap,B_psi_Lap=nd.construct_matrix(op='laplacian',
                                                dirichlet_nodes=i_dirichlet_nodes,
                                                zero_tangential_nodes=i_tan_groups,
                                                gradient_nodes=psi_gradient_nodes)
        Mblocks.append( [M_psi_Lap,None] )
        Bblocks.append( B_psi_Lap )
    if 1: # PHI
        # including phi_gradient_nodes, and the derivative links below
        # is redundant but balanced.
        M_phi_Lap,B_phi_Lap=nd.construct_matrix(op='laplacian',
                                                dirichlet_nodes=j_dirichlet_nodes,
                                                zero_tangential_nodes=j_tan_groups,
                                                gradient_nodes=phi_gradient_nodes)
        Mblocks.append( [None,M_phi_Lap] )
        Bblocks.append( B_phi_Lap )
    if 1:
        # Not sure what the "right" value is here.
        # When the grid is coarse and irregular, the
        # error in these blocks can overwhelm the BCs
        # above.  This scaling decreases the weight of
        # these blocks.
        # 0.1 was okay
        # Try normalizing based on degrees of freedom.
        # how many dofs are we short?
        # This assumes that the scale of the rows above is of
        # the same order as the scale of a derivative row below.

        # each of those rows constrains 1 dof, and I want the
        # set of derivative rows to constrain dofs. And there
        # are 2*Nnodes() rows.
        # Hmmm.  Had a case where it needed to be bigger (lagoon)
        # Not sure why.
        if self.gradient_scale=='scaled':
            gradient_scale = dofs / (2*gtri.Nnodes())
        else:
            gradient_scale=self.gradient_scale

        # PHI-PSI relationship
        # When full dirichlet is used, this doesn't help, but if
        # just zero-tangential-gradient is used, this is necessary.
        Mdx,Bdx=nd.construct_matrix(op='dx')
        Mdy,Bdy=nd.construct_matrix(op='dy')
        if gradient_scale!=1.0:
            Mdx *= gradient_scale
            Mdy *= gradient_scale
            Bdx *= gradient_scale
            Bdy *= gradient_scale
        Mblocks.append( [Mdy,-Mdx] )
        Mblocks.append( [Mdx, Mdy] )
        Bblocks.append( np.zeros(Mdx.shape[1]) )
        Bblocks.append( np.zeros(Mdx.shape[1]) )

    self.Mblocks=Mblocks
    self.Bblocks=Bblocks

    bigM=sparse.bmat( Mblocks )
    rhs=np.concatenate( Bblocks )

    psi_phi,*rest=sparse.linalg.lsqr(bigM,rhs)
    self.psi_phi=psi_phi
    self.psi=psi_phi[:gtri.Nnodes()]
    self.phi=psi_phi[gtri.Nnodes():]

    # Using the tan_groups, set the values to be exact
    for i_grp in i_tan_groups:
        self.psi[i_grp]=self.psi[i_grp].mean()
    for j_grp in j_tan_groups:
        self.phi[j_grp]=self.phi[j_grp].mean()


qg.calc_psi_phi()



## 
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
#     this gets simpler, since I now store the angles explicitly.
#  - create_intermediate_grid_tri(src='IJ')
#     interpolates ij values along the boundary.  depending on how this is used,
#     might be enough just to explicitly store the generating nodes where fixed.
#     should also copy edge angles here.
#  - calc_psi_phi()
#     grouping can be derived from edge angles.  i,j values are currently used
#     to select nodes which are then used for dirichlet BCs.
#  - create_final_by_patches()


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
