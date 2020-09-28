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

qg=quads.QuadGen(gen_src,cell=0,final='anisotropic',execute=False,
                 nom_res=5,
                 scales=[field.ConstantField(5),
                         field.ConstantField(5)])

# Need a better way of reading these in:
qg.add_internal_edge([23,36])
qg.add_internal_edge([20,32])

qg.execute()
qg.plot_result()

## 

qg.prepare_angles()
qg.add_bezier(qg.gen)
# qg.plot_gen_bezier()
qg.g_int=qg.create_intermediate_grid_tri()

qg.calc_psi_phi()

qg.plot_psi_phi_setup()
plt.axis('equal')

qg.plot_psi_phi(thinning=0.5)

qg.g_final=qg.create_final_by_patches()

##


## 
#  - create_final_by_patches()
#     adjust to recalculate the swath resolution directly from scale field.


NodeDiscretization=quads.NodeDiscretization
@utils.add_to(qg)
def create_final_by_patches(self):
    fixed_int_to_gen = self.map_fixed_int_to_gen(self.g_int,self.gen)
    n_fixed=list(fixed_int_to_gen.keys())

    g_int=self.g_int
    angles=np.zeros(g_int.Nedges(),np.float32)
    angles=np.where( g_int.edges['gen_j']>=0,
                     self.gen.edges['angle'][g_int.edges['gen_j']],
                     np.nan )
    g_int.add_edge_field('angle',angles,on_exists='overwrite')

    # misnomer.  Not final.  Just for finding exact intersections
    g_final=exact_delaunay.Triangulation(extra_edge_fields=[
        #('dij',np.float64,2),
        #('ij',np.float64,2),
        ('angle',np.float64),
        ('psiphi',np.float64,2)])
    
    # g_final.edge_defaults['dij']=np.nan
    # Not great - when edges get split, this will at least leave the fields as nan
    # instead of 0.
    # g_final.edge_defaults['ij']=np.nan
    g_final.edge_defaults['psiphi']=np.nan
    g_final.edge_defaults['angle']=np.nan

    def trace_contour(b,angle):
        """
        angle: 0 is constant psi, with psi increasing to left
        """
        if angle==90:
            # trace constant phi
            node_field=self.phi # the field to trace a contour of
            cval_pos='right' # guess and check
        elif angle==270:
            node_field=self.phi # the field to trace a contour of
            cval_pos='left'
        elif angle==0:
            node_field=self.psi
            cval_pos='left' # guess and check
        elif angle==180:
            node_field=self.psi
            cval_pos='right'
        else:
            raise Exception("what?")
        cval=node_field[b]
        return g_int.trace_node_contour(n0=b,cval=cval,
                                        node_field=node_field,
                                        pos_side=cval_pos,
                                        return_full=True)

    # g_final node index =>
    # list of [
    #   ( dij, from the perspective of leaving the node,
    #     'internal' or 'boundary' )
    node_exits=defaultdict(list)

    def insert_contour(trace_items,angle=None,
                       psiphi0=[np.nan,np.nan]):
        assert np.isfinite(psiphi0[0]) or np.isfinite(psiphi0[1])

        trace_points=np.array( [pnt
                                for typ,idx,pnt in trace_items
                                if pnt is not None])

        # Check whether the ends need to be forced into the boundary
        # but here we preemptively doctor up the ends
        for i in [0,-1]:
            if trace_items[i][0]=='edge':
                # When it hits a non-cartesian edge this will fail (which is okay)
                # Feels a bit fragile:
                j_int=trace_items[i][1].j # it's a halfedge
                j_gen=g_int.edges['gen_j'][j_int] # from this original edge
                angle_gen=self.gen.edges['angle'][j_gen]
                if angle_gen%90 != 0:
                    print("Not worrying about contour hitting diagonal")
                    continue

                # Force that point into an existing constrained edge of g_final
                pnt=trace_points[i]
                best=[None,np.inf]
                for j in np.nonzero(g_final.edges['constrained'] & (~g_final.edges['deleted']))[0]:
                    d=utils.point_segment_distance( pnt,
                                                    g_final.nodes['x'][g_final.edges['nodes'][j]] )
                    if d<best[1]:
                        best=[j,d]
                j,d=best
                # Typ. 1e-10 when using UTM coordinates
                assert d<1e-5
                if d>0.0:
                    n_new=g_final.split_constraint(x=pnt,j=j)

        trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert',
                                                                   on_exists='stop')
        if angle is not None:
            g_final.edges['angle'][trace_edges]=angle
        #if ij0 is not None:
        #    g_final.edges['ij'][trace_edges]=ij0
        if psiphi0 is not None:
            g_final.edges['psiphi'][trace_edges]=psiphi0

        # Update node_exits:
        exit_angle=angle
        for a in trace_nodes[:-1]:
            node_exits[a].append( (exit_angle,'internal') )
        if angle is not None:
            angle=(angle+180)%360
        for b in trace_nodes[1:]:
            node_exits[b].append( (exit_angle,'internal') )

    def trace_and_insert_contour(b,angle):
        # does dij_angle fall between the angles formed by the boundary, including
        # a little slop.
        print(f"{angle} looks good")
        gn=fixed_int_to_gen[b] # below we already check to see that b is in there.

        # ij0=self.gen.nodes['ij'][gn].copy()
        # only pass the one constant along the contour
        if angle%180==0:
            psiphi0=[self.psi[b],np.nan]
        elif angle%180==90:
            psiphi0=[np.nan,self.phi[b]]

        trace_items=trace_contour(b,angle=angle)
        return insert_contour(trace_items,angle=angle,
                              psiphi0=psiphi0)

    def trace_and_insert_boundaries(cycle):
        for a,b in utils.progress( zip( cycle, np.roll(cycle,-1) )):
            j=g_int.nodes_to_edge(a,b)
            angle=g_int.edges['angle'][j] # angle from a to b
            if angle%90!=0: continue # ragged edge
            
            trace_points=g_int.nodes['x'][[a,b]]
            trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert')
            g_final.edges['angle'][trace_edges]=angle

            # Update node_exits, which are referenced by nodes in g_final
            for a_fin in trace_nodes[:-1]:
                node_exits[a_fin].append( (angle,'boundary') )
            opp_angle=(angle+180)%360
            for b_fin in trace_nodes[1:]:
                node_exits[b_fin].append( (opp_angle,'boundary') )

            # This used to also fill in ij, but we don't have that now.
            # need to update psiphi for these edges, too.
            if angle%180==0: # psi constant
                psiphi=[self.psi[a],np.nan]
            elif angle%180==90:
                psiphi=[np.nan, self.phi[a]]
            else:
                assert False

            g_final.edges['psiphi'][trace_edges]=psiphi
            
    # Add boundaries when they coincide with contours
    cycle=g_int.boundary_cycle() # could be multiple eventually...

    print("Tracing boundaries...",end="")
    trace_and_insert_boundaries(cycle)
    print("done")

    # Need to get all of the boundary contours in first, then
    # return with internal.
    for a,b,c in zip(cycle,
                     np.roll(cycle,-1),
                     np.roll(cycle,-2)):
        # if b==290: # side-channel
        #     g_int.plot_nodes(mask=g_int.nodes['rigid'],labeler='id')
        #     g_int.plot_nodes(mask=[a,c], labeler='id')
        #     g_int.plot_edges(mask=[j_ab,j_bc],labeler='angle')
        #     import pdb
        #     pdb.set_trace()
            
        if b not in fixed_int_to_gen: continue

        j_ab=g_int.nodes_to_edge(a,b)
        j_bc=g_int.nodes_to_edge(b,c)
        # flip to be the exit angle
        angle_ba = (180+g_int.edges['angle'][j_ab])%360
        angle_bc = g_int.edges['angle'][j_bc]
        
        for angle in [0,90,180,270]:
            # is angle into the domain?
            trace=None

            # if angle is left of j_ab and right of j_bc,
            # then it should be into the domain and can be traced
            # careful with sting angles
            # a,b,c are ordered CCW on the cycle, domain is to the
            # left.
            # so I want bc - angle - ba to be ordered CCW
            if ( ((angle_bc==angle_ba) and (angle!=angle_bc))
                 or (angle-angle_bc)%360 < ((angle_ba-angle_bc)%360) ):
                b_final=g_final.select_nodes_nearest(g_int.nodes['x'][b],max_dist=0.0)
                dupe=False
                if b_final is not None:
                    for exit_angle,exit_type in node_exits[b_final]:
                        if exit_angle==angle:
                            dupe=True
                            print("Duplicate exit for internal trace from %d. Skip"%b)
                            break
                if not dupe:
                    trace_and_insert_contour(b,angle)

    def tri_to_grid(g_final):
        g_final2=g_final.copy()

        for c in g_final2.valid_cell_iter():
            g_final2.delete_cell(c)

        for j in np.nonzero( (~g_final2.edges['deleted']) & (~g_final2.edges['constrained']))[0]:
            g_final2.delete_edge(j)

        g_final2.modify_max_sides(2000)
        g_final2.make_cells_from_edges()
        return g_final2

    g_final2=tri_to_grid(g_final)

    if 1: 
        # Add any diagonal edges here, so that ragged edges
        # get cells in g_final2, too.
        ragged=np.isfinite(g_int.edges['angle']) & (g_int.edges['angle']%90!=0.0)
        j_ints=np.nonzero( ragged )[0]
        for j_int in j_ints:
            nodes=[g_final2.add_or_find_node(g_int.nodes['x'][n])
                   for n in g_int.edges['nodes'][j_int]]
            j_fin2=g_final2.nodes_to_edge(nodes)
            angle=g_int.edges['angle'][j_int]
            if j_fin2 is None:
                j_fin2=g_final2.add_edge(nodes=nodes,constrained=True,
                                         angle=angle,
                                         psiphi=[np.nan,np.nan])

        g_final2.make_cells_from_edges()

    # Patch grid g_final2 completed.
    # fixed the missing the ragged edge.

    plt.clf()
    g_final2.plot_edges()
    plt.draw()
    
    #import pdb
    #pdb.set_trace()

    # --- Compile Swaths ---
    e2c=g_final2.edge_to_cells(recalc=True)

    i_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)
    j_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)

    for j in g_final2.valid_edge_iter():
        c1,c2=e2c[j,:]
        if c1<0 or c2<0: continue

        # if the di of dij is 0, the edge joins cell in i_adj
        # I think angle==0 is the same as dij=[1,0]
        
        #if g_final2.edges['dij'][j,0]==0:
        if g_final2.edges['angle'][j] % 180==0: # guess failed.
            i_adj[c1,c2]=i_adj[c2,c1]=True
        elif g_final2.edges['angle'][j] % 180==90:
            j_adj[c1,c2]=j_adj[c2,c1]=True
        else:
            print("What? Ragged edge okay, but it shouldn't have both cell neighbors")

    n_comp_i,labels_i=sparse.csgraph.connected_components(i_adj.astype(np.int32),directed=False)
    n_comp_j,labels_j=sparse.csgraph.connected_components(j_adj,directed=False)

    # preprocessing for contour placement
    nd=NodeDiscretization(g_int)
    Mdx,Bdx=nd.construct_matrix(op='dx')
    Mdy,Bdy=nd.construct_matrix(op='dy')
    psi_dx=Mdx.dot(self.psi)
    psi_dy=Mdy.dot(self.psi)
    phi_dx=Mdx.dot(self.phi)
    phi_dy=Mdy.dot(self.phi)

    # These should be about the same.  And they are, but
    # keep them separate in case the psi_phi solution procedure
    # evolves.
    psi_grad=np.sqrt( psi_dx**2 + psi_dy**2)
    phi_grad=np.sqrt( phi_dx**2 + phi_dy**2)

    pp_grad=[psi_grad,phi_grad]

    # Just figures out the contour values and sets them on the patches.
    patch_to_contour=[{},{}] # coord, cell index=>array of contour values

    def add_swath_contours_new(comp_cells,node_field,coord,scale):
        # Check all of the nodes to find the range ij
        comp_nodes=[ g_final2.cell_to_nodes(c) for c in comp_cells ]
        comp_nodes=np.unique( np.concatenate(comp_nodes) )
        comp_ijs=[] # Certainly could have kept this info along while building...

        field_values=[]

        plt.figure(2).clf()
        g_final2.plot_edges(color='k',lw=0.5)
        g_final2.plot_cells(mask=comp_cells)
        # g_final2.plot_nodes(mask=comp_nodes)
        plt.draw()
        
        # comp_ij=np.array(g_final2.edges['ij'][ g_final2.cell_to_edges(comp_cells[0]) ])
        comp_pp=np.array(g_final2.edges['psiphi'][ g_final2.cell_to_edges(comp_cells[0]) ])

        # it's actually the other coordinate that we want to consider.
        field_min=np.nanmin( comp_pp[:,1-coord] )
        field_max=np.nanmax( comp_pp[:,1-coord] )

        # coord_min=np.nanmin( comp_ij[:,1-coord] )
        # coord_max=np.nanmax( comp_ij[:,1-coord] )

        # Could do this more directly from topology if it mattered..
        swath_poly=ops.cascaded_union( [g_final2.cell_polygon(c) for c in comp_cells] )
        swath_nodes=g_int.select_nodes_intersecting(swath_poly)
        swath_vals=node_field[swath_nodes]
        swath_grad=pp_grad[1-coord][swath_nodes] # right?
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

        # HERE -- calculate this from resolution
        # might have i/j swapped.  range of s is 77m, and field
        # is 1. to 1.08.  better now..
        local_scale=scale( g_int.nodes['x'][swath_nodes] ).mean(axis=0)
        n_swath_cells=int(np.round( (s.max() - s.min())/local_scale))
        n_swath_cells=max(1,n_swath_cells)
        
        s_contours=np.linspace(s[0],s[-1],1+n_swath_cells)
        adj_contours=np.interp( s_contours,
                                s,o_vals)
        adj_contours[0]=field_min
        adj_contours[-1]=field_max

        for c in comp_cells:
            patch_to_contour[coord][c]=adj_contours

    if 1: # Swath processing
        # TODO replace with a real pair of field
        self.scales=[field.ConstantField(self.nom_res),
                     field.ConstantField(self.nom_res)]
        
        for coord in [0,1]: # i/j
            print("Coord: ",coord)
            if coord==0:
                labels=labels_i
                n_comp=n_comp_i
                node_field=self.phi # feels backwards.. it's right, just misnamed 
            else:
                labels=labels_j
                n_comp=n_comp_j
                node_field=self.psi

            for comp in range(n_comp):
                print("Swath: ",comp)
                comp_cells=np.nonzero(labels==comp)[0]
                add_swath_contours_new(comp_cells,node_field,coord,self.scales[coord])

    # Direct grid gen from contour specifications:

    @utils.add_to(g_int)
    def fields_to_xy(self,target,node_fields,x0):
        """
        target: values of node_fields to locate
        x0: starting point

        NB: edges['cells'] must be up to date before calling
        """
        c=self.select_cells_nearest(x0)

        while 1:
            c_nodes=self.cell_to_nodes(c)
            M=np.array( [ node_fields[0][c_nodes],
                          node_fields[1][c_nodes],
                          [1,1,1] ] )
            b=[target[0],target[1],1.0]

            weights=np.linalg.solve(M,b)
            if min(weights)<0: # not there yet.
                min_w=np.argmin(weights)
                c_edges=self.cell_to_edges(c,ordered=True)# nodes 0--1 is edge 0, ...
                sel_j=c_edges[ (min_w+1)%(len(c_edges)) ]
                edges=self.edges['cells'][sel_j]
                if edges[0]==c:
                    next_c=edges[1]
                elif edges[1]==c:
                    next_c=edges[0]
                else:
                    raise Exception("Fail.")
                if next_c<0:
                    if weights.min()<-1e-5:
                        print("Left triangulation (min weight: %f)"%weights.min())
                        # Either the starting cell didn't allow a simple path
                        # to the target, or the target doesn't fall inside the
                        # grid (e.g. ragged edge)
                        return [np.nan,np.nan]
                    # Clip the answer to be within this cell (will be on an edge
                    # or node).
                    weights=weights.clip(0)
                    weights=weights/weights.sum()
                    break
                c=next_c
                continue
            else:
                break
        x=(self.nodes['x'][c_nodes]*weights[:,None]).sum(axis=0)
        return x

    patch_grids=[]

    g_int.edge_to_cells()

    for c in utils.progress(g_final2.valid_cell_iter()):
        psi_cvals=patch_to_contour[1][c]
        phi_cvals=patch_to_contour[0][c]

        g_patch=unstructured_grid.UnstructuredGrid(max_sides=4)
        g_patch.add_rectilinear( [0,0], [len(psi_cvals)-1,len(phi_cvals)-1],
                                 len(psi_cvals),len(phi_cvals))
        g_patch.add_node_field( 'ij', g_patch.nodes['x'].astype(np.int32))
        pp=np.c_[ psi_cvals[g_patch.nodes['ij'][:,0]],
                  phi_cvals[g_patch.nodes['ij'][:,1]] ]
        g_patch.add_node_field( 'pp', pp)

        x0=g_final2.cells_centroid([c])[0]
        for n in g_patch.valid_node_iter():
            x=g_int.fields_to_xy(g_patch.nodes['pp'][n],
                                 node_fields=[self.psi,self.phi],
                                 x0=x0)
            if np.isnan(x[0]):
                # If it's a ragged cell, probably okay.
                edge_angles=g_final2.edges['angle'][ g_final2.cell_to_edges(c) ]
                ragged_js=(edge_angles%90!=0.0)
                if np.any(ragged_js):
                    print("fields_to_xy() failed, but cell is ragged.")
                    g_patch.delete_node_cascade(n)
                else:
                    print("ERROR: fields_to_xy() failed. Cell not ragged.")
            g_patch.nodes['x'][n]=x
            # Hmm -
            # When it works, this probably reduces the search time considerably,
            # but there is the possibility, particularly at corners, that
            # this x will be a corner, that corner will lead to the cell *around*
            # the corner, and then we get stuck.
            # Even the centroid isn't great since it might not even fall inside
            # the cell.
            # x0=x 
        patch_grids.append(g_patch)

    g=patch_grids[0]
    for g_next in patch_grids[1:]:
        g.add_grid(g_next,merge_nodes='auto',tol=1e-6)

    return g

qg.g_final=qg.create_final_by_patches()

##
g=qg.g_final

plt.figure(4).clf()
fig,ax=plt.subplots(num=4)
g.plot_edges(color='k',lw=0.5)
g.plot_cells(color='0.8',lw=0,zorder=-2)

ax.axis('off')
ax.set_position([0,0,1,1])
##

g.write_ugrid('by_angle_output.nc')
