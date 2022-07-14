from stompy.grid import unstructured_grid

g=unstructured_grid.UnstructuredGrid.read_ugrid('quad_tri_v21-edit41.nc')

##
e2c=g.edge_to_cells(recalc=True)
Ac=g.cells_area()
cc=g.cells_center()

removesmalllinkstrsh=0.1 # ??

bad_links=[]
for j in g.valid_edge_iter():
    c1,c2 = e2c[j,:]
    if c1<0 or c2<0: continue # assuming they only care about internal edges
    
    dxlim  = 0.90*removesmalllinkstrsh*0.5*(np.sqrt(Ac[c1]) + np.sqrt(Ac[c2]))
    dxlink = utils.dist( cc[c1], cc[c2])
    if dxlink<dxlim:
        bad_links.append(j)
print(f"Bad links:")
print(g.edges_center()[bad_links])
        
    
