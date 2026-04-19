# DNDSR Ideas {#ideas}

## Next Step:

- [x] wrap mesh for better access
- [ ] check dist topo and comm with a global BFS data
- [x] start CFV methods
- [x] check on general json parsing
- [x] octree and periodic bc ~~(using CGAL)~~ using nanoflann
- [x] mesh reader: CGNS abutting interface with 1-to-1 face
- [x] CFV limiters
- [x] Euler Solver
- [x] Facial Value output in Euler Solver
- [x] kdtree for euler solver's wall dist finding
- [ ] mesh: automatic global refinement
- [ ] reorganize comm strategy and communication callings
- [x] more bases in CFV
- [x] more functional in CFV
- [x] fully serial mesh partitioning
- [ ] try scotch?
- [x] serialization of euler solver
- [x] restarting of euler solver
- [x] data post-process utility (with VTK?)
- [ ] wrap the mesh and vfv into iterator or range based for
- [ ] wrap array into iterator or range based for
- [ ] wrap derived arrays into iterator or range based for


## About Periodic:

- periodic is defined with bnd faces
- periodic makes faces 2x duplicate, and nodes 8x duplicate max

steps in doing periodic ?: 

at least in serial reader:

meshinfo:

- coordSerial
- cell2nodeSerial
- bnd2nodeSerial
- cellElemInfoSerial
- bndElemInfoSerial
- bnd2cellSerial

well, when we have this, we need to add something:

- 1. detect face-to-face_donor
- 2. decide node de-duplication, and get new set of nodes_deduplicated
- 3. alter cell2node/bnd2node to point to correct nodes_deduplicated
- 4. coord now stores nodes_deduplicated
- partitioned:
- 5. use faceID to detect cells that are affected by periodic, record them
- 6. mask any mesh->coord[iNode] queries
- 7. in cfv, mask any coord-related (center, quadrature...) queries

the intention is to make topology unique and complete, treating faces with periodic like the inner ones

to ease cell2node query, in parallel:

- create appended physcial_coord
- for affected cells, convert cell2node(pointing to local) to point to physcial_coord
 (with minus meaning the physical_coord indices) (need a new adj state)
- create means of inverting the process (to get original cell2node(pointing to local) topo)
- when printing vis files, physcial_coord should be appended, and cell2node need to point to those, and coord data on physical_coord should copy those of original coord
- mesh serialization state should be unaffected by periodic (not pointing to appended, or the physical_node_to_actual_node mapping should also be serialized)


### findings:

if the periodic mesh is 2xN cell (3x(N+1)nodes), the current face interpolation makes wrong face findings 
(with less faces created than needed;); this thing could be ignored for now?

### new rules:

periodic donor and main must be non-adjacent (not sharing nodes) (main-donor node mapping each pair is not identical) [reasoning: if periodicity needs this, means singularity point, replace this point with a small face with sym condition]

then, record cell2nodePi, and face2nodePi, to augment cell2node ..., and:

nodePi: bit1-bit2-bit3, means [if peri1][if peri2][if peri3]

in the sense of peri-duplicated mesh, node in a cell is different when both cell2node(iCell,ic2n) and cell2nodePI(iCell,ic2n) are different; so use both cell2node and cell2nodePi for inner-cell coord calculation

in the sense of de-duplicated mesh, cell2node(iCell,ic2n) is unique, and points to a coord; 

to get interpolate face, the faces are the same when both are true: 
- face2node are the same,
- face2nodePi are collaborating: the xor (face2nodePiL_if2n, face2nodePiR_if2n) are the same

face2nodePi are got from the first cell, like face2node
faceAtr are got from the first cell, (for de-duplicated faces, could be periodic-donor or main)

to query face-coords in cell, check is is periodic, if donor, face2cell[1] needs the face to trans-back; if main, face2cell[1] needs the face to trans; face2cell[0] is always good

to query other cell-coord through face, if face is periodic main, if cell is face2cell[0] then other cell needs trans back; cell is face2cell[1] then other cell needs trans 


## About Exporting Static Interfaces

Use pybind11? 

Array objects for exporting:

- MPI Top interface
- Adj array (non uniform)
- 3d coord array
- UDof array
- URec array
- Eigen Vec / Mat / Mats array ?
- Rec Matrices array ?

## About Increasing Performance:

- using vectorized Riemann solver / face flux
- using face - single - dissipation Riemann Solver
- 4-cached DBV
  
- SGS init with last du
- start with lower order
- CFL adapt
- RCM? reordering

## mesh elevator

- could just use serial
- parallel: need some control on bnd-bnd / node-bnd
- geometrical edge detection: 1. each node has set of norms on bnds; 2. on edge: see if norms of a endpoint is close to normal to the edge(or else dispose of it), which means filtering out those deviates the edge norm too much
- rbf field: nodes include original bnd nodes and new bnd nodes
