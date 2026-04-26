# Geometry + CFV Usage Guide {#geom_usage}

@tableofcontents

This guide takes you from a raw CGNS file to a running PDE solver on
top of the DNDSR infrastructure.  It covers three layers:

1. **Geom** (`src/Geom/`) -- distributed unstructured mesh, topology,
   node coordinates, element/quadrature access.
2. **CFV::FiniteVolume** (`src/CFV/FiniteVolume.hpp`) -- geometric
   integration support: cell volumes, face areas, Jacobian determinants,
   quadrature metrics.
3. **CFV::VariationalReconstruction** (`src/CFV/VariationalReconstruction.hpp`)
   -- polynomial basis, reconstruction matrices, iterative VR solver,
   limiters.

For array concepts (father/son, ghost communication) see
@ref array_infrastructure.

## When to Use Which Layer

| You are writing... | Minimum layer you need |
|---|---|
| A mesh-manipulation tool (no PDE) | `Geom::UnstructuredMesh` |
| A first-order finite volume solver | `Geom` + `CFV::FiniteVolume` |
| A high-order FV solver with polynomial reconstruction | `Geom` + `CFV::FiniteVolume` + `CFV::VariationalReconstruction` |
| A new PDE on an existing scheme | plug your physics into a `ModelEvaluator` subclass |

## Part 1: Mesh Building

A CFD solver needs topology (cells, nodes, faces), geometry (coords),
quadrature, and MPI distribution.  Phase A gets you the mesh; phases
B--D add the CFV layer.

### C++ Pipeline (Phase A)

Each step is explained with *why* it is needed.
(Source: `Geom/Mesh/Mesh.hpp` for declarations;
`Geom/Mesh/Mesh_Serial_ReadFromCGNS.cpp` for the CGNS reader implementation.)

```cpp
#include "Geom/Mesh/Mesh.hpp"
using namespace DNDS;
using namespace DNDS::Geom;

MPIInfo mpi;
mpi.setWorld();

int dim = 2;
auto mesh = std::make_shared<UnstructuredMesh>(mpi, dim);
UnstructuredMeshSerialRW reader(mesh, 0);

// --- Step 1: Read CGNS on rank 0 ---
// Populates serial arrays: cell2node, coords, cellElemInfo, etc.
// Only rank 0 does actual I/O; other ranks wait at the barrier.
reader.ReadFromCGNSSerial("mesh.cgns");

// --- Step 2 (optional): Periodic boundary deduplication ---
// Merges coincident periodic nodes so that the mesh connectivity
// treats periodic boundaries as internal faces with a transform.
reader.Deduplicate1to1Periodic(/*eps=*/1e-9);

// --- Step 3: Build cell-to-cell adjacency ---
// Two cells are neighbors if they share at least one node.
// This is the graph that METIS will partition.
reader.BuildCell2Cell();

// --- Step 4: Partition with METIS ---
// Splits cells across MPI ranks, minimizing edge cuts.
UnstructuredMeshSerialRW::PartitionOptions opts;
opts.metisSeed = 42;   // deterministic for testing
reader.MeshPartitionCell2Cell(opts);

// --- Step 5: Distribute to all ranks ---
// Scatters serial arrays into distributed father arrays.
// After this, each rank owns a slice of the cells and nodes.
reader.PartitionReorderToMeshCell2Cell();

// --- Step 6: Build inverse relations ---
// node2cell, node2bnd are needed by Step 7.
mesh->RecoverNode2CellAndNode2Bnd();
mesh->RecoverCell2CellAndBnd2Cell();

// --- Step 7: Build ghost layers ---
// For each local cell, its cell2cell neighbors on other ranks
// become ghosts.  AdjGlobal2LocalPrimary converts global indices
// to local father+son indices.
mesh->BuildGhostPrimary();
mesh->AdjGlobal2LocalPrimary();

// --- Step 8: Build face arrays ---
// InterpolateFace creates face2node, face2cell, cell2face, bnd2face
// from cell-to-node and boundary connectivity.  Needed by any
// finite-volume scheme that loops over faces.
mesh->InterpolateFace();
mesh->AssertOnFaces();
```

After these 8 steps the mesh is ready.  Optional steps
(order elevation, h-bisection) can be inserted between steps 7 and 8;
see `Mesh/Mesh.hpp` for `BuildO2FromO1Elevation` and
`BuildBisectO1FormO2`.

### Python Pipeline

The helper function `create_mesh_from_CGNS` wraps the entire 8-step
pipeline:

```python
from DNDSR import DNDS, Geom
from DNDSR.Geom.utils import create_mesh_from_CGNS

mpi = DNDS.MPIInfo()
mpi.setWorld()

mesh, reader, name2ID = create_mesh_from_CGNS(
    meshFile="data/mesh/NACA0012.cgns",
    mpi=mpi,
    dim=2,
    meshElevation="O2",      # optional: elevate O1 -> O2
    meshDirectBisect=1,      # optional: bisect once for h-refinement
)
```

`name2ID` maps boundary zone names (from the CGNS file) to integer IDs,
which you use when setting boundary conditions.
(Source: `python/DNDSR/Geom/utils.py`.)

## Part 2: Mesh Topology -- Arrays and Access Patterns

After building, the mesh provides these `ArrayPair` objects.  Father
stores owned entities; son stores ghosts from neighboring ranks.

### Primary Connectivity

(Source: `Geom/Mesh/Mesh.hpp`.)

| Array | Row entity | Column content | Row width |
|---|---|---|---|
| `cell2node` | cell | node indices | variable (3 for tri, 4 for quad, ...) |
| `cell2cell` | cell | neighbor cell indices | variable |
| `bnd2cell` | boundary face | `[innerCell, ghostCell]` | fixed 2 |
| `coords` | node | `[x, y, z]` | fixed 3 |
| `cellElemInfo` | cell | `ElemInfo` struct | fixed 1 |

### Face Connectivity (After `InterpolateFace`)

(Source: `Geom/Mesh/Mesh.hpp`.)

| Array | Row entity | Column content | Row width |
|---|---|---|---|
| `face2node` | face | node indices | variable |
| `face2cell` | face | `[leftCell, rightCell]` | fixed 2 |
| `cell2face` | cell | face indices | variable |
| `bnd2face` | boundary | face index | fixed 1 |

### Size Queries

(Source: `Geom/Mesh/Mesh.hpp`.)

```cpp
mesh->NumCell()        // owned cells
mesh->NumCellGhost()   // ghost cells
mesh->NumCellProc()    // owned + ghost (total on this rank)
mesh->NumCellGlobal()  // sum across all ranks (collective MPI call)
// Same pattern for NumNode, NumFace, NumBnd.
```

### Traversal Patterns

**Looping over all cells including ghosts** -- useful for reconstruction
passes that need to write into ghost cells too:

```cpp
for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
{
    // iCell < NumCell()   -> owned
    // iCell >= NumCell()  -> ghost
}
```

**Looping over faces** -- the standard finite-volume flux pattern:

```cpp
for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
{
    index cellL = mesh->face2cell(iFace, 0);
    index cellR = mesh->face2cell(iFace, 1);
    // cellR == UnInitIndex for boundary faces at the domain edge.
}
```

**Looping over a cell's faces**:

```cpp
for (rowsize iF = 0; iF < mesh->cell2face.RowSize(iCell); iF++)
{
    index iFace = mesh->cell2face(iCell, iF);
    // ... compute face flux, accumulate into cell RHS ...
}
```

## Part 3: Element Types and Node Coordinates

Each cell, face, and boundary carries an `ElemInfo` that records its
element type.  The `Element` struct gives shape functions and geometric
queries.  (Source: `Geom/Elements.hpp:186` for `Element`;
`Geom/ElemEnum.hpp:18` for `ElemType`.)

### Getting the Element Object

```cpp
Elem::Element elem = mesh->GetCellElement(iCell);

int nNodes = elem.GetNumNodes();     // 9 for Quad9
int nVerts = elem.GetNumVertices();  // 4 for Quad9
int dim    = elem.GetDim();          // 2 for Quad, 3 for Hex
int order  = elem.GetOrder();        // 1 linear, 2 quadratic
int nFaces = elem.GetNumFaces();     // 4 for Quad, 6 for Hex
```

### Supported Element Types

| Enum | Name | Nodes | Dim | Order |
|---|---|---|---|---|
| `Line2` | Linear edge | 2 | 1 | 1 |
| `Tri3` | Triangle | 3 | 2 | 1 |
| `Quad4` | Quad | 4 | 2 | 1 |
| `Tet4` | Tet | 4 | 3 | 1 |
| `Hex8` | Hex | 8 | 3 | 1 |
| `Prism6` | Prism | 6 | 3 | 1 |
| `Pyramid5` | Pyramid | 5 | 3 | 1 |
| `Tri6` | Quadratic tri | 6 | 2 | 2 |
| `Quad9` | Quadratic quad | 9 | 2 | 2 |
| `Tet10` | Quadratic tet | 10 | 3 | 2 |
| `Hex27` | Quadratic hex | 27 | 3 | 2 |

### Node Coordinates

Coordinates are always 3D (`Eigen::Vector3d`), even for 2D meshes
(z=0).  `GetCoordsOnCell` (in `Geom/Mesh/Mesh.hpp`) applies periodic
coordinate transformations when a cell straddles a periodic boundary;
always use it instead of manually reading `coords[cell2node[iCell][j]]`
unless you know the mesh is not periodic.

```cpp
tPoint p = mesh->coords[iNode];   // Eigen::Vector3d

Eigen::Matrix<real, 3, Eigen::Dynamic> cs;
mesh->GetCoordsOnCell(iCell, cs);
// cs.col(j) = coordinate of the j-th node of cell iCell
```

## Part 4: Raw Quadrature (Without CFV)

For simple geometric computations (e.g. computing cell volumes yourself)
you can use the `Quadrature` class directly.  For a production solver,
however, CFV caches all of this, and you should use CFV's accessors
(Part 6 below).

(Source: `Geom/Quadrature.hpp:96`.)

```cpp
#include "Geom/Quadrature.hpp"

Elem::Element elem = mesh->GetCellElement(iCell);
Elem::Quadrature quad(elem, /*intOrder=*/3);
// Quad4 + order 3: 2x2 Gauss-Legendre = 4 points.
// Tri3  + order 3: 6 symmetric Hammer points.

Geom::tSmallCoords cs;
mesh->GetCoordsOnCell(iCell, cs);

real volume = 0;
quad.Integration(
    volume,
    [&](real &acc, int iG, tPoint pParam, Elem::tD01Nj D01Nj)
    {
        // D01Nj: (1+dim) x nNodes
        //   row 0:    N_j(xi)       shape function values
        //   rows 1-dim: dN_j/d(xi_k) parametric derivatives

        // CellJacobianDet takes the full D01Nj (it knows which rows
        // are derivatives based on the dim template parameter).
        acc = Elem::CellJacobianDet<2>(cs, D01Nj);
        // Integration accumulates: volume += acc * weight
    });
```

## Part 5: FiniteVolume -- Geometric Integration Support

The `CFV::FiniteVolume` class (`src/CFV/FiniteVolume.hpp:15`) holds
cached geometric data the solver uses every time step: cell volumes,
barycenters, face areas, Jacobian determinants at every quadrature
point, face normals, etc.  If you are writing a first-order FV solver,
this is the only CFV class you need.

For high-order schemes you use `VariationalReconstruction` instead,
which *inherits* from `FiniteVolume` and adds reconstruction matrices
(Part 6).

### Settings

The minimum configuration is `maxOrder` (polynomial order of
reconstruction) and `intOrder` (quadrature order).  Additional settings
(`CFV/FiniteVolumeSettings.hpp` for FV,
`CFV/VRSettings.hpp:32-239` for the VR-extended struct) control caching,
anisotropic functionals, limiter behavior, and more.

Minimum baseline:

```json
{ "maxOrder": 2, "intOrder": 5, "cacheDiffBase": true }
```

### Construction: Phase B (VR Settings)

For a bare-FV scheme (no reconstruction), construct a `FiniteVolume`
directly; for high-order, construct a `VariationalReconstruction<dim>`:

```cpp
#include "CFV/VariationalReconstruction.hpp"

auto vfv = std::make_shared<CFV::VariationalReconstruction<2>>(mpi, mesh);
vfv->parseSettings(vrSettingsJson);          // merge user overrides onto defaults

vfv->SetAxisSymmetric(0);                    // default; set to 1 for axisym flows
vfv->SetPeriodicTransformations(...);        // or SetPeriodicTransformationsNoOp() for scalars
```

The periodic-transformations argument tells VR how to rotate or reflect
vector variables (velocity components) across periodic boundaries.
For a scalar test field, the no-op variant (`SetPeriodicTransformationsNoOp`)
is correct.  See `VariationalReconstruction.hpp:228-255` for the
signature.

### Construction: Phase C (Geometric Metrics)

`ConstructMetrics()` runs all 15 geometric construction steps in the
correct order.  Each step populates a protected `ArrayPair`.
(Source: `VariationalReconstruction.cpp:7-44`; individual methods in
`FiniteVolume.cpp`.)

```cpp
vfv->ConstructMetrics();
// Internally runs, in order:
//   SetCellAtrBasic                     FiniteVolume.cpp:7
//   ConstructCellVolume                 :22
//   ConstructCellBary                   :176
//   ConstructCellCent                   :215
//   ConstructCellIntJacobiDet           :87
//   ConstructCellIntPPhysics            :144
//   ConstructCellAlignedHBox            :243
//   ConstructCellMajorHBoxCoordInertia  :271
//   SetFaceAtrBasic                     :338
//   ConstructFaceCent                   :398
//   ConstructFaceArea                   :353
//   ConstructFaceIntJacobiDet           :428
//   ConstructFaceIntPPhysics            :471
//   ConstructFaceUnitNorm               :501
//   ConstructFaceMeanNorm               :539
//   ConstructCellSmoothScale            :592
```

After `ConstructMetrics` you can query:

```cpp
vfv->GetCellVol(iCell);                  // cell volume
vfv->GetCellBary(iCell);                 // barycenter (Eigen::Vector3d)
vfv->GetFaceArea(iFace);                 // face area (length in 2D)
vfv->GetFaceNorm(iFace, iG);             // outward unit normal at quad pt iG
vfv->GetFaceNorm(iFace, -1);             // area-weighted mean normal
vfv->GetCellJacobiDet(iCell, iG);        // det(dx/dxi) at quad pt iG
vfv->GetFaceJacobiDet(iFace, iG);        // face Jacobian det at quad pt iG
vfv->GetCellQuad(iCell);                 // Quadrature object of order intOrder
vfv->GetFaceQuad(iFace);                 // same for a face
```

(Source: `FiniteVolume.hpp:198-229, 251-273`.)

### Python: Phase B + C

The Python driver from `test/CFV/test_vr_correctness.py:55-82` is the
concise equivalent:

```python
from DNDSR import DNDS, CFV
import json

vfv = CFV.VariationalReconstruction_2(mpi, mesh)   # 2 = spatial dim

settings = vfv.GetSettings()
settings.update({
    "maxOrder": 2, "intOrder": 5, "cacheDiffBase": True,
    "jacobiRelax": 1.0, "SORInstead": False,
    "functionalSettings": {
        "dirWeightScheme": "HQM_OPT",
        "geomWeightScheme": "HQM_SD",
        "geomWeightPower": 0.5,
        "geomWeightBias": 1,
    },
})
vfv.ParseSettings(settings)     # merge-patch onto defaults
vfv.SetPeriodicTransformationsNoOp()

vfv.ConstructMetrics()          # Phase C: all 15 geometric steps
```

## Part 6: VariationalReconstruction -- High-Order Machinery

Phases D--F build the reconstruction matrices.  These phases are only
needed if you want high-order (`maxOrder >= 2`).

### Phase D: Boundary Weights

The `ConstructBaseAndWeight` call needs a callback that tells VR
*which polynomial orders to constrain on each boundary zone*.  The
callback signature is
`real(Geom::t_index bcID, int iOrder) -> real` returning a weight.

The canonical pattern from `EulerEvaluator::InitializeFV`
(`EulerEvaluator.hpp:138-152`) dispatches on BC type:

```cpp
vfv->ConstructBaseAndWeight([&](Geom::t_index id, int iOrder) -> real {
    auto type = pBCHandler->GetTypeFromID(id);
    if (type == BCSpecial || type == BCOut) return 0;                // unconstrained
    if (type == BCFar)       return iOrder ? 0. : 1.;                // Dirichlet mean only
    if (type == BCWallInvis || type == BCSym)
                             return iOrder ? 0. : 1.;
    if (Geom::FaceIDIsPeriodic(id)) return 1.;                       // treat as internal
    return iOrder ? 0. : 1.;                                         // default Dirichlet
});
```

`iOrder == 0` is the mean-value constraint; `iOrder >= 1` are
derivative-order constraints.  Returning 0 drops that order from the
face functional.

In Python, pass a dict `{(bc_id, iOrder): weight}`:

```python
bc_weights = {(name2ID[name], 0): 1.0 for name in ("wall", "farfield")}
vfv.ConstructBaseAndWeight_map(bc_weights)
```

### Phase E: Reconstruction Coefficients

```cpp
vfv->ConstructRecCoeff();
```

This assembles per-cell matrices `matrixAAInvB` (face coefficients) and
`vectorAInvB` (RHS coefficients) used by the iterative VR solver.
(Source: `VariationalReconstruction.cpp:507-874`.)  After this call,
the VR object is ready for reconstruction.

### Phase F: Build DOF Arrays

Three DOF types for a typical second-order PDE:

```cpp
using CFV::tUDof;       // cell means:           nVars x 1
using CFV::tURec;       // rec coefficients:     (NDOF-1) x nVars
using CFV::tUGrad;      // gradients:            dim x nVars

tUDof<5> u;      vfv->BuildUDof (u, 1);      // 5 conservative vars
tURec<5> uRec;   vfv->BuildURec (uRec, 1);
tUGrad<5,2> grad; vfv->BuildUGrad(grad, 1);
```

`NDOF` (number of reconstruction DOFs) is determined by `maxOrder` and
`dim`: for 2D, `NDOF = 1, 3, 6, 10` for orders `0, 1, 2, 3`.  The
reconstruction array `tURec` stores `NDOF-1` rows because the mean is
kept separately in `tUDof`.

`BuildUDof` also sets up the ghost mapping (borrowing from
`mesh->cell2node.trans`) and initializes persistent MPI send/recv
requests so you can pull ghosts with two calls:

```cpp
u.trans.startPersistentPull();
u.trans.waitPersistentPull();
```

(Source: `CFV/DOFFactory.hpp:17-154` for the generic factory;
`VariationalReconstruction.hpp:807-891` for the typed builders.)

## Part 7: Writing a PDE Solver on Top of CFV

A PDE solver on top of the VR infrastructure consists of:

1. A **boundary callback** that returns the ghost state at each
   boundary quadrature point as a function of the interior state and
   boundary zone ID.
2. An **`EvaluateRHS`** method that loops over faces, reconstructs the
   state at each face quadrature point, computes a numerical flux, and
   accumulates into the cell RHS.

### The Boundary Callback `FBoundary`

Signature (`VariationalReconstruction.hpp:893-902`):

```cpp
TFBoundary<nVarsFixed> = std::function<
    Eigen::Vector<real, nVarsFixed>(
        const Eigen::Vector<real, nVarsFixed>& UBL,     // extrapolated state at face quad pt
        const Eigen::Vector<real, nVarsFixed>& UMEAN,   // cell-mean state
        index iCell, index iFace, int ig,
        const Geom::tPoint& norm,                        // outward unit normal
        const Geom::tPoint& pPhy,                        // physical coords of quad pt
        Geom::t_index fType                              // boundary zone ID
    )>;
```

You return the **ghost state** (the boundary-imposed value of `u`) as
a function of the interior state plus geometry plus zone ID.  VR uses
this both during reconstruction (in `GetBoundaryRHS`, see
`VariationalReconstruction_Reconstruction.hxx:19-96`) and during the
flux evaluation for the "right" state at boundary faces.

Typical structure (production example in `Euler/EulerEvaluator.hpp:1353`,
`generateBoundaryValue`):

```cpp
auto FBoundary = [this, t](const TU& UBL, const TU& UMEAN,
                            index iCell, index iFace, int iG,
                            const tPoint& norm, const tPoint& pPhy,
                            Geom::t_index fType) -> TU
{
    switch (bcHandler->GetTypeFromID(fType))
    {
        case BCWallInvis: /* reflect normal velocity */         return ...;
        case BCFar:       /* far-field Riemann invariants */    return ...;
        case BCOut:       /* pressure outlet */                 return ...;
        default:          return UMEAN;
    }
};
```

### The RHS Loop

This is the heart of your solver.  The minimal template, following
`ModelEvaluator::EvaluateRHS` (`src/CFV/ModelEvaluator.cpp:6-115`):

```cpp
void MyEvaluator::EvaluateRHS(tUDof<nV> &rhs, const tUDof<nV> &u,
                              const tURec<nV> &uRec, real t)
{
    // Clear owned RHS.
    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        rhs[iCell].setZero();

    // Loop over all faces (owned + ghost).
    for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
    {
        auto f2c = mesh->face2cell[iFace];
        auto fQuad = vfv->GetFaceQuad(iFace);

        Eigen::Matrix<real, nV, 1> flux_accum = Eigen::Matrix<real, nV, 1>::Zero();

        fQuad.IntegrationSimple(flux_accum,
            [&](auto &finc, int iG, real w)
            {
                // 1. Normal at this quadrature point.
                tVec n = vfv->GetFaceNorm(iFace, iG)(Seq012);

                // 2. Reconstruct left state at quad pt.
                //    GetIntPointDiffBaseValue returns a (1 x NDOF-1) row of
                //    basis values; multiplied by uRec gives the polynomial value.
                TU UL = u[f2c[0]] +
                    (vfv->GetIntPointDiffBaseValue(
                        f2c[0], iFace, /*if2c=*/0, iG,
                        std::array<int,1>{0}, /*maxDiff=*/0) * uRec[f2c[0]]
                    ).transpose();

                // 3. Reconstruct right state (or apply BC).
                TU UR;
                if (f2c[1] != UnInitIndex)
                {
                    UR = u[f2c[1]] +
                        (vfv->GetIntPointDiffBaseValue(
                            f2c[1], iFace, /*if2c=*/1, iG,
                            std::array<int,1>{0}, 0) * uRec[f2c[1]]
                        ).transpose();
                }
                else
                {
                    // Boundary face: apply FBoundary.
                    tPoint pPhy = vfv->GetFaceQuadraturePPhys(iFace, iG);
                    UR = FBoundary(UL, u[f2c[0]], f2c[0], iFace, iG,
                                   vfv->GetFaceNorm(iFace, iG), pPhy,
                                   mesh->GetFaceZone(iFace));
                }

                // 4. Your numerical flux (Roe, HLLC, LF, ...)
                finc = myRiemannSolver(UL, UR, n);

                // 5. Multiply by face Jacobian determinant.
                finc *= vfv->GetFaceJacobiDet(iFace, iG);
            });

        // Accumulate into cell RHS with appropriate signs and volumes.
        rhs[f2c[0]] += flux_accum / vfv->GetCellVol(f2c[0]);
        if (f2c[1] != UnInitIndex)
            rhs[f2c[1]] -= flux_accum / vfv->GetCellVol(f2c[1]);
    }

    // (Optional) volume source integral: another loop using GetCellQuad.
}
```

Key VR accessors used in the loop:

| Call | Returns | Purpose |
|---|---|---|
| `GetFaceQuad(iFace)` | `Quadrature` | Face quadrature rule (order `faceAtr.intOrder`) |
| `GetFaceNorm(iFace, iG)` | `Vector3d` | Outward unit normal at quad pt iG (-1 = mean) |
| `GetFaceJacobiDet(iFace, iG)` | `real` | Metric at quad pt |
| `GetFaceQuadraturePPhys(iFace, iG)` | `Vector3d` | Physical coords of quad pt |
| `GetIntPointDiffBaseValue(iC, iF, if2c, iG, diffs, maxD)` | `MatrixXR` | Basis values/derivatives at quad pt.  Multiplied by `uRec[iCell]` to get the polynomial value or derivative there. |
| `GetCellVol(iCell)` | `real` | Cell volume |
| `GetFaceArea(iFace)` | `real` | Face area |

(Sources: `FiniteVolume.hpp:198-281`, `VariationalReconstruction.hpp:378-442`.)

For periodic meshes, use the `*FromCell` variants
(`GetFaceNormFromCell`, `GetOtherCellBaryFromCell`) which apply the
periodic transform automatically.  See `FiniteVolume.hpp:259-273,
319-337`.

## Part 8: The Full Solver Loop

Per physical time step, a typical RK-style solver does:

```cpp
for (int iRK = 0; iRK < nStages; iRK++)
{
    // 1. Synchronize ghost cell means.
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();

    // 2. Iterative variational reconstruction.
    //    Multiple SOR sweeps converge uRec from the current u.
    for (int iVR = 0; iVR < nVRIter; iVR++)
    {
        vfv->DoReconstructionIter(uRec, uRecNew, u, FBoundary,
                                  /*putIntoNew=*/false, /*recordInc=*/false,
                                  /*uRecIsZero=*/(iVR == 0));
    }
    uRec.trans.startPersistentPull();
    uRec.trans.waitPersistentPull();

    // 3. (Optional) shock-capturing limiter.
    vfv->DoCalculateSmoothIndicator(si, u, uRec);
    vfv->DoLimiterWBAP_C(u, uRec, uRecNew, si, FBoundary);

    // 4. Evaluate RHS.
    eval->EvaluateRHS(rhs, u, uRec, t);

    // 5. Update u (RK combination).
    u.addTo(rhs, rkAlpha[iRK] * dt);
}
```

(Source: `DoReconstructionIter` declared at
`VariationalReconstruction.hpp:985`, impl at
`VariationalReconstruction_Reconstruction.hxx:485-615`.)

For the complete production solver, including all optimizations
(persistent reconstruction requests, implicit time integration,
preconditioned linear solves, shock-capturing limiters), see:

- `Euler/EulerSolver.hxx` -- the main solver loop
- `Euler/EulerEvaluator.hpp:123` (`InitializeFV`) -- Phases B--F
- `Euler/EulerEvaluator.hpp:399` (`EvaluateRHS`) -- production RHS
- `Euler/EulerEvaluator.hpp:980` (`fluxFace`) -- batched face flux
- `Euler/EulerEvaluator.hpp:1353` (`generateBoundaryValue`) -- BC dispatch

## Part 9: Minimum Python Example

A complete Python pipeline from mesh to RHS, adapted from
`test/CFV/test_vr_correctness.py`:

```python
from DNDSR import DNDS, CFV
from DNDSR.Geom.utils import create_mesh_from_CGNS
import numpy as np

mpi = DNDS.MPIInfo()
mpi.setWorld()

# Phase A: mesh
mesh, reader, name2Id = create_mesh_from_CGNS(
    "data/mesh/Uniform_3x3.cgns", mpi, dim=2)

# Phase B: VR object + settings
vfv = CFV.VariationalReconstruction_2(mpi, mesh)
s = vfv.GetSettings()
s.update({"maxOrder": 2, "intOrder": 5, "cacheDiffBase": True,
          "jacobiRelax": 1.0, "SORInstead": False})
vfv.ParseSettings(s)
vfv.SetPeriodicTransformationsNoOp()

# Phase C: geometric metrics
vfv.ConstructMetrics()

# Phase D: BC weights (empty for periodic)
vfv.ConstructBaseAndWeight_map({})

# Phase E: reconstruction coefficients
vfv.ConstructRecCoeff()

# Phase F: DOF arrays
u       = CFV.tUDof_D();  vfv.BuildUDof_D(u, 1)
uRec    = CFV.tURec_D();  vfv.BuildURec_D(uRec, 1)
uRecNew = CFV.tURec_D();  vfv.BuildURec_D(uRecNew, 1)
rhs     = CFV.tUDof_D();  vfv.BuildUDof_D(rhs, 1)

# Phase G: evaluator (demo advection-diffusion)
eval_obj = CFV.ModelEvaluator(mesh, vfv,
                               {"ax": 1.0, "ay": 0.0, "sigma": 0.0}, 1)

# Initial condition
for iCell in range(mesh.NumCell()):
    c = np.array(vfv.GetCellBary(iCell), copy=False)
    u[iCell][0] = np.sin(c[0]) * np.cos(c[1])
u.trans.startPersistentPull()
u.trans.waitPersistentPull()

# Solver step: VR + RHS
for _ in range(3):
    eval_obj.DoReconstructionIter(uRec, uRecNew, u, 0.0, putIntoNew=True)
    uRec, uRecNew = uRecNew, uRec
eval_obj.EvaluateRHS(rhs, u, uRec, 0.0)

print(f"RHS L2 norm: {rhs.norm2():.4e}")
```

See `test/CFV/test_vr_correctness.py:41-120` for the full working test,
and `test/cpp/Euler/test_EulerEvaluator.cpp` for the C++ equivalent
exercising a production Euler/NS evaluator.
