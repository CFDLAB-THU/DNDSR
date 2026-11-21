# Cauchy multidomain configuration and restart

Each domain owns one prepared `UnstructuredMesh`, equation state, FV/CFV plans
and local solver. `Simulation` owns domain placement, the coordinated clock
and directed interface connections. Physical time settings are shared; initial
conditions, boundaries, numerics and implicit controls belong to each domain.

## Configuration shape

Resolution applies schema defaults, the domain's selected root profiles, root
`defaults`, domain `overrides`, then CLI JSON-pointer overlays. The expanded
root has `domains[i].settings`; maintained input uses `overrides` instead.
Named boundary IDs are local to each mesh/domain.

The following illustrates four control ranks split into two domain groups. It
assumes the two CGNS files contain a boundary named `SLIDE` and have no other
unmapped physical boundaries. Supply all names/zones required by real inputs.

```json
{
  "defaults": {
    "equation": {"system": "compressible", "eos": "perfectGas"},
    "initialPrimitive": [1, 0.1, 0.2, 0, 1],
    "mesh": {"dimension": 2, "boundaryIds": {"SLIDE": 101}},
    "boundaries": {"101": {"kind": "interface"}},
    "numerics": {"reconstruction": "variational", "cfv": {"maxOrder": 3}}
  },
  "domains": [
    {"name": "outer", "ranks": [0, 1],
     "overrides": {"mesh": {"file": "outer.cgns"}}},
    {"name": "disk", "ranks": [2, 3],
     "overrides": {"mesh": {"file": "disk.cgns"},
                   "motion": {"angularVelocity": [0, 0, 0.02], "center": [5, 5, 0]}}}
  ],
  "interfaces": [
    {"donor": "outer", "receiver": "disk", "donorZone": 101, "receiverZone": 101, "maxGap": 0.02},
    {"donor": "disk", "receiver": "outer", "donorZone": 101, "receiverZone": 101, "maxGap": 0.02}
  ],
  "time": {"method": "ssprk3", "dt": 0.002, "steps": 1000},
  "output": {"prefix": "output/vortex"}
}
```

Omitting `ranks` assigns all control ranks to that domain. Shared-rank domains
execute their local phases in the agreed domain order; disjoint groups may run
concurrently. Python supplies one mesh slot per configured domain, using `None`
exactly on ranks outside that domain. Mesh communicators must have the same
membership and order as the configured domain group. Keep every control rank
in the same `Simulation` collective call sequence.

Interfaces are directed point-interpolation connections; use two entries for a
two-way connection. The donor and receiver zone numbers need not be equal.
Connected domains must agree on the equation, material properties and code-unit
scales. ACM patches must be reciprocal. A physical outlet or farfield anywhere
in the component sets its datum. In a closed component, the first configured
member supplies the target pressure mean. The simulation shifts every member
by the same amount. Reference mesh coordinates
stay fixed. At an interface, absolute momentum (compressible) or absolute
velocity (ACM) is rotated into the receiver axes. Pressure and turbulence
variables transfer as scalars. See the [ACM qualification cases](../architecture/CauchyACM.md).

For SA and SST, each mesh supplies its own nodal wall-distance field. When
that field is absent, setup searches for no-slip walls on the domain
communicator. In a fluid region whose nearest wall belongs to a different
domain, this search misses that wall. Relative motion can also change which
wall is nearest. Include this limitation when interpreting turbulence sources
near a moving interface. The moving-disk SA/SST qualification covers wall-free transport.

## Mesh input and HDF archives

The current default C++ `Simulation` loader reads **one CGNS file per domain**
from `mesh.file`, distributes owned data and prepares native topology/ghosts.
CGNS boundary names require explicit `mesh.boundaryIds`; periodic translations
use `mesh.periodicLengths`, expressed in the input file's coordinate system.
Periodic matching happens before `mesh.scale`/rotation; the later geometry
transform updates both coordinates and periodic frames. For a unit-cube mesh
scaled by `2*pi`, specify periodic lengths `[1,1,1]`, not `[2*pi,2*pi,2*pi]`.
A shared multi-zone CGNS file is not automatically
split into fluid domains. Paths resolve relative to the executable's working
directory.

An injected C++ `Simulation::MeshFactory`, or Python's preloaded mesh list, may
supply prepared meshes without a file path. The rotating-vortex evidence driver
uses this path to generate its deterministic Delaunay outer/disk meshes.

Geom already provides native HDF mesh archive readers:
`ReadNativeMeshArchiveOwned` restores owned arrays before preparation, while
`ReadNativeMeshArchive` also rebuilds ghost closure. Matching-rank or explicit
contiguous repartition reads are supported; a later mesh transition can apply
the desired computing partition. These readers can feed the mesh factory.
**Cauchy's JSON `mesh.file` loader currently selects CGNS only**; renaming the
input to `.h5` does not select a native HDF reader. A configurable HDF loader is
a frontend extension, not an implemented option.

## Solution restart shape

For multiple domains, `Simulation::WriteCheckpoint("restart.h5")` creates:

```text
restart.h5                         small HDF simulation manifest
restart.h5.parts/<generation>/
  domain0.h5                       ordinary native Domain checkpoint
  domain1.h5
```

The manifest stores its format version, accepted time/step and JSON mapping of
ordered domain names to relative child paths. Each child stores the complete
resolved root config, domain name, physical/equation/frame metadata, persistent
`cell_ids`, owned means, compatible time history and the mesh-validation
envelope (topology, coordinates and periodic transforms). Extended states retain
all flow/RANS/species fields. Runtime Array I/O preserves native layouts rather
than constructing a full AoS normalization buffer.

Reconstruction is omitted by default. Missing coefficients or an incompatible
reconstruction degree cold-start to zero; saving compatible coefficients and
method history permits exact warm continuation for supported profiles. Derived
property/diagnostic buffers are rebuilt. A single domain covering the whole
control communicator retains the existing single-file Domain format.

The write publishes the new manifest only after every child succeeds. Existing
children stay immutable; old generations are retained, and cleanup is an
explicit user operation. During reading, all domains validate and reserve
destinations before any accepted states are published. Interface packets are
refreshed at the restored physical time. This is coordinated software failure
handling, not recovery from a lost MPI process.

Construct the destination meshes first, then call `ReadCheckpoint`. Persistent
cell IDs allow a different MPI count, partition/order and supported field layout.
Domain names/order, physical model and validated mesh identity must remain
compatible; this is not arbitrary remeshing interpolation. Mesh-validation
arrays in a solution checkpoint do not replace a standalone mesh archive or a
mesh construction frontend. Keep the manifest and its `.parts` directory together
when moving a restart. Visualization instead writes one `<prefix>_domainN.vtkhdf`
per domain, in reference coordinates; a renderer applies the recorded rigid pose
when displaying the ground-frame geometry.
