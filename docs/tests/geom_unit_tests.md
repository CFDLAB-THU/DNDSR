# Geom Module Unit Tests {#geom_unit_tests}

@tableofcontents

Tests for the mesh geometry infrastructure in `src/Geom/`.
All C++ tests use [doctest](https://github.com/doctest/doctest).
MPI-aware tests are registered with CTest at multiple process counts.

## Building and Running

```sh
# Build all Geom C++ test executables
cmake --build build -t geom_unit_tests -j8

# Run every Geom CTest
ctest --test-dir build -R geom_ --output-on-failure

# Run a single suite
ctest --test-dir build -R geom_elements --output-on-failure
```

## Target Summary

| CMake target | CTest name | Source file | Timeout |
|---|---|---|---|
| `geom_test_elements` | `geom_elements` | test_Elements.cpp | 120 s |
| `geom_test_quadrature` | `geom_quadrature` | test_Quadrature.cpp | 120 s |
| `geom_test_mesh_index_conversion` | `geom_mesh_index_conversion_np{1,2,4}` | test_MeshIndexConversion.cpp | 120 s |
| `geom_test_mesh_pipeline` | `geom_mesh_pipeline_np{1,2,4}` | test_MeshPipeline.cpp | 120 s |
| `geom_test_mesh_distributed_read` | `geom_mesh_distributed_read_np{1,2,4}` | test_MeshDistributedRead.cpp | 120 s |
| `geom_test_mesh_connectivity` | `geom_mesh_connectivity_np{1,2,4}` | test_MeshConnectivity.cpp | 120 s |
| `geom_test_mesh_connectivity_ghost` | `geom_mesh_connectivity_ghost_np{1,2,4}` | test_MeshConnectivity_Ghost.cpp | 120 s |
| `geom_test_mesh_connectivity_interpolate` | `geom_mesh_connectivity_interpolate_np{1,2,4}` | test_MeshConnectivity_Interpolate.cpp | 120 s |

---

## Element Types (test_Elements.cpp) {#geom_test_elements}
@see test_Elements.cpp

Serial-only tests for element geometry definitions, node counts, and
shape function evaluation.

---

## Quadrature Rules (test_Quadrature.cpp) {#geom_test_quadrature}
@see test_Quadrature.cpp

Serial-only tests for Gaussian quadrature points and weights on
standard elements (triangles, quads, tetrahedra, hexahedra, prisms,
pyramids).

---

## Mesh Index Conversion (test_MeshIndexConversion.cpp) {#geom_test_mesh_index_conversion}
@see test_MeshIndexConversion.cpp

MPI-parallel tests verifying local-to-global and global-to-local index
conversions on partitioned meshes.

---

## Mesh Pipeline (test_MeshPipeline.cpp) {#geom_test_mesh_pipeline}
@see test_MeshPipeline.cpp

MPI-parallel end-to-end tests for the mesh construction pipeline:
reading, partitioning, ghost creation, and boundary extraction.

---

## Distributed Mesh I/O (test_MeshDistributedRead.cpp) {#geom_test_mesh_distributed_read}
@see test_MeshDistributedRead.cpp

MPI-parallel tests for reading CGNS mesh files in parallel and
redistributing across different partition counts.

---

## MeshConnectivity DSL (test_MeshConnectivity.cpp) {#geom_test_mesh_connectivity}
@see test_MeshConnectivity.cpp

MPI-parallel tests for the `MeshConnectivity` standalone DSL operations:
`Inverse`, `Compose`, `ComposeFiltered`, `Interpolate`, and adjacency
registry management. Validates cone/support inversion, compose with
predicates, and entity-kind registration.

---

## Ghost Tree Evaluation (test_MeshConnectivity_Ghost.cpp) {#geom_test_mesh_connectivity_ghost}
@see test_MeshConnectivity_Ghost.cpp

MPI-parallel tests for `evaluateGhostTree`, `GhostSpec`,
`CompiledGhostTree`, and multi-layer ghost cell support. Tests include
single-layer and multi-layer ghost chains, scratch-pull between BFS
levels, and 2D tiled synthetic grids with analytical ghost formulas.

---

## Interpolation (test_MeshConnectivity_Interpolate.cpp) {#geom_test_mesh_connectivity_interpolate}
@see test_MeshConnectivity_Interpolate.cpp

MPI-parallel tests for `InterpolateGlobal` — distributed sub-entity
extraction with global deduplication and periodic-aware matching.
