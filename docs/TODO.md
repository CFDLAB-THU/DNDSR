# TODO

## Miscellaneous

- DPW: see separation size
- Outer tangential functional (1 3 3 1 -> 1 9 9 1)

- 2EQ: other equations ?

## Primary Features

- Rotational source and velocity
- Rotational bc
- Multi Zone Support
- Forcing Source
- LES
- DES

## Supplementary Features

- CGNS writer for mesh
- CGNS writer for plt data
- H5 (parallel) serializer
- CGNS parallel read/write
- parallel mesh partitioning
- true serial mesh distribution
- mesh bisect refinement

## Refactoring

- EulerSolver/Evaluator: better modulated
- Use CRTP somewhere?
- Reorganize Elements and Quadratures
- manage classes' visibility better

## Unit Test

- Basic tests for MPI and important external libs
- arrays: serial operations and comm
- Elems and Quads
- meshes
- reconstruction
- ode and Krylov solvers
- euler evaluator

## Python

- export interface?
- inject scripts?