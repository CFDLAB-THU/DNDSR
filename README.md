# DNDSR

DNDSR is currently being developed for implementation of Compact Finite Volume CFD programs.

The command line solver apps include:

- euler, 2D N-S solver
- euler3D, 3D N-S solver
- eulerSA, 2D N-S solver with S-A RANS model
- eulerSA3D, 3D version of eulerSA
- euler2EQ, 2D N-S solver with some 2-equation RANS models
- euler2EQ3D, 3D version of euler2EQ

## Building

### On Linux

- First build / get your mpi dev package, HDF5, CGNS, metis and parmetis, and make them available to cmake (like via CMAKE_PREFIX for libs and PATH for executables).
An optional way is to use an existing SDK for these libraries. 
For Linux with x86_64 CPU with **openmpi** dev package installed, you can get the `Linux-x86_64-GS.tar.gz` package from [here](https://cloud.tsinghua.edu.cn/d/35deb3d4f740449da29b/) and extract it inside the `external` folder, making the path look like `external/Linux-x86_64/include` ...
- Then get the `external_headerOnlys` here [here](https://cloud.tsinghua.edu.cn/d/35deb3d4f740449da29b/) to directly get the sources of referenced repos. Extract it into `external` folder, making the path look like `external/eigen...` and so on.
- Cmake configure and build. From the project root:

```bash
mkdir build && cd build
cmake ..
make euler -j8
```

Here euler could be replaced with any command line app.

## Running

In shell, assuming we directly run from `build`, use:

```bash
app/euler.exe <configFile.json>
```

for serial running, and 

```bash
mpirun -np <np> app/euler.exe <configFile.json>
```

for parallel running (on local machine).

All the input/configuration/parameters are defined in the jsonc (although named .json) file <configFile.json>. The input parameters are explained in [this example input json](cases/euler_default_config_commented.json).