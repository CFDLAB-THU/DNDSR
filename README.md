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

- Make sure mpi and compiler environment is available:
  - Compiler
    - Compiler should be GCC 9 / Clang 8 or newer
    - C++17 support
  - MPI
    - `mpicc`, `mpicxx`, `mpirun` (or srun)
    - `mpicc --version`: check the compiler wrapped by `mpicc` wrapper
    - `mpicc -show` (for MPICH/OpenMPI) or `mpicc --showme` (for OpenMPI) to check full wrapped compiler command
    - MPI-3 standard compatible
  - CMake + GNU Make or Ninja
    - CMake >= 3.21
  - Python 3
    - `python3-dev` needed if on Debian systems and using system python
    - if use python modules, recommend using virtual environments (conda, venv ...)
    - python >= 3.9 for python modules
- First build / get your mpi dev package, zlib, HDF5, CGNS, metis and parmetis, and make them available to cmake (like via CMAKE_PREFIX for libs and PATH for executables).
  - Recommended way: [use git submodules and build](#using-git-submodule-to-build-cfd_externals). 
    - No need to provide extra variables to CMake or set extra environment variables if using git submodules
  - No internet: use source tarball [cfd_externals_source.tar.gz](https://github.com/harryzhou2000/cfd_externals/releases) released
    - extract inside `external` and `tar -zcvf` the tarball
    - goto `external/cfd_externals` and execute the cfd_externals_build.py
  - An optional way is to use existing system SDK for these libraries.
    - Make sure MPI-HDF-CGNS and MPI-parmetis are all compatible.

- Then get the `cfd_externals_headerOnlys` tarball [here](https://github.com/harryzhou2000/cfd_externals_headeronlys/releases) to directly get the sources of referenced repos. Extract it into `external` folder, making the path look like `external/eigen...` and so on.
  
- Cmake configure and build. From the project root, do:


```bash
mkdir build && cd build
CC=mpicc CXX=mpicxx cmake ..
cmake --build . -t <target> -j 8
```

to build the target. Replace `<target>` with `euler`, `eulerSA` and so on.

### Using Git Submodule to build `cfd_externals`

Some external libraries are adopted in DNDSR in the form of binary libraries linkages, and they are not integrated with CMake FetchContent for now.

```bash
git submodule update --init --recursive --depth=1
```

If every clone and checkout succeeds, move into `external/cfd_externals` and run the build script:

```bash
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
```

After this, all libraries are installed in `external/cfd_externals/insall`.

If you have trouble cloning the submodules from github, you can package the git repos on a machine with access to github, and redirect github.com to the local directory. See instructions in [cfd_externals](https://github.com/harryzhou2000/cfd_externals?tab=readme-ov-file#redirect-to-local-repos).

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