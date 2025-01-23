#!/bin/bash

export CMAKE_LIBRARY_PATH=${LIBRARY_PATH} 


cmake .. \
-D EXTERNAL_LIB_ZLIB=${ZLIB_LIB}/libz.a \
-D EXTERNAL_LIB_HDF5=${HDF5_LIBDIR}/libhdf5.a \
-D DNDS_EXTERNAL_INCLUDE_HDF5=${HDF5_INCDIR} \
-D EXTERNAL_LIB_CGNS=~/tools/cfd_externals/install/lib/libcgns.a \
-D DNDS_EXTERNAL_INCLUDE_CGNS=~/tools/cfd_externals/install/include \
-D EXTERNAL_LIB_METIS=~/tools/cfd_externals/install/lib/libmetis.a \
-D DNDS_EXTERNAL_INCLUDE_METIS=~/tools/cfd_externals/install/include \
-D EXTERNAL_LIB_PARMETIS=~/tools/cfd_externals/install/lib/libparmetis.a \
-D DNDS_EXTERNAL_INCLUDE_PARMETIS=~/tools/cfd_externals/install/include \
