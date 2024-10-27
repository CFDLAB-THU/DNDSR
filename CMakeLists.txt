## For configuration on Windows, use clang/gcc + makefile
## cmake .. -G "Unix Makefiles" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang


if(UNIX OR MINGW)
    cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.20)
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/bin CACHE FILEPATH "install place")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Release or Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Release" "Debug" "RelWithDebInfo" "MinSizeRel")


# ##########################################
project(DNDSR LANGUAGES CXX) #! need to be in front of find_package(MPI)
enable_language(C) #! building C sub-projects

message("external install hard guess at: external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(HARD_GUESS_PATH external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

# First Hard Guess
set(CMAKE_PREFIX_PATH ${HARD_GUESS_PATH})

# need CUDA project support ?
# manual perferences
# set(CMAKE_CXX_STANDARD_REQUIRED 17)
# set(CMAKE_CXX_STANDARD 17)

#! why?
# find_package(MPI REQUIRED)

set(DNDS_SUPRESS_INSERT_CHECK ON CACHE BOOL "close the inserted outputs")
set(DNDS_USE_NDEBUG_MACRO OFF CACHE BOOL "use NDEBUG macro for eigen output and cassert etc") # performance impact is small so off by default(without -DNDEBUG)
set(DNDS_USE_PARALLEL_MACRO OFF CACHE BOOL "for auto multi-thread?")
set(DNDS_USE_OMP ON CACHE BOOL "for openmp enabling")
set(DNDS_FAST_BUILD_FAST ON CACHE BOOL "don't -g on basic modules")
set(DNDS_USE_NO_OMIT_FRAME_POINTER OFf CACHE BOOL "in fast build fast part, use -fno-omit-frame-pointer to get correct call graph in perf's fp mode")
set(DNDS_UNSAFE_MATH_OPT OFF CACHE BOOL "use -funsafe-math-optimizations")
set(DNDS_NATIVE_ARCH     OFF CACHE BOOL "use -march=native")
set(DNDS_LTO OFF CACHE BOOL "use -flto")
set(DNDS_LTO_THIN OFF CACHE BOOL "use -flto=thin")
set(DNDS_USE_FULL_TEMPLATE_TRACE OFF CACHE BOOL "use -ftemplate-backtrace-limit=0")
set(DNDS_USE_RDYNAMIC ON CACHE BOOL "use -rdynamic on posix")


set(DNDS_EIGEN_USE_BLAS OFF CACHE BOOL "use blas in eigen")
set(DNDS_EIGEN_USE_LAPACK OFF CACHE BOOL "use lapack in eigen")

set(DNDS_SUPERLU_DIST_USE_BLAS OFF CACHE BOOL "use blas in superlu_dist")

set(DNDS_RECORD_COMMIT ON CACHE BOOL "record commit id for each cmake configure")
set(DNDS_VERBOSE_BUILDING ON CACHE BOOL "make the makefile verbose, overrides CMAKE_VERBOSE_MAKEFILE")

if(DNDS_RECORD_COMMIT)
    execute_process(COMMAND git rev-parse HEAD OUTPUT_FILE ${CMAKE_BINARY_DIR}/commitID.txt WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} TIMEOUT 120 RESULT_VARIABLE RET)
    message(STATUS "git rev-parse returned ${RET}")
    if(NOT RET EQUAL 0)
        message(WARNING "git rev-parse failed")
        set(DNDS_RECORDED_COMMIT_HASH "UNKNOWN" CACHE INTERNAL  "Project Public Current Commit Hash")
    else()
        # message(STATUS "current commit id: ")
        # execute_process(COMMAND cat commitID.txt WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 10)
        file(READ "${CMAKE_BINARY_DIR}/commitID.txt" DNDS_RECORDED_COMMIT_HASH)
        string(STRIP ${DNDS_RECORDED_COMMIT_HASH} DNDS_RECORDED_COMMIT_HASH)
        set(DNDS_RECORDED_COMMIT_HASH ${DNDS_RECORDED_COMMIT_HASH} CACHE INTERNAL  "Project Public Current Commit Hash")
        message(STATUS "current commit id: ${DNDS_RECORDED_COMMIT_HASH}")
    endif()
endif()

if(DNDS_VERBOSE_BUILDING)
    set(CMAKE_VERBOSE_MAKEFILE ON)
else()
    set(CMAKE_VERBOSE_MAKEFILE OFF)
endif()

if(DNDS_LTO)
    if(DNDS_LTO_THIN)
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto=thin")
    else()
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto")
    endif()
    # ! todo: add changing to gcc-ar gcc-ranlib gcc-nm
endif()

if(DNDS_USE_FULL_TEMPLATE_TRACE)
    add_compile_options(-ftemplate-backtrace-limit=0)
endif()


if(UNIX)
    if(DNDS_USE_RDYNAMIC)
        add_compile_options(-rdynamic)
        add_link_options(-rdynamic)
    endif()
endif()


find_package(MPI REQUIRED)
if(UNIX)
    # set(CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "compiler with your MPI wrapping")
    # set(CMAKE_FIND_LIBRARY_SUFFIXES ".a") # ! using static libs
elseif(MSVC OR WIN32 OR MINGW)
    # nothing
    
    if(MPI_CXX_FOUND)
        set (CMAKE_EXE_LINKER_FLAGS "${MPI_C_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
    else()
        message((FATAL_ERROR "MPI NOT FOUND"))
    endif()
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()





set(CMAKE_USE_RELATIVE_PATHS ON)




# ##########################################
find_library(EXTERNAL_LIB_BACKTRACE NAMES backtrace)
# need zlibstatic first to find the static version on windows
find_library(EXTERNAL_LIB_ZLIB NAMES zlibstatic zlib z z_D libzlibstatic.a REQUIRED)
find_library(EXTERNAL_LIB_HDF5 NAMES libhdf5.a hdf5 PATHS 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
     REQUIRED)
# find_library(EXTERNAL_LIB_HDF5_HL NAMES libhdf5_hl.a hdf5_hl PATHS 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
#      REQUIRED)
find_library(EXTERNAL_LIB_CGNS NAMES libcgns.a cgns PATHS 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
     REQUIRED)
find_library(EXTERNAL_LIB_METIS NAMES libmetis.a metis PATHS 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
         REQUIRED)
find_library(EXTERNAL_LIB_PARMETIS NAMES libparmetis.a parmetis PATHS 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
     REQUIRED)
# find_library(EXTERNAL_LIB_TECIO tecio PATHS 
    # "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" 
    # "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/lib"
    # NO_DEFAULT_PATH) #not needed for now
find_library(EXTERNAL_LIB_BLAS NAMES openblas blas )
find_library(EXTERNAL_LIB_LAPACK NAMES lapack )
find_library(EXTERNAL_LIB_LAPACKE NAMES lapacke)

find_path(EXTERNAL_INCLUDE_ZLIB zlib.h REQUIRED)
find_path(EXTERNAL_INCLUDE_HDF5 hdf5.h PATHS 
    "${PROJECT_SOURCE_DIR}/external/HDF5/include"
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
    REQUIRED)
find_path(EXTERNAL_INCLUDE_CGNS cgnslib.h PATHS 
    "${PROJECT_SOURCE_DIR}/external/cgns/include" 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
    REQUIRED)
find_path(EXTERNAL_INCLUDE_METIS metis.h PATHS 
    "${PROJECT_SOURCE_DIR}/external/metis/include" 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
    REQUIRED)
find_path(EXTERNAL_INCLUDE_PARMETIS parmetis.h PATHS 
    "${PROJECT_SOURCE_DIR}/external/parmetis/include" 
    "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
    REQUIRED)
# find_path(EXTERNAL_INCLUDE_TECIO TECIO.h PATHS 
#     "${PROJECT_SOURCE_DIR}/external/tecio/include" 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
#     NO_DEFAULT_PATH) #!not needed for now
find_path(EXTERNAL_INCLUDE_EIGEN eigen3.pc.in PATHS 
    "${PROJECT_SOURCE_DIR}/external/eigen-3.4.0" 
    "${PROJECT_SOURCE_DIR}/external/eigen" 
    NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_BOOST boost PATHS 
    "${PROJECT_SOURCE_DIR}/external/boost_1_82_0" 
    NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_CGAL CGAL PATHS 
    "${PROJECT_SOURCE_DIR}/external/CGAL-5.6/include" 
    "${PROJECT_SOURCE_DIR}/external/CGAL-5.6" 
    NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_JSON json.hpp PATHS 
    "${PROJECT_SOURCE_DIR}/external/nlohmann" 
    NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_ARGPARSE argparse.hpp PATHS 
    "${PROJECT_SOURCE_DIR}/external/argparse-3.0/include/argparse" 
    NO_DEFAULT_PATH REQUIRED)
# find_path(EXTERNAL_INCLUDE_CPPTRACE cpptrace.hpp PATHS 
#     "${PROJECT_SOURCE_DIR}/external/cpptrace-0.5.1/include/cpptrace" 
#     NO_DEFAULT_PATH REQUIRED)
find_path(EXTERNAL_INCLUDE_CPPCODEC base32_hex.hpp PATHS
    "${PROJECT_SOURCE_DIR}/external/cppcodec" 
    NO_DEFAULT_PATH REQUIRED)
# find_path(EXTERNAL_INCLUDE_RAPIDJSON rapidjson 
#     PATHS 
#     "${PROJECT_SOURCE_DIR}/external/rapidjson/include" 
#     "${PROJECT_SOURCE_DIR}/external/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/include"
#     NO_DEFAULT_PATH)
find_path(EXTERNAL_INCLUDE_NANOFLANN nanoflann.hpp PATHS
    "${PROJECT_SOURCE_DIR}/external/nanoflann-1.4.3" 
    NO_DEFAULT_PATH REQUIRED)

find_path(EXTERNAL_INCLUDE_PYBIND11 pybind11 PATHS
    "${PROJECT_SOURCE_DIR}/external/pybind11-2.11.1/include" 
    NO_DEFAULT_PATH REQUIRED)

find_path(EXTERNAL_INCLUDE_FMT fmt PATHS
    "${PROJECT_SOURCE_DIR}/external/fmt-10.1.1/include" 
    NO_DEFAULT_PATH REQUIRED)

find_path(EXTERNAL_INCLUDE_SUPERLU superlu_dist.pc.in PATHS
    "${PROJECT_SOURCE_DIR}/external/superlu_dist-8.2.1" 
    NO_DEFAULT_PATH)

find_path(EXTERNAL_DOXYGEN_AWESOME doxygen-awesome.css PATHS 
    "${PROJECT_SOURCE_DIR}/external/doxygen-awesome-css-2.2.1" 
    NO_DEFAULT_PATH)

add_subdirectory("${EXTERNAL_INCLUDE_PYBIND11}/..")
add_subdirectory("${EXTERNAL_INCLUDE_FMT}/..")
# add_subdirectory("${EXTERNAL_INCLUDE_CPPTRACE}/../..")

if(EXTERNAL_INCLUDE_SUPERLU)
    set(USE_XSDK_DEFAULTS ON CACHE BOOL "Init by DNDS") # for superlu
    if (DNDS_SUPERLU_DIST_USE_BLAS)
        message("Setting DNDS_SUPERLU_DIST_USE_BLAS")
        set(TPL_ENABLE_INTERNAL_BLASLIB OFF CACHE BOOL "Init by DNDS")
        set(TPL_BLAS_LIBRARIES ${EXTERNAL_LIB_BLAS} CACHE FILEPATH "Init by DNDS")
    else()
        set(TPL_ENABLE_INTERNAL_BLASLIB ON CACHE BOOL "Init by DNDS")
    endif()
    set(XSDK_ENABLE_Fortran OFF CACHE BOOL "Init by DNDS")
    set(TPL_PARMETIS_INCLUDE_DIRS ${EXTERNAL_INCLUDE_PARMETIS}  CACHE PATH "Init by DNDS")
    set(TPL_PARMETIS_LIBRARIES ${EXTERNAL_LIB_PARMETIS}  CACHE FILEPATH "Init by DNDS")
    set(XSDK_INDEX_SIZE 64 CACHE STRING "Init by DNDS")
    if(WIN32 OR MSVC)
        if(MSVC_VERSION) #! is using mingw/clang+msvc toolchain, inform superlu that do not link -lm
            set(MSVC 1 CACHE INTERNAL "Init by DNDS")
        endif()
    endif()
    add_subdirectory("${EXTERNAL_INCLUDE_SUPERLU}")
    #! for superlu, remember to comment the #include<unistd.h>, not needed
endif()


set (EXTERNAL_LIBS "")

if (MPI_CXX_FOUND)
    set(EXTERNAL_LIBS ${MPI_CXX_LIBRARIES})
    message(STATUS "MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES}")
endif()

set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    fmt::fmt
    # cpptrace::cpptrace
    CACHE INTERNAL
    "Project Public External Lib Dependencies"
    )

if(EXTERNAL_INCLUDE_SUPERLU)
set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    superlu_dist
    CACHE INTERNAL
    "Project Public External Lib Dependencies"
    )
endif()

if(EXTERNAL_LIB_BACKTRACE)
set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    ${EXTERNAL_LIB_BACKTRACE}
    CACHE INTERNAL
    "Project Public External Lib Dependencies"
    )
add_compile_definitions(BOOST_STACKTRACE_USE_BACKTRACE)
endif()

# set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
#     fmt::fmt
#     CACHE INTERNAL
#     "Project Public External Lib Dependencies"
#     )

set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    ${EXTERNAL_LIB_CGNS}
    # ${EXTERNAL_LIB_HDF5_HL} # HDF5_HL relies on HDF5
    ${EXTERNAL_LIB_HDF5}
    ${EXTERNAL_LIB_PARMETIS}
    ${EXTERNAL_LIB_METIS}
    ${EXTERNAL_LIB_ZLIB}
    # ${EXTERNAL_LIB_TECIO}
    CACHE INTERNAL
    "Project Public External Lib Dependencies"
    )
if (DNDS_EIGEN_USE_BLAS)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    ${EXTERNAL_LIB_BLAS})
endif()
if (DNDS_EIGEN_USE_LAPACK)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    ${EXTERNAL_LIB_LAPACK}
    ${EXTERNAL_LIB_LAPACKE})
endif()
message("EXTERNAL_LIBS  " ${EXTERNAL_LIBS})

set(EXTERNAL_INCLUDES 
    ${EXTERNAL_INCLUDE_ZLIB}
    ${EXTERNAL_INCLUDE_HDF5}
    ${EXTERNAL_INCLUDE_CGNS}
    ${EXTERNAL_INCLUDE_METIS}
    ${EXTERNAL_INCLUDE_PARMETIS}
    # ${EXTERNAL_INCLUDE_TECIO}
    ${EXTERNAL_INCLUDE_EIGEN}
    ${EXTERNAL_INCLUDE_JSON}
    ${EXTERNAL_INCLUDE_ARGPARSE}
    # ${EXTERNAL_INCLUDE_CPPTRACE}
    # ${EXTERNAL_INCLUDE_RAPIDJSON}
    ${EXTERNAL_INCLUDE_CPPCODEC}
    ${EXTERNAL_INCLUDE_NANOFLANN}
    ${EXTERNAL_INCLUDE_BOOST}
    ${EXTERNAL_INCLUDE_CGAL}
    ${EXTERNAL_INCLUDE_PYBIND11}
    ${EXTERNAL_INCLUDE_FMT}
    CACHE INTERNAL "Project External Includes")

if(EXTERNAL_INCLUDE_SUPERLU)
set(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES}
    ${EXTERNAL_INCLUDE_SUPERLU}/SRC
    CACHE INTERNAL "Project External Includes")
endif()

add_compile_definitions(__DNDS_REALLY_COMPILING__)

if(DNDS_EIGEN_USE_BLAS)
    add_compile_definitions(EIGEN_USE_BLAS)
endif()

if(DNDS_EIGEN_USE_LAPACK)
    add_compile_definitions(EIGEN_USE_LAPACKE_STRICT)
endif()

if(UNIX OR MINGW)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
    stdc++fs)
    if(UNIX)
        set(EXTERNAL_LIBS ${EXTERNAL_LIBS}
        dl) # add dl
    endif()
    if(MINGW)
        add_compile_options(-Wa,-mbig-obj) # too many sections
    endif()
    message(${CMAKE_CXX_COMPILER_ID})
    add_compile_options(-std=c++17)
    if(DNDS_NATIVE_ARCH)
        add_compile_options(-march=native)
        add_compile_options(-mtune=native)
        message(WARNING "using -march=native")
    endif()
    if(DNDS_UNSAFE_MATH_OPT)
        add_compile_options(-funsafe-math-optimizations)
        message(WARNING "using -funsafe-math-optimization")
    endif()
    
    ### set warnings
    add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare -Werror=return-type)
    

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if (DNDS_USE_OMP)
            add_compile_options(-fopenmp)
            add_link_options(-fopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            # doesn't seem available
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if (DNDS_USE_OMP)
            add_compile_options(-fopenmp)
            add_link_options(-fopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            add_definitions(-D_GLIBCXX_PARALLEL)
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # add_compile_options(-fopenmp)
        if (DNDS_USE_OMP)
            add_compile_options(-fiopenmp)
            add_link_options(-fiopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            # add_definitions(-D_GLIBCXX_PARALLEL) #! not yet found 
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
        add_compile_options(-diag-disable=1011) # branch no return 
        add_compile_options(-diag-disable=2196) # both inline and no inline
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
        if (DNDS_USE_OMP)
            add_compile_options(-fiopenmp)
            add_link_options(-fiopenmp)
        endif()
        if (DNDS_USE_PARALLEL_MACRO)
            # add_compile_definitions(_GLIBCXX_PARALLEL)
            # add_definitions(-D_GLIBCXX_PARALLEL) #! not yet found 
            message(WARNING "${CMAKE_CXX_COMPILER_ID} compliler not using DNDS_USE_PARALLEL_MACRO")
        endif()
        add_compile_options(-Wno-tautological-constant-compare) # ! is this risky or not
        message(WARNING "${CMAKE_CXX_COMPILER_ID} using -Wno-tautological-constant-compare")
    endif()
    

elseif(WIN32 OR MSVC)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
        add_compile_options(-std=c++17 )
        add_link_options(-lshlwapi) # when hdf is compiled with msvc, needs this
        # add_compile_options(--target=x86_64-pc-windows-gnu)
        # add_link_options(--target=x86_64-pc-windows-gnu)
        add_compile_options(-D_CRT_SECURE_NO_WARNINGS -DCOMPILER_MSC)
        if(DNDS_UNSAFE_MATH_OPT)
            add_compile_options(-funsafe-math-optimizations)
            message(WARNING "using -funsafe-math-optimization")
        endif()
        if(DNDS_NATIVE_ARCH)
            add_compile_options(-march=native)
            message(WARNING "using -march=native")
        endif()
        message(STATUS "MSVC ${MSVC}")
        message(STATUS "MSVC_VERSION ${MSVC_VERSION}")
    endif()
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS} # no dl
    )
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # add_compile_options(-Wall)
        add_compile_options(/std:c++17 /IGNORE:C2124) # cl is buggy in c++17 mode ...
        add_compile_options(-Wall)
        add_compile_options(-Wno-unused-variable -Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-parameter) # for clang-cl!
        add_compile_options(-Wno-reserved-identifier -Wno-documentation  -Wno-old-style-cast)
        add_compile_options(-Wno-old-style-cast -Wno-exit-time-destructors -Wno-global-constructors -Wno-zero-as-null-pointer-constant)
        add_compile_options(-Wno-sign-conversion -Wno-unused-template)
        add_compile_options(-Wno-documentation-unknown-command)
        add_compile_options(-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-c++11-compat -Wno-c++11-compat-pedantic -Wno-c++14-compat -Wno-c++14-compat-pedantic)
        add_compile_options(-Wno-extra-semi-stmt -Wno-comma -Wno-float-equal -Wno-missing-noreturn)
        add_compile_options(-Wno-undef -Wno-implicit-fallthrough -Wno-redundant-parens)
        # add_compile_options(-Wextra-semi-stmt  -Wshorten-64-to-32 -Wnewline-eof)
    endif()
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS} # no dl
    )
    message(${CMAKE_CXX_COMPILER_ID})
    
    
    if (DNDS_USE_OMP)
        #
        if(MSVC AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")) #! force use /openmp
            # set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "/openmp")
            add_compile_options(-openmp)
        elseif(WIN32 AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))#! force use -fopenmp
            add_compile_options(-fopenmp)
            add_link_options(-fopenmp)
        else()
            find_package(OpenMP REQUIRED)
            if(OpenMP_CXX_FLAGS)
                set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
            else()
                message(FATAL_ERROR "OMP not found")
            endif()
        endif()

        

    endif()
    if (DNDS_USE_PARALLEL_MACRO)
        # ! to be tested
    endif()
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()


if (MPI_CXX_FOUND)
    set(EXTERNAL_INCLUDES ${MPI_CXX_INCLUDE_DIRS} ${EXTERNAL_INCLUDES})
endif()

set(DNDS_INCLUDES ${PROJECT_SOURCE_DIR}/src)

if (UNIX)
add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
endif()

if (DNDS_SUPRESS_INSERT_CHECK)
    # add_compile_definitions(NINSERT)
    add_definitions(-DNINSERT)
endif()
if (DNDS_USE_NDEBUG_MACRO)
    # add_compile_definitions(NDEBUG)
    add_definitions(-DNDEBUG)
else()
    add_compile_options(-UNDEBUG)
endif()
if (DNDS_USE_OMP)
    # add_compile_definitions(DNDS_USE_OMP)
    add_definitions(-DDNDS_USE_OMP)
endif()

# if (DNDS_USE_PARALLEL_MACRO)
#     add_compile_definitions(PARALLEL)
# endif()

message(STATUS "EXTERNAL_LIBS ${EXTERNAL_LIBS}")
message(STATUS "EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES}")

set(INCLUDES \\
${DNDS_INCLUDES} CACHE INTERNAL "Project Public Includes")



## app exes

set(APPS_EXTERNAL
cgns_APITest
eigen_Test
STL_Test
json_Test
cgal_AABBTest
mpi_test
)

set(APPS_DNDS
array_Test
arrayTrans_test
arrayDerived_test
serializerJSON_Test
stdPowerTest
)

set(APPS_Geom
elements_Test
meshSerial_Test
ofReader_Test
partitionMeshSerial
)

set(APPS_CFV
vrStatic_Test
vrBasic_Test
diffTensors_Test
)

set(APPS_Euler
gasTest
euler
euler3D
eulerSA
eulerSA3D
euler2EQ
euler2EQ3D
jacobiLUTest
oneDimProfileTest
)


add_subdirectory(${CMAKE_SOURCE_DIR}/src/DNDS)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/Geom)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/CFV)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/Euler)

set(LIBS
euler_library
euler_library_fast
cfv
geom
dnds
CACHE INTERNAL
"Public Project Internal Static Libs"
)

set(EXE_SUFFIX "")
if (UNIX)
    set(EXE_SUFFIX ".exe")
endif()



macro(ADD_EXE_APP EXES MAIN_DIR)
    message(STATUS ${EXES})
    foreach(EXE ${EXES})
        # add_executable(${EXE} app/external/${EXE}.cpp $<TARGET_OBJECTS:${OBJS}>)
        add_executable(${EXE} ${MAIN_DIR}/${EXE}.cpp)
        target_link_libraries(${EXE} PUBLIC ${LIBS})
        target_link_libraries(${EXE} PUBLIC ${EXTERNAL_LIBS})
        target_include_directories(${EXE} PUBLIC ${EXTERNAL_INCLUDES} PUBLIC ${INCLUDES})
        set_target_properties(${EXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/app RUNTIME_OUTPUT_NAME "${EXE}${EXE_SUFFIX}")
        if(DNDS_RECORD_COMMIT)
            target_compile_definitions(${EXE} PUBLIC DNDS_CURRENT_COMMIT_HASH=${DNDS_RECORDED_COMMIT_HASH})
        endif()
    endforeach()
endmacro(ADD_EXE_APP)

ADD_EXE_APP("${APPS_EXTERNAL}" "app/external")
ADD_EXE_APP("${APPS_DNDS}" "app/DNDS")
ADD_EXE_APP("${APPS_Geom}" "app/Geom")
ADD_EXE_APP("${APPS_CFV}" "app/CFV")
ADD_EXE_APP("${APPS_Euler}" "app/Euler")



set(PYEXTS_EXTERNAL
pybind11_basicTest
)

function(add_python_ext EXT_NAME CPPS)
    pybind11_add_module(${EXT_NAME} ${CPPS})
endfunction(add_python_ext)

function(ADD_EXT_APP PYEXTS PREFIX)
    foreach(PYEXT ${PYEXTS})
        add_python_ext(${PYEXT} "${PREFIX}/${PYEXT}.cpp")
        target_link_libraries(${PYEXT} PUBLIC ${LIBS})
        target_link_libraries(${PYEXT} PUBLIC ${EXTERNAL_LIBS})
        target_include_directories(${PYEXT} PUBLIC ${EXTERNAL_INCLUDES} PUBLIC ${INCLUDES})
    endforeach(PYEXT)
endfunction(ADD_EXT_APP)

ADD_EXT_APP("${PYEXTS_EXTERNAL}" "app/external")






#######################################################
## doxygen
#######################################################

find_package(Doxygen)
if (DOXYGEN_FOUND)
    message(STATUS "Doxygen Found") 
    #! TODO: use automatic config Doxygen.in file
    ##### using custom DOXYGEN
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs ALL 
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT Doing Doxygen
        VERBATIM
    )
    execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/docs/getAllAttachForDox.py 
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/docs")

    # find_file(DOXYGEN_PDF_FILES docs/*.pdf NO_DEFAULT_PATH)
    # message(STATUS ${DOXYGEN_PDF_FILES})
    ##### using CMAKE convenient DOXYGEN!only in 3.9 +!!!
    
else(DOXYGEN_FOUND)
    message("Doxygen Not Found")
endif(DOXYGEN_FOUND)



