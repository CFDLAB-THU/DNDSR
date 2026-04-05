# cmake/DndsCompilerFlags.cmake
# LTO, MPI discovery, platform-specific compiler/linker flags, and OpenMP.
# Must be included after DndsOptions.cmake and DndsCudaSetup.cmake.
#
# NOTE: This file does NOT touch DNDS_EXTERNAL_LIBS or DNDS_EXTERNAL_INCLUDES.
# Library-list assembly happens in DndsExternalDeps.cmake.

# -------------------------------------------------------------------
# LTO
# -------------------------------------------------------------------
if(DNDS_LTO)
    if(DNDS_LTO_THIN)
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto=thin")
    else()
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto=${DNDS_LTO_N}")
    endif()
    # ! todo: add changing to gcc-ar gcc-ranlib gcc-nm
endif()

# -------------------------------------------------------------------
# Misc global flags
# -------------------------------------------------------------------
if(DNDS_USE_FULL_TEMPLATE_TRACE)
    add_compile_options(-ftemplate-backtrace-limit=0)
endif()

if(UNIX)
    if(DNDS_USE_RDYNAMIC)
        add_compile_options(-rdynamic)
        add_link_options(-rdynamic)
    endif()
endif()

# -------------------------------------------------------------------
# MPI discovery and Windows linker flags
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Platform-specific compiler flags, warnings, OpenMP
# -------------------------------------------------------------------

############################
# Platform specific settings
############################

if(UNIX OR MINGW)
    message(STATUS "UNIX OR MINGW")
    if(MINGW)
        add_compile_options(-Wa,-mbig-obj) # too many sections
    endif()
    message(${CMAKE_CXX_COMPILER_ID})
    if(DNDS_NATIVE_ARCH)
        add_compile_options(-march=native)
        add_compile_options(-mtune=native)
        message(WARNING "using -march=native")
    endif()
    if(DNDS_UNSAFE_MATH_OPT)
        add_compile_options(-funsafe-math-optimizations)
        message(WARNING "using -funsafe-math-optimization")
    endif()
    add_compile_options(-fdiagnostics-color=always)

    ### set warnings
    # -Wconversion
    add_compile_options(
        -Wall -Wno-unused-but-set-variable -Wno-unused-variable  -Wno-sign-compare
        $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror=return-type>
        )
    # Force RPATH (not RUNPATH) so bundled libstdc++ takes precedence
    # over LD_LIBRARY_PATH (e.g. from conda environments).
    # OpenMPI's mpicxx passes --enable-new-dtags; we override it.
    add_link_options("-Wl,--disable-new-dtags")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>) # declared but never referenced
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=2417>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>)
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--diag_error=host_call_from_device>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:cross-execution-space-call>)

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
    message(STATUS "WIN32 OR MSVC")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
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
        add_compile_options(-fdiagnostics-color=always)
        message(STATUS "MSVC ${MSVC}")
        message(STATUS "MSVC_VERSION ${MSVC_VERSION}")
    endif()
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

    if(WIN32 AND("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM"))
         #! we use msvc-style abi in library (which is default in clang but not for icx)
        string(APPEND CMAKE_CXX_FLAGS " -fms-compatibility -fms-extensions")
        # add_compile_options(-fms-compatibility -fms-extensions)
        # add_link_options(-fms-compatibility -fms-extensions)

        add_compile_options(-Wall -Wno-unused-but-set-variable -Wno-unused-variable -Wno-sign-compare)
        add_compile_options(-D_CRT_SECURE_NO_WARNINGS)

        if(DNDS_UNSAFE_MATH_OPT)
            add_compile_options(-ffast-math)
            message(WARNING "using -ffast-math")
        endif()

        link_libraries(shlwapi) # when hdf is compiled with msvc, needs this
    endif()


    if (DNDS_USE_OMP)
        #
        if(MSVC AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")) #! force use /openmp
            # set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "/openmp")
            add_compile_options(-openmp)
        elseif(WIN32 AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))#! force use -fopenmp
            add_compile_options(-fopenmp)
            add_link_options(-fopenmp)
        elseif(WIN32 AND("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM"))
            add_compile_options(-openmp)
            add_link_options(-openmp)
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
