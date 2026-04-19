# examples/CMakeLists.txt
#
# Auto-discovered compilable examples from the documentation guides.
#
# Build all:  cmake --build build -t examples -j32
# Build one:  cmake --build build -t ex_array_basic -j32
# Run:        mpirun -np 2 build/examples/ex_array_basic
#
# Adding a new example: drop a .cpp file here.  The filename prefix
# determines which libraries it links against:
#
#   ex_array_*   -> dnds
#   ex_geom_*    -> geom + dnds
#   ex_cfv_*     -> cfv + geom + dnds
#   ex_euler_*   -> euler_library_NS + ... + cfv + geom + dnds
#   (anything else) -> cfv + geom + dnds   (safe default)

# Glob all .cpp files in this directory.
file(GLOB _example_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ex_*.cpp")

set(_example_targets "")

foreach(_src ${_example_sources})
    get_filename_component(_name ${_src} NAME_WE)   # e.g. "ex_array_basic"

    # Determine libraries from filename prefix.
    if(_name MATCHES "^ex_array_")
        set(_libs "dnds")
    elseif(_name MATCHES "^ex_geom_")
        set(_libs "geom;dnds")
    elseif(_name MATCHES "^ex_euler_")
        set(_libs "euler_library_NS;euler_library_fast_NS;cfv;geom;dnds")
    else()
        # ex_cfv_* and anything else
        set(_libs "cfv;geom;dnds")
    endif()

    add_executable(${_name} ${_src})
    target_include_directories(${_name} PRIVATE
        ${DNDS_EXTERNAL_INCLUDES}
        ${DNDS_INCLUDES})
    target_link_libraries(${_name} PRIVATE ${_libs} ${DNDS_EXTERNAL_LIBS})
    if(DNDS_USE_CCACHE)
        set_property(TARGET ${_name} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
    endif()
    set_target_properties(${_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        EXCLUDE_FROM_ALL ON)

    list(APPEND _example_targets ${_name})
endforeach()

# Aggregate target: cmake --build build -t examples
add_custom_target(examples DEPENDS ${_example_targets})

message(STATUS "Examples: ${_example_targets}")
