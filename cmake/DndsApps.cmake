# cmake/DndsApps.cmake
# Application executable lists and the ADD_EXE_APP helper function.
# Must be included after all library targets are defined.

# -------------------------------------------------------------------
# App executable lists
# -------------------------------------------------------------------

## app exes

set(DNDS_APPS_EXTERNAL
cgns_APITest
eigen_Test
STL_Test
json_Test
cgal_AABBTest
mpi_test
)

set(DNDS_APPS_EXTERNAL_CU
cuda_test
)

set(DNDS_APPS_DNDS
array_Test
arrayTrans_test
arrayDerived_test
arrayDOF_test
serializerJSON_Test
serializerH5_Test
stdPowerTest
objectPool_test
)

set(DNDS_APPS_DNDS_CU
array_cuda_Test
array_cuda_Bench
arrayDOF_test_cuda
)

set(DNDS_APPS_Geom
elements_Test
meshSerial_Test
ofReader_Test
partitionMeshSerial
)

set(DNDS_APPS_CFV
vrStatic_Test
vrBasic_Test
diffTensors_Test
)

set(DNDS_APPS_Solver
krylovTest
)



set(DNDS_APPS_Euler
gasTest
jacobiLUTest
oneDimProfileTest
)

# -------------------------------------------------------------------
# Symlinks for in-place development
# -------------------------------------------------------------------
if (NOT SKBUILD_PROJECT_NAME) # do not create links for SCIKIT-BUILD build
    install(CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/DNDSR/bin ${CMAKE_SOURCE_DIR}/src/bin SYMBOLIC)" COMPONENT py)
    install(CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/DNDSR/lib ${CMAKE_SOURCE_DIR}/src/lib SYMBOLIC)" COMPONENT py)
endif()
# Stub generation runs automatically as the final step of
# `cmake --install build --component py` (see DndsTooling.cmake).
# The generated .pyi files in python/DNDSR/ should be committed to git.

# -------------------------------------------------------------------
# ADD_EXE_APP helper function
# -------------------------------------------------------------------
set(EXE_SUFFIX "")
if (UNIX)
    set(EXE_SUFFIX ".exe")
endif()


function(ADD_EXE_APP EXES MAIN_DIR LIBS USE_EXCLUDE_FROM_ALL SRC_SUFFIX)
    message(STATUS "To add exes: ${EXES} with libs: ${LIBS}")
    foreach(EXE ${EXES})
        # add_executable(${EXE} app/external/${EXE}.cpp $<TARGET_OBJECTS:${OBJS}>)
        add_executable(${EXE} ${MAIN_DIR}/${EXE}.${SRC_SUFFIX})
        if(USE_EXCLUDE_FROM_ALL)
            set_target_properties(${EXE} PROPERTIES EXCLUDE_FROM_ALL ON)
        endif()
        target_link_libraries(${EXE} PUBLIC ${LIBS})
        target_link_libraries(${EXE} PUBLIC ${DNDS_EXTERNAL_LIBS})
        target_include_directories(${EXE} PUBLIC ${DNDS_EXTERNAL_INCLUDES} PUBLIC ${DNDS_INCLUDES})
        set_target_properties(${EXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/app RUNTIME_OUTPUT_NAME "${EXE}${EXE_SUFFIX}")
        if(DNDS_RECORD_COMMIT)
            target_compile_definitions(${EXE} PUBLIC DNDS_CURRENT_COMMIT_HASH=${DNDS_RECORDED_COMMIT_HASH})
        endif()
        target_compile_definitions(${EXE} PUBLIC DNDS_VERSION_STRING="${DNDS_VERSION_FULL}")
        if( DNDS_USE_CCACHE )
            set_property(TARGET ${EXE} PROPERTY C_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
            set_property(TARGET ${EXE} PROPERTY CXX_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
            set_property(TARGET ${EXE} PROPERTY CUDA_COMPILER_LAUNCHER ${DNDS_CCACHE_EXEC})
        endif()
    endforeach()
endfunction(ADD_EXE_APP)

# -------------------------------------------------------------------
# Register all app executables
# -------------------------------------------------------------------

# In topolocical order of the dependency graph
# euler_library
# euler_library_fast
# cfv
# geom
# dnds

## Mind That the TOPOLOGICAL ORDER should be obeyed!
ADD_EXE_APP("${DNDS_APPS_EXTERNAL}" "app/external" ";" ON cpp)
ADD_EXE_APP("${DNDS_APPS_DNDS}" "app/DNDS" "dnds;" ON cpp)
if(DNDS_USE_CUDA)
    ADD_EXE_APP("${DNDS_APPS_EXTERNAL_CU}" "app/external" ";" ON cu)
    ADD_EXE_APP("${DNDS_APPS_DNDS_CU}" "app/DNDS" "dnds;" ON cu)
endif()
ADD_EXE_APP("${DNDS_APPS_Solver}" "app/Solver" "dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_Geom}" "app/Geom" "geom;dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_CFV}" "app/CFV" "cfv;geom;dnds;" ON cpp)
ADD_EXE_APP("${DNDS_APPS_Euler}" "app/Euler" "cfv;geom;dnds;" ON cpp)



set(DNDS_APPS_Euler_Models)

foreach(item IN LISTS DNDS_Euler_Models_List)
    string(REPLACE "=" ";" keyval ${item})
    list(GET keyval 0 key)
    list(GET keyval 1 value)
    set(EXE_NAME "euler${value}")
    list(APPEND DNDS_APPS_Euler_Models "${EXE_NAME}")
    message(DEBUG "${keyval} --- ${value} --- ${EXE_NAME}")
    ## Mind That the TOPOLOGICAL ORDER should be obeyed!
    ADD_EXE_APP("${EXE_NAME}" "app/Euler" "euler_library_${key};euler_library_fast_${key};cfv;geom;dnds;" ON cpp)
    ## This works because cmake detects dependency and reorders the libraries
    # ADD_EXE_APP("${EXE_NAME}" "app/Euler" "dnds;geom;cfv;euler_library_fast_${key};euler_library_${key};")
endforeach()
