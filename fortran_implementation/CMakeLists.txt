# Specify minimum CMake version
cmake_minimum_required(VERSION 3.12)

# Project name and language
project(squishyplanet Fortran)

# Find BLAS source files
file(GLOB BLAS
    "${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/BLAS/SRC/*.f"
)
add_library(blas STATIC ${BLAS})

file(GLOB_RECURSE LAPACK_ROUTINES
    "${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/lapack_routines/*.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/lapack_routines/*.f90"
)
add_library(local_lapack_routines STATIC ${LAPACK_ROUTINES})

file(GLOB QUADPACK
    "${CMAKE_CURRENT_SOURCE_DIR}/external_libraries/QUADPACK/*.F90"
)
add_library(local_quadpack STATIC ${QUADPACK})

# Add executable and specify source files
add_executable(squishyplanet
    main.f90
    read_in_files.f90
    model_types.f90
    squishyplanet_2d.f90
    constants.f90
    keplerian.f90
    parametric_ellipse.f90
    intersection_pts.f90
    integrals.f90
    planet_3d.f90
    squishyplanet_3d.f90
    squishyplanet_luna_interface.f90
)

add_library(squishyplanet_tmp  STATIC
    read_in_files.f90
    model_types.f90
    squishyplanet_2d.f90
    constants.f90
    keplerian.f90
    parametric_ellipse.f90
    intersection_pts.f90
    integrals.f90
    planet_3d.f90
    squishyplanet_3d.f90
    squishyplanet_luna_interface.f90
)

target_link_libraries(local_lapack_routines PUBLIC blas)
target_link_libraries(squishyplanet PRIVATE
    local_lapack_routines  # This already includes blas
    local_quadpack
)
target_link_libraries(squishyplanet_tmp PRIVATE
    local_lapack_routines  # This already includes blas
    local_quadpack
)


add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libsquishyplanet.a
    COMMAND ${CMAKE_COMMAND} -E make_directory temp_obj
    COMMAND cd temp_obj && ar x ${CMAKE_BINARY_DIR}/libsquishyplanet_tmp.a
    COMMAND cd temp_obj && ar x ${CMAKE_BINARY_DIR}/liblocal_quadpack.a
    COMMAND cd temp_obj && ar x ${CMAKE_BINARY_DIR}/liblocal_lapack_routines.a
    COMMAND cd temp_obj && ar x ${CMAKE_BINARY_DIR}/libblas.a
    COMMAND ar crs libsquishyplanet.a temp_obj/*.o
    COMMAND ${CMAKE_COMMAND} -E remove_directory temp_obj
    DEPENDS local_quadpack local_lapack_routines blas
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libsquishyplanet.a
        ${CMAKE_BINARY_DIR}/../libsquishyplanet.a
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(complete_library ALL DEPENDS ${CMAKE_BINARY_DIR}/libsquishyplanet.a)
