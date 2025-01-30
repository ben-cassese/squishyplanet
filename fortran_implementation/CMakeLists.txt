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
    squishyplanet.f90
    constants.f90
    keplerian.f90
    parametric_ellipse.f90
    intersection_pts.f90
    integrals.f90
)

target_link_libraries(local_lapack_routines PUBLIC blas)
target_link_libraries(squishyplanet PRIVATE
    local_lapack_routines  # This already includes blas
    local_quadpack
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -fcheck=all -fbacktrace -g -O0 -Wall -Wextra -fimplicit-none")
endif()
