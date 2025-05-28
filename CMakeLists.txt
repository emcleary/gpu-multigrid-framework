cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_experimenting LANGUAGES CXX CUDA)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "-O3 -Wall -Wno-array-bounds -DNDEBUG")
endif (CMAKE_BUILD_TYPE STREQUAL "Release")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "-g -O0 -Wall -Wno-array-bounds")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 3012") # a volatile destination type for a compound assignment expression is deprecated
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

############
# Packages #
############

find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)

#############
# Libraries #
#############

add_subdirectory(examples)
add_subdirectory(performance)

add_subdirectory(src)
include_directories(src)

###############
# Executables #
###############

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# add_executable(main main.cpp)
# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(main PUBLIC CUDA::cudart CUDA::cublas)
# if (TARGET Src)
#   target_link_libraries(main PUBLIC Src)
# endif()
