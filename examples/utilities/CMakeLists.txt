file (GLOB_RECURSE headers CONFIGURE_DEPENDS "*.hpp")
file (GLOB_RECURSE source CONFIGURE_DEPENDS "*.cpp")
file (GLOB_RECURSE source_cuda CONFIGURE_DEPENDS "*.cu")

set(LIBRARY ExUtil)
add_library(${LIBRARY} STATIC ${source} ${source_cuda} ${headers})
target_include_directories(${LIBRARY} PUBLIC ../..)
target_link_libraries(${LIBRARY} PUBLIC CUDA::cublas CUDA::cudart)
