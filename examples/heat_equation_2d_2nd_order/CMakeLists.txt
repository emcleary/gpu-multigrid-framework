set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples)
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
file(GLOB SRC *.cpp)
file(GLOB CUDA *.cu)

add_executable(${TARGET_NAME} ${SRC} ${CUDA})
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${TARGET_NAME} PUBLIC ../..)
target_link_libraries(${TARGET_NAME} PUBLIC Src ExUtil CUDA::cudart CUDA::cublas)
target_link_libraries(${TARGET_NAME} PRIVATE ${OpenMP_CXX_LIBRARIES})


file (GLOB_RECURSE headers CONFIGURE_DEPENDS "*.hpp")
file (GLOB_RECURSE source CONFIGURE_DEPENDS "*.cpp")
file (GLOB_RECURSE source_cuda CONFIGURE_DEPENDS "*.cu")

set(LIBRARY HeatEquation2D)
add_library(${LIBRARY} STATIC ${source} ${source_cuda} ${headers})
target_include_directories(${LIBRARY} PUBLIC ..)
target_link_libraries(${LIBRARY} PUBLIC CUDA::cublas CUDA::cudart Src)
target_link_libraries(${LIBRARY} PRIVATE ${OpenMP_CXX_LIBRARIES})
