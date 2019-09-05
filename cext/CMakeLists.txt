cmake_minimum_required(VERSION 3.5)

add_definitions(-DGOOGLE_CUDA)

find_package(CUDA REQUIRED)

include_directories($ENV{TF_INC})
include_directories($ENV{TF_INC}/external/nsync/public)
include_directories("/usr/local")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

file(GLOB_RECURSE cuda_srcs src/*.cu.cc)
file(GLOB_RECURSE srcs src/*.cc)
foreach(val ${cuda_srcs})
    set_source_files_properties(${val} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endforeach(val)

link_directories($ENV{TF_LINK})

cuda_add_library(
    primitive_gen ${srcs} SHARED
)
target_link_libraries(primitive_gen tensorflow_framework ${OPENGL_LIBRARIES})

# Put the compiled library in the python package folder, rather than whatever build folder is being used
set_target_properties(
        primitive_gen PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)