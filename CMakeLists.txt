cmake_minimum_required(VERSION 3.16)
project(cvrptw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS ON)  # Enable GNU extensions for SHORT_CIRCUIT macro
if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
endif()

file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ROOT)
message(STATUS "Got root of repository: ${ROOT}")

file(GLOB_RECURSE COMMON CONFIGURE_DEPENDS "${ROOT}/src/common/*.cpp")

add_executable(branch_and_bound "${ROOT}/src/branch_and_bound.cpp" "${COMMON}")
target_include_directories(
    branch_and_bound PRIVATE
    "${ROOT}/include"
)
target_precompile_headers(branch_and_bound PRIVATE "${ROOT}/include/pch.hpp")

add_executable(recursion "${ROOT}/src/recursion.cpp" "${COMMON}")
target_include_directories(
    recursion PRIVATE
    "${ROOT}/include"
)
target_precompile_headers(recursion PRIVATE "${ROOT}/include/pch.hpp")
