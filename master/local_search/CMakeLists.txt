cmake_minimum_required(VERSION 3.16)
project(cvrptw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS ON)

file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ROOT)
message(STATUS "Got root of repository: ${ROOT}")

file(GLOB_RECURSE COMMON CONFIGURE_DEPENDS "${ROOT}/src/common/*.cpp")

add_library(local_search SHARED "${ROOT}/src/local_search.cpp" "${COMMON}")
target_include_directories(
    local_search PRIVATE
    "${ROOT}/include"
)
target_precompile_headers(local_search PRIVATE "${ROOT}/include/pch.hpp")
target_compile_definitions(local_search PRIVATE LOCAL_SEARCH_EXPORTS)
