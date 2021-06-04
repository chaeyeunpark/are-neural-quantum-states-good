cmake_minimum_required(VERSION 3.9)

project(are-neural-quantum-good)


add_subdirectory("${PROJECT_SOURCE_DIR}/yannq")

set(YANNQ_DIR "${PROJECT_SOURCE_DIR}/yannq")
include_directories(${YANNQ_DIR}/externals)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

link_libraries(Eigen3::Eigen)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DEIGEN_USE_MKL_ALL")


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	include_directories("$ENV{MKLROOT}/include")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
	link_libraries(mkl_intel_lp64 mkl_tbb_thread mkl_core tbb pthread m dl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -tbb")
	include_directories("$ENV{MKLROOT}/include")
	link_libraries(tbb stdc++ pthread m dl)
endif()

add_executable(RBM_XXZ_SR "src/RBM_XXZ_SR.cpp")
target_link_libraries(RBM_XXZ_SR PRIVATE yannq)

add_executable(RBM_XXZ_ER "src/RBM_XXZ_ER.cpp")
target_link_libraries(RBM_XXZ_ER PRIVATE yannq)

add_executable(RBM_J1J2_SR "src/RBM_J1J2_SR.cpp")
target_link_libraries(RBM_J1J2_SR PRIVATE yannq)

add_executable(RBM_J1J2_ER "src/RBM_J1J2_ER.cpp")
target_link_libraries(RBM_J1J2_ER PRIVATE yannq)

add_executable(RBM_TXYZ_SR "src/RBM_TXYZ_SR.cpp")
target_link_libraries(RBM_TXYZ_SR PRIVATE yannq)

add_executable(RBM_TXYZ_ER "src/RBM_TXYZ_ER.cpp")
target_link_libraries(RBM_TXYZ_ER PRIVATE yannq)

