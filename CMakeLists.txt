cmake_minimum_required(VERSION 2.8)
project(SFRSS)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

set(CMAKE_CXX_STANDARD 11)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
  option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
endif()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -g -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O3 -march=native")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
if (OpenCV_VERSION VERSION_LESS 3.4)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.4")
endif()

configure_file(${PROJECT_SOURCE_DIR}/src/EstimateDepths/include/libsgm_config.h.in
               ${PROJECT_SOURCE_DIR}/src/EstimateDepths/include/libsgm_config.h
               ${PROJECT_SOURCE_DIR}/src/EstimateDepths/include/stereomatch.h)

find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system thread) 
find_package(Eigen3 3.1.0 REQUIRED)
# find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/EstimateDepths
${PROJECT_SOURCE_DIR}/EstimateMotion
${PROJECT_SOURCE_DIR}/System
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(src/EstimateDepths)
include_directories(
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
) 

add_library(${PROJECT_NAME} SHARED  # SHARED: Dynamic link
src/System/System.cc
src/EstimateMotion/ORBextractor.cc
src/EstimateMotion/ORBmatcher.cc
src/EstimateMotion/Frame.cc
src/EstimateMotion/Initializer.cc
src/EstimateMotion/EstMotion.cc
src/EstimateMotion/MinimalSolver.cc
src/EstimateDepths/stereomatch.cpp
src/RefineMotion/RefineMotion.cpp
src/RefineMotion/CoarseInitializer.cpp
src/RefineMotion/PixelSelector2.cpp
src/util/settings.cpp
src/util/FramePym.cpp
src/util/globalCalib.cpp
src/util/ImageRW_OpenCV.cpp
src/util/ImageDisplay_OpenCV.cpp
src/util/DUtils/Random.cpp
src/util/DUtils/Timestamp.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${CUDA_LIBRARIES}
)

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

cuda_add_executable(SFRSS_EXE ./src/main.cc)
target_link_libraries(SFRSS_EXE ${PROJECT_NAME} sgm boost_system) 
