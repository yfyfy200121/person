cmake_minimum_required(VERSION 3.4.1)
project(rknn_yolov5_3588_bytetrack)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wl,--allow-shlib-undefined")

# 安装路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_3588_new_${CMAKE_SYSTEM_NAME})
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 查找系统安装的 rknnrt 库
find_library(RKNN_RT_LIB rknnrt PATHS /usr/lib REQUIRED)

# 包含头文件路径
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  /usr/include/rknn
  /usr/local/include/eigen3
)

# OpenCV
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  set(OpenCV_DIR /usr/local/share/opencv4)
endif()
find_package(OpenCV REQUIRED)

# 源码
add_executable(rknn_yolov5_3588_bytetrack
  src/bytetrack.cpp
  src/BYTETracker.cpp
  src/kalmanFilter.cpp
  src/lapjv.cpp
  src/STrack.cpp
  src/utils.cpp
)

# 链接库
target_link_libraries(rknn_yolov5_3588_bytetrack
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  pthread
)

# 安装目标
install(TARGETS rknn_yolov5_3588_bytetrack DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
