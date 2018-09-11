cmake_minimum_required(VERSION 3.5)
project(CAP)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp CAP.cpp guidedfilter.cpp)
include_directories("./")
link_directories("./")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS -march=native)

add_executable(CAP ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
target_link_libraries(CAP ${OpenCV_LIBS})
