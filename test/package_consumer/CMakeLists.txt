cmake_minimum_required(VERSION 3.20)
project(cpptensor_package_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(cpptensor CONFIG REQUIRED)

add_executable(cpptensor_package_consumer main.cpp)
target_link_libraries(cpptensor_package_consumer PRIVATE cpptensor::cpptensor)
