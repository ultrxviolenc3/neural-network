cmake_minimum_required(VERSION 3.10)
project(neyro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(data SHARED src/data.cpp)
target_include_directories(data PUBLIC include)

add_library(generator SHARED src/generator.cpp)
target_include_directories(generator PUBLIC include)

add_library(neuralnetwork SHARED src/neuralnetwork.cpp)
target_include_directories(neuralnetwork PUBLIC include)
target_link_libraries(neuralnetwork PRIVATE data)

add_executable(neyro main.cpp)
target_link_libraries(neyro PRIVATE neuralnetwork data generator)
