cmake_minimum_required(VERSION 3.16)
project(neyro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

find_package(Boost REQUIRED COMPONENTS program_options)

add_library(data SHARED source/data.cpp)
target_include_directories(data PUBLIC include)

add_library(generator SHARED source/generator.cpp)
target_include_directories(generator PUBLIC include)

add_library(neuralnetwork SHARED source/neuralnetwork.cpp)
target_include_directories(neuralnetwork PUBLIC include)
target_link_libraries(neuralnetwork PRIVATE data)

add_executable(neyro main.cpp)
target_link_libraries(neyro PRIVATE neuralnetwork data generator Boost::program_options)

add_executable(neyro_tests test.cpp)
target_link_libraries(neyro_tests PRIVATE neuralnetwork data generator)

enable_testing()
add_test(NAME neuralnetwork_tests COMMAND neyro_tests)