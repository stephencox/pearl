cmake_minimum_required(VERSION 3.10.0)

project(pearl)
set(CMAKE_C_STANDARD 11)

enable_testing()

include_directories(external/Unity/src)
include_directories(external/parson)
include_directories(src)

add_library(pearl
        external/parson/parson.c
        src/pearl_activation.c
        src/pearl_json.c
        src/pearl_layer.c
        src/pearl_loss.c
        src/pearl_network.c
        src/pearl_print.c
        src/pearl_tensor.c
        src/pearl_util.c
        src/pearl_version.c)

add_executable(pearl-test test/main.c external/Unity/src/unity.c)
add_executable(pearl-example example/main.c)

target_link_libraries(pearl-test pearl m)
target_link_libraries(pearl-example pearl m)

add_test(NAME test COMMAND pearl-test)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
add_compile_options(-O3 -Wall -Wextra -pedantic)
