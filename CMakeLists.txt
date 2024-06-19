cmake_minimum_required(VERSION 3.7)
project(pearl)
set(CMAKE_C_STANDARD 11)

enable_testing()

include_directories(external/munit)
include_directories(external/parson)
include_directories(PearlLib/PearlLib)

add_library(pearl
        external/parson/parson.c
        PearlLib/PearlLib/pearl_activation.c
        PearlLib/PearlLib/pearl_json.c
        PearlLib/PearlLib/pearl_layer.c
        PearlLib/PearlLib/pearl_loss.c
        PearlLib/PearlLib/pearl_network.c
        PearlLib/PearlLib/pearl_print.c
        PearlLib/PearlLib/pearl_tensor.c
        PearlLib/PearlLib/pearl_util.c
        PearlLib/PearlLib/pearl_version.c)

add_executable(pearl-test PearlLib/PearlLibTest/main.c external/munit/munit.c)

target_link_libraries(pearl-test pearl m)

add_test(NAME test COMMAND pearl-test)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
