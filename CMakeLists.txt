cmake_minimum_required(VERSION 3.15)
project(hand_detector C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# final executable
add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME} PUBLIC detector_lib ds_arena)

# the memory arena allocator library
add_subdirectory(memory)

# Main library (tensor + loss, no main())
add_library(detector_lib
    src/tensor.c
    src/loss.c
)
target_include_directories(detector_lib PUBLIC src/include/)
target_link_libraries(detector_lib PUBLIC ds_arena m) # link the detector_library with the ds_arena allocator library.

# Find Criterion
find_package(PkgConfig REQUIRED)
pkg_check_modules(CRITERION REQUIRED criterion)

if(CRITERION_FOUND)
    enable_testing()

    add_executable(test_tensor tests/test_tensor.c)
    target_link_libraries(test_tensor detector_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_tensor PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME tensor_tests COMMAND test_tensor)

    add_executable(test_loss tests/test_loss.c)
    target_link_libraries(test_loss detector_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_loss PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME loss_tests COMMAND test_loss)
endif()
