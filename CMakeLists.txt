cmake_minimum_required(VERSION 3.15)
project(hand_detector C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# the memory arena allocator library
add_subdirectory(memory)

# the stb image library
add_subdirectory(externals)

# final executable
add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME} PUBLIC detector_lib stb_img ds_arena OpenMP::OpenMP_C)

# find openmp
find_package(OpenMP REQUIRED)

# find sdl3 + webcam test executable.
find_package(SDL3 REQUIRED CONFIG)
add_executable(webcam_test webcam_test.c)
target_link_libraries(webcam_test PUBLIC detector_lib ds_arena stb_img m OpenMP::OpenMP_C SDL3::SDL3)

# Main library (tensor + loss, no main())
add_library(detector_lib
    src/tensor.c
    src/loss.c
    src/layers.c
    src/network.c
    src/optimizer.c
    src/dataloader.c
    src/augmentation.c
    src/preprocess.c
    src/bn_fold.c
    src/checkpoint.c
    src/inference.c
)
target_include_directories(detector_lib PUBLIC src/include/)
target_link_libraries(detector_lib PUBLIC ds_arena stb_img m OpenMP::OpenMP_C) # link the detector_library with the ds_arena allocator library.

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

    add_executable(test_layers tests/test_layers.c)
    target_link_libraries(test_layers detector_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_layers PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME layers_tests COMMAND test_layers)

    add_executable(test_network tests/test_network.c)
    target_link_libraries(test_network detector_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_network PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME network_tests COMMAND test_network)

    add_executable(test_dataloader tests/test_dataloader.c)
    target_link_libraries(test_dataloader detector_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_dataloader PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME dataloader_tests COMMAND test_dataloader)

    add_executable(test_augmentation tests/test_augmentation.c)
    target_link_libraries(test_augmentation detector_lib ${CRITERION_LIBRARIES} m)
    target_include_directories(test_augmentation PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME augmentation_tests COMMAND test_augmentation)

    add_executable(test_preprocess tests/test_preprocess.c)
    target_link_libraries(test_preprocess detector_lib ${CRITERION_LIBRARIES} m stb_img)
    target_include_directories(test_preprocess PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME preprocess_tests COMMAND test_preprocess)

    add_executable(test_checkpoint tests/test_checkpoint.c)
    target_link_libraries(test_checkpoint detector_lib ${CRITERION_LIBRARIES} m stb_img)
    target_include_directories(test_checkpoint PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME checkpoint_tests COMMAND test_checkpoint)

    add_executable(test_bn_fold tests/test_bn_fold.c)
    target_link_libraries(test_bn_fold detector_lib ${CRITERION_LIBRARIES} m stb_img)
    target_include_directories(test_bn_fold PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME bn_fold_tests COMMAND test_bn_fold)

    add_executable(test_inference tests/test_inference.c)
    target_link_libraries(test_inference detector_lib ${CRITERION_LIBRARIES} m stb_img)
    target_include_directories(test_inference PRIVATE ${CRITERION_INCLUDE_DIRS})
    add_test(NAME inference_tests COMMAND test_inference)
endif()
