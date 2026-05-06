# Collect all .cpp test files in this directory
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/lazy_init_smoke\\.cpp$")

# Create a single test executable with all test files
add_executable(cpptensor_test ${TEST_SOURCES})

# Link with main cpptensor lib and Catch2
target_link_libraries(cpptensor_test PRIVATE cpptensor Catch2::Catch2WithMain)

# Enable test discovery via CTest and Catch2
include(CTest)
include(Catch)
catch_discover_tests(cpptensor_test)

add_executable(cpptensor_lazy_init_smoke lazy_init_smoke.cpp)
target_link_libraries(cpptensor_lazy_init_smoke PRIVATE cpptensor)
add_test(NAME cpptensor_lazy_init_smoke COMMAND cpptensor_lazy_init_smoke)

if(CMAKE_GENERATOR)
    add_test(
        NAME cmake_configure_disables_avx_on_aarch64
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DBINARY_DIR=${CMAKE_BINARY_DIR}/cmake-configure-aarch64
            -DGENERATOR=${CMAKE_GENERATOR}
            -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DPROCESSOR=aarch64
            -DREQUIRED_PATTERNS=AVX2\ kernels:\ requested=OFF,\ enabled=OFF;AVX-512\ kernels:\ requested=OFF,\ enabled=OFF
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake_configure_guard.cmake
    )

    add_test(
        NAME cmake_rejects_forced_avx2_on_aarch64
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DBINARY_DIR=${CMAKE_BINARY_DIR}/cmake-configure-aarch64-avx2-on
            -DGENERATOR=${CMAKE_GENERATOR}
            -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DPROCESSOR=aarch64
            -DEXPECT_FAILURE=ON
            -DFORCE_BUILD_AVX2=ON
            -DREQUIRED_PATTERNS=BUILD_AVX2=ON\ requested,\ but\ AVX2\ kernels\ are\ unavailable
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake_configure_guard.cmake
    )
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME benchmark_regression_check_passes
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/benchmarks/compare_benchmark_results.py
            --baseline ${CMAKE_SOURCE_DIR}/test/data/benchmarks/baseline.json
            --candidate ${CMAKE_SOURCE_DIR}/test/data/benchmarks/candidate-ok.json
            --max-regression-pct 5
    )

    add_test(
        NAME benchmark_regression_check_detects_slowdown
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/benchmarks/compare_benchmark_results.py
            --baseline ${CMAKE_SOURCE_DIR}/test/data/benchmarks/baseline.json
            --candidate ${CMAKE_SOURCE_DIR}/test/data/benchmarks/candidate-regression.json
            --max-regression-pct 5
    )
    set_tests_properties(benchmark_regression_check_detects_slowdown PROPERTIES WILL_FAIL TRUE)
endif()
