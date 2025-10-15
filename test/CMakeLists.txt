# Collect all .cpp test files in this directory
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Create a single test executable with all test files
add_executable(cpptensor_test ${TEST_SOURCES})

# Link with main cpptensor lib and Catch2
target_link_libraries(cpptensor_test PRIVATE cpptensor Catch2::Catch2WithMain)

# Enable test discovery via CTest and Catch2
include(CTest)
include(Catch)
catch_discover_tests(cpptensor_test)