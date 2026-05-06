if(NOT DEFINED SOURCE_DIR OR NOT DEFINED BINARY_DIR OR NOT DEFINED GENERATOR OR NOT DEFINED PROCESSOR)
    message(FATAL_ERROR "SOURCE_DIR, BINARY_DIR, GENERATOR, and PROCESSOR are required.")
endif()

if(NOT DEFINED CXX_COMPILER OR CXX_COMPILER STREQUAL "")
    message(FATAL_ERROR "CXX_COMPILER is required.")
endif()

file(REMOVE_RECURSE "${BINARY_DIR}")

set(configure_cmd
    "${CMAKE_COMMAND}"
    -S "${SOURCE_DIR}"
    -B "${BINARY_DIR}"
    -G "${GENERATOR}"
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
    -DCPPTENSOR_SYSTEM_PROCESSOR_OVERRIDE=${PROCESSOR}
    -DBUILD_CUDA=OFF
    -DUSE_OPENBLAS=OFF
)

if(DEFINED FORCE_BUILD_AVX2 AND FORCE_BUILD_AVX2)
    list(APPEND configure_cmd -DBUILD_AVX2=ON)
endif()

if(DEFINED FORCE_BUILD_AVX512 AND FORCE_BUILD_AVX512)
    list(APPEND configure_cmd -DBUILD_AVX512=ON)
endif()

execute_process(
    COMMAND ${configure_cmd}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr
)

set(configure_output "${configure_stdout}${configure_stderr}")

if(DEFINED EXPECT_FAILURE AND EXPECT_FAILURE)
    if(configure_result EQUAL 0)
        message(FATAL_ERROR "Expected configure failure, but command succeeded. Output:\n${configure_output}")
    endif()
else()
    if(NOT configure_result EQUAL 0)
        message(FATAL_ERROR "Expected configure success, but command failed. Output:\n${configure_output}")
    endif()
endif()

if(DEFINED REQUIRED_PATTERNS AND NOT REQUIRED_PATTERNS STREQUAL "")
    foreach(required_pattern IN LISTS REQUIRED_PATTERNS)
        string(REGEX MATCH "${required_pattern}" matched "${configure_output}")
        if(NOT matched)
            message(FATAL_ERROR "Missing required output pattern '${required_pattern}'. Full output:\n${configure_output}")
        endif()
    endforeach()
endif()

message(STATUS "Validated configure output for processor '${PROCESSOR}'.")
