if(NOT DEFINED PRODUCER_BUILD_DIR)
    message(FATAL_ERROR "PRODUCER_BUILD_DIR is required")
endif()
if(NOT DEFINED CONSUMER_SOURCE_DIR)
    message(FATAL_ERROR "CONSUMER_SOURCE_DIR is required")
endif()

set(INSTALL_PREFIX "${PRODUCER_BUILD_DIR}/cpptensor-install-smoke")
set(CONSUMER_BINARY_DIR "${PRODUCER_BUILD_DIR}/cpptensor-consumer-smoke")

file(REMOVE_RECURSE "${INSTALL_PREFIX}" "${CONSUMER_BINARY_DIR}")

set(INSTALL_CMD "${CMAKE_COMMAND}" --install "${PRODUCER_BUILD_DIR}" --prefix "${INSTALL_PREFIX}")
if(BUILD_CONFIG)
    list(APPEND INSTALL_CMD --config "${BUILD_CONFIG}")
endif()
execute_process(
    COMMAND ${INSTALL_CMD}
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_stdout
    ERROR_VARIABLE install_stderr
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Install step failed:\n${install_stdout}\n${install_stderr}")
endif()

set(CONFIGURE_CMD
    "${CMAKE_COMMAND}"
    -S "${CONSUMER_SOURCE_DIR}"
    -B "${CONSUMER_BINARY_DIR}"
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}
)
if(DEFINED GENERATOR AND NOT GENERATOR STREQUAL "")
    list(APPEND CONFIGURE_CMD -G "${GENERATOR}")
endif()

execute_process(
    COMMAND ${CONFIGURE_CMD}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Consumer configure failed:\n${configure_stdout}\n${configure_stderr}")
endif()

set(BUILD_CMD "${CMAKE_COMMAND}" --build "${CONSUMER_BINARY_DIR}")
if(BUILD_CONFIG)
    list(APPEND BUILD_CMD --config "${BUILD_CONFIG}")
endif()

execute_process(
    COMMAND ${BUILD_CMD}
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_stdout
    ERROR_VARIABLE build_stderr
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Consumer build failed:\n${build_stdout}\n${build_stderr}")
endif()
