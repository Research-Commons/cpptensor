# Collect all example source files in this directory
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS *.cpp)

# Try to locate the profiler library (if enabled globally)
if(CPPTENSOR_ENABLE_GPERFTOOLS)
    find_library(PROFILER_LIB profiler)
    if(PROFILER_LIB)
        message(STATUS "✅ Found libprofiler: ${PROFILER_LIB}")
    else()
        message(WARNING "⚠️  libprofiler not found, CPPTENSOR_ENABLE_GPERFTOOLS will be ignored for examples.")
    endif()
endif()

foreach(file ${EXAMPLE_SOURCES})
    get_filename_component(example ${file} NAME_WE)

    # Build one executable per example file
    add_executable(${example} ${file})
    target_link_libraries(${example} PRIVATE cpptensor)
    cpptensor_apply_build_profile(${example})
    cpptensor_apply_build_safety_flags(${example})

    # Link profiler if requested
    if(CPPTENSOR_ENABLE_GPERFTOOLS AND PROFILER_LIB)
        target_link_libraries(${example} PRIVATE ${PROFILER_LIB})
        message(STATUS "Profiling enabled for example: ${example}")
    endif()
endforeach()
