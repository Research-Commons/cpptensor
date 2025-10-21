# Collect all example source files in this directory
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS *.cpp)

# Option to enable profiling (default: OFF)
option(ENABLE_PROFILING "Link executables with gperftools profiler" OFF)

# Try to locate the profiler library (if enabled)
if(ENABLE_PROFILING)
    find_library(PROFILER_LIB profiler)
    if(PROFILER_LIB)
        message(STATUS "✅ Found libprofiler: ${PROFILER_LIB}")
    else()
        message(WARNING "⚠️  libprofiler not found, profiling disabled.")
        set(ENABLE_PROFILING OFF)
    endif()
endif()

foreach(file ${EXAMPLE_SOURCES})
    get_filename_component(example ${file} NAME_WE)

    # Build one executable per example file
    add_executable(${example} ${file})
    target_link_libraries(${example} PRIVATE cpptensor)

    # Ensure debug symbols and frame pointers for profiling accuracy
    target_compile_options(${example} PRIVATE -g -fno-omit-frame-pointer -O2)

    # Link profiler if requested
    if(ENABLE_PROFILING AND PROFILER_LIB)
        target_link_libraries(${example} PRIVATE ${PROFILER_LIB})
        message(STATUS "Profiling enabled for example: ${example}")
    endif()
endforeach()