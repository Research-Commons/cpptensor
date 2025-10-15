cmake_minimum_required(VERSION 3.20)

# ==== Options ====
# Build AVX objects (compile-time flags). Turn off if compiler can't support or you want smaller binary.
option(BUILD_AVX2  "Build AVX2 optimized kernels (object library)"  OFF)
option(BUILD_AVX512 "Build AVX-512 optimized kernels (object library)" OFF)
option(BUILD_CUDA "Use CUDA kernels" OFF)
option(USE_STD_SIMD "Use C++ std::simd kernels instead of AVX intrinsics" OFF)

#Failsafe for mac users
if(APPLE)
    if(BUILD_AVX2 OR BUILD_AVX512)
        message(WARNING "AVX2/AVX512 not supported on Apple Silicon. Forcing them OFF.")
        set(BUILD_AVX2 OFF CACHE BOOL "" FORCE)
        set(BUILD_AVX512 OFF CACHE BOOL "" FORCE)
    endif()
    if(BUILD_CUDA)
        message(WARNING "NO CUDA ON MAC :/ . Forcing them OFF.")
        set(BUILD_CUDA OFF CACHE BOOL "" FORCE)
    endif ()
endif()

if(BUILD_CUDA)
    project(cpptensor LANGUAGES CXX CUDA)
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "Building cpptensor with build type: ${CMAKE_BUILD_TYPE}")
else()
    project(cpptensor LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 26)
    message(STATUS "Building cpptensor with build type: ${CMAKE_BUILD_TYPE}")
endif()

if(BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA support enabled.")
else()
    message(STATUS "CUDA support disabled.")
endif()

# ==== Catch2 Submodule ====
add_subdirectory(third_party/Catch2)

# ==== Google Benchmark Submodule ====
message(STATUS "Adding Google Benchmark from third_party/googlebenchmark")
set(BENCHMARK_ENABLE_TESTING OFF) # disable GB's own tests
set(BENCHMARK_ENABLE_INSTALL OFF)
add_subdirectory(third_party/googlebenchmark)

# ==== Source Files ====
file(GLOB_RECURSE ALL_CPP CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
if(BUILD_CUDA)
    file(GLOB_RECURSE ALL_CUDA CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cu)
else()
    set(ALL_CUDA "")
endif()

set(ALL_SRC ${ALL_CPP} ${ALL_CUDA})

# remove ISA path files from ALL_SRC so they are compiled separately with special flags
#list(FILTER ALL_SRC EXCLUDE REGEX ".*/src/backend/isa/.*")

# ==== Explicitly exclude ISA source files from generic compilation ====
# (We will add them separately as object libraries based on BUILD_AVX2 / BUILD_AVX512)
list(FILTER ALL_SRC EXCLUDE REGEX ".*/backend/isa/avx2.*\\.cpp$")
list(FILTER ALL_SRC EXCLUDE REGEX ".*/backend/isa/avx512.*\\.cpp$")

file(GLOB AVX2_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/backend/isa/*avx2.cpp)
file(GLOB AVX512_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/backend/isa/*avx512.cpp)

# ==== Create object libraries for ISA specializations ====
if(BUILD_AVX2 AND AVX2_SRC)
    add_library(cpptensor_isa_avx2 OBJECT ${AVX2_SRC})
    if(MSVC)
        target_compile_options(cpptensor_isa_avx2 PRIVATE /arch:AVX2)
    elseif(NOT APPLE)
        target_compile_options(cpptensor_isa_avx2 PRIVATE -mavx2 -mfma -O3)
    else()
        message(STATUS "Skipping AVX2 flags on macOS (unsupported)")
    endif()
    target_include_directories(cpptensor_isa_avx2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

if(BUILD_AVX512 AND AVX512_SRC)
    add_library(cpptensor_isa_avx512 OBJECT ${AVX512_SRC})
    if(MSVC)
        # MSVC's AVX-512 support varies by version/toolset. Use with caution.
        target_compile_options(cpptensor_isa_avx512 PRIVATE /arch:AVX512)
    elseif(NOT APPLE)
        target_compile_options(cpptensor_isa_avx512 PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -O3)
    else()
        message(STATUS "Skipping AVX-512 flags on macOS (unsupported)")
    endif()
    target_include_directories(cpptensor_isa_avx512 PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

# ==== Create final static library, including object libs if present ====
set(cpptensor_OBJ_SOURCES ${ALL_SRC})

# Construct the add_library call with OBJECTS expanded if they exist
if(BUILD_AVX2 AND AVX2_SRC)
    list(APPEND cpptensor_OBJ_SOURCES $<TARGET_OBJECTS:cpptensor_isa_avx2>)
endif()
if(BUILD_AVX512 AND AVX512_SRC)
    list(APPEND cpptensor_OBJ_SOURCES $<TARGET_OBJECTS:cpptensor_isa_avx512>)
endif()

# ==== Create Static Library ====
add_library(cpptensor STATIC ${cpptensor_OBJ_SOURCES})

# ==== Pass build options as compiler definitions ====
target_compile_definitions(cpptensor PUBLIC
        $<$<BOOL:${BUILD_AVX2}>:BUILD_AVX2>
        $<$<BOOL:${BUILD_AVX512}>:BUILD_AVX512>
        $<$<BOOL:${BUILD_CUDA}>:BUILD_CUDA>
        #$<$<BOOL:${USE_STD_SIMD}>:USE_STD_SIMD>
)

set_target_properties(cpptensor PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

# ==== Include Directories ====
target_include_directories(cpptensor
        PUBLIC  ${CMAKE_SOURCE_DIR}/include            # Enables: #include "cpptensor/tensor/..."
        PRIVATE ${CMAKE_SOURCE_DIR}/include/cpptensor    # Enables: #include "tensor/..."
)

## ==== Link Cuda Backend ====
if(BUILD_CUDA)
    set_target_properties(cpptensor PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(cpptensor PUBLIC CUDA::cudart)
endif()

# ==== Enable Testing ====
enable_testing()

# ==== Add subdirectories====
add_subdirectory(examples)
add_subdirectory(test)
add_subdirectory(benchmarks)