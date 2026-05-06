cmake_minimum_required(VERSION 3.20)

# ==== User Options ====
option(BUILD_CUDA "Use CUDA kernels" OFF)
option(USE_STD_SIMD "Use C++ std::simd kernels instead of AVX intrinsics" OFF)
option(USE_OPENBLAS "Use OpenBLAS if found on this system" ON)
option(CPPTENSOR_INSTALL_PKGCONFIG "Install pkg-config metadata for cpptensor" ON)
option(CPPTENSOR_BUILD_TESTS "Build cpptensor test targets" OFF)
option(CPPTENSOR_BUILD_EXAMPLES "Build cpptensor example targets" OFF)
option(CPPTENSOR_BUILD_BENCHMARKS "Build cpptensor benchmark targets" OFF)
option(CPPTENSOR_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(CPPTENSOR_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)
option(CPPTENSOR_ENABLE_TSAN "Enable ThreadSanitizer instrumentation" OFF)
option(CPPTENSOR_ENABLE_STRICT_WARNINGS "Enable stricter compiler warning profile for cpptensor targets" OFF)
option(CPPTENSOR_WARNINGS_AS_ERRORS "Treat enabled warnings as errors for cpptensor targets" OFF)
option(CPPTENSOR_ENABLE_PROFILING "Enable profiling-friendly frame-pointer settings" OFF)
option(CPPTENSOR_ENABLE_LTO "Enable interprocedural optimization (LTO) for Release/RelWithDebInfo builds" OFF)
option(CPPTENSOR_ENABLE_GPERFTOOLS "Link example binaries with libprofiler when available" OFF)

project(cpptensor VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CPPTENSOR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/cpptensor" CACHE PATH
        "Install path for cpptensor CMake package metadata")

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "Configuring cpptensor for build type: ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Configuring cpptensor for multi-config generator: ${CMAKE_CONFIGURATION_TYPES}")
endif()

set(CPPTENSOR_LTO_SUPPORTED FALSE)
if(CPPTENSOR_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT CPPTENSOR_LTO_SUPPORTED OUTPUT CPPTENSOR_LTO_ERROR LANGUAGES CXX)
    if(NOT CPPTENSOR_LTO_SUPPORTED)
        message(FATAL_ERROR "CPPTENSOR_ENABLE_LTO=ON requested, but IPO/LTO is not supported: ${CPPTENSOR_LTO_ERROR}")
    endif()
endif()

if(BUILD_CUDA)
    if(APPLE)
        message(WARNING "CUDA is not supported in this build on Apple platforms. Forcing BUILD_CUDA OFF.")
        set(BUILD_CUDA OFF CACHE BOOL "Use CUDA kernels" FORCE)
    else()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
endif()

include(CheckCXXSourceCompiles)

set(CPPTENSOR_SANITIZER_COMPILE_OPTIONS "")
set(CPPTENSOR_SANITIZER_LINK_OPTIONS "")
set(CPPTENSOR_WARNING_OPTIONS "")
set(CPPTENSOR_WARNING_ERROR_OPTION "")
set(CPPTENSOR_ENABLED_SANITIZERS "")

if(CPPTENSOR_ENABLE_ASAN AND CPPTENSOR_ENABLE_TSAN)
    message(FATAL_ERROR "CPPTENSOR_ENABLE_ASAN and CPPTENSOR_ENABLE_TSAN cannot both be ON in the same build.")
endif()

set(CPPTENSOR_SANITIZER_REQUESTED FALSE)
if(CPPTENSOR_ENABLE_ASAN OR CPPTENSOR_ENABLE_UBSAN OR CPPTENSOR_ENABLE_TSAN)
    set(CPPTENSOR_SANITIZER_REQUESTED TRUE)
endif()

if(CPPTENSOR_SANITIZER_REQUESTED AND BUILD_CUDA)
    message(FATAL_ERROR "Sanitizer builds currently require BUILD_CUDA=OFF in cpptensor.")
endif()

if(CPPTENSOR_SANITIZER_REQUESTED)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "Sanitizers are currently supported only with GCC/Clang-style compilers. Detected: ${CMAKE_CXX_COMPILER_ID}")
    endif()

    if(CPPTENSOR_ENABLE_ASAN)
        list(APPEND CPPTENSOR_ENABLED_SANITIZERS address)
        list(APPEND CPPTENSOR_SANITIZER_COMPILE_OPTIONS -fsanitize=address)
        list(APPEND CPPTENSOR_SANITIZER_LINK_OPTIONS -fsanitize=address)
    endif()

    if(CPPTENSOR_ENABLE_UBSAN)
        list(APPEND CPPTENSOR_ENABLED_SANITIZERS undefined)
        list(APPEND CPPTENSOR_SANITIZER_COMPILE_OPTIONS -fsanitize=undefined)
        list(APPEND CPPTENSOR_SANITIZER_LINK_OPTIONS -fsanitize=undefined)
    endif()

    if(CPPTENSOR_ENABLE_TSAN)
        list(APPEND CPPTENSOR_ENABLED_SANITIZERS thread)
        list(APPEND CPPTENSOR_SANITIZER_COMPILE_OPTIONS -fsanitize=thread)
        list(APPEND CPPTENSOR_SANITIZER_LINK_OPTIONS -fsanitize=thread)
    endif()

    list(APPEND CPPTENSOR_SANITIZER_COMPILE_OPTIONS -fno-omit-frame-pointer)
endif()

if(CPPTENSOR_ENABLE_STRICT_WARNINGS)
    if(MSVC)
        list(APPEND CPPTENSOR_WARNING_OPTIONS /W4 /permissive-)
    else()
        list(APPEND CPPTENSOR_WARNING_OPTIONS
            -Wall
            -Wextra
            -Wpedantic
            -Wformat=2
            -Wnull-dereference
            -Wnon-virtual-dtor
        )
    endif()
endif()

if(CPPTENSOR_WARNINGS_AS_ERRORS)
    if(MSVC)
        set(CPPTENSOR_WARNING_ERROR_OPTION /WX)
    else()
        set(CPPTENSOR_WARNING_ERROR_OPTION -Werror)
    endif()
endif()

function(cpptensor_apply_build_profile target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Debug>:/Od /Zi /RTC1>
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:/O2 /Zi /DNDEBUG>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Debug>:-O0 -g3>
            $<$<CONFIG:Release>:-O3 -DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
        )
    endif()

    if(CPPTENSOR_ENABLE_PROFILING)
        if(MSVC)
            target_compile_options(${target_name} PRIVATE /Oy-)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target_name} PRIVATE -fno-omit-frame-pointer)
        endif()
    endif()

    if(CPPTENSOR_ENABLE_LTO AND CPPTENSOR_LTO_SUPPORTED)
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    endif()
endfunction()

function(cpptensor_apply_build_safety_flags target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "cpptensor_apply_build_safety_flags called for unknown target '${target_name}'")
    endif()

    if(CPPTENSOR_WARNING_OPTIONS)
        set(_warning_compile_options "")
        foreach(_warning_option IN LISTS CPPTENSOR_WARNING_OPTIONS)
            list(APPEND _warning_compile_options "$<$<COMPILE_LANGUAGE:CXX>:${_warning_option}>")
        endforeach()
        target_compile_options(${target_name} PRIVATE ${_warning_compile_options})
    endif()

    if(NOT CPPTENSOR_WARNING_ERROR_OPTION STREQUAL "")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CPPTENSOR_WARNING_ERROR_OPTION}>")
    endif()

    if(CPPTENSOR_SANITIZER_COMPILE_OPTIONS)
        set(_sanitizer_compile_options "")
        foreach(_sanitizer_option IN LISTS CPPTENSOR_SANITIZER_COMPILE_OPTIONS)
            list(APPEND _sanitizer_compile_options "$<$<COMPILE_LANGUAGE:CXX>:${_sanitizer_option}>")
        endforeach()
        target_compile_options(${target_name} PRIVATE ${_sanitizer_compile_options})
    endif()

    if(CPPTENSOR_SANITIZER_LINK_OPTIONS)
        target_link_options(${target_name} PRIVATE ${CPPTENSOR_SANITIZER_LINK_OPTIONS})
    endif()
endfunction()

if(CPPTENSOR_ENABLED_SANITIZERS)
    list(JOIN CPPTENSOR_ENABLED_SANITIZERS ", " CPPTENSOR_ENABLED_SANITIZERS_DISPLAY)
else()
    set(CPPTENSOR_ENABLED_SANITIZERS_DISPLAY "none")
endif()
message(STATUS "Sanitizers enabled: ${CPPTENSOR_ENABLED_SANITIZERS_DISPLAY}")
message(STATUS "Strict warnings profile: ${CPPTENSOR_ENABLE_STRICT_WARNINGS} (warnings-as-errors=${CPPTENSOR_WARNINGS_AS_ERRORS})")

function(cpptensor_check_source_compiles required_flags source out_var)
    set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${required_flags}")
    check_cxx_source_compiles("${source}" ${out_var})
    set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")
    set(${out_var} "${${out_var}}" PARENT_SCOPE)
endfunction()

set(CPPTENSOR_EFFECTIVE_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(DEFINED CPPTENSOR_SYSTEM_PROCESSOR_OVERRIDE AND NOT CPPTENSOR_SYSTEM_PROCESSOR_OVERRIDE STREQUAL "")
    set(CPPTENSOR_EFFECTIVE_SYSTEM_PROCESSOR "${CPPTENSOR_SYSTEM_PROCESSOR_OVERRIDE}")
endif()

string(TOLOWER "${CPPTENSOR_EFFECTIVE_SYSTEM_PROCESSOR}" CPPTENSOR_SYSTEM_PROCESSOR_LOWER)
set(CPPTENSOR_TARGET_IS_X86 FALSE)
if(CPPTENSOR_SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|x86_64h|amd64|x64|i[3-6]86|x86)$")
    set(CPPTENSOR_TARGET_IS_X86 TRUE)
endif()

set(CPPTENSOR_AVX2_SUPPORTED FALSE)
set(CPPTENSOR_AVX512_SUPPORTED FALSE)
set(CPPTENSOR_AVX2_UNSUPPORTED_REASON "target architecture '${CPPTENSOR_EFFECTIVE_SYSTEM_PROCESSOR}' is not x86/x86_64")
set(CPPTENSOR_AVX512_UNSUPPORTED_REASON "target architecture '${CPPTENSOR_EFFECTIVE_SYSTEM_PROCESSOR}' is not x86/x86_64")
set(CPPTENSOR_AVX2_COMPILE_FLAGS "")
set(CPPTENSOR_AVX512_COMPILE_FLAGS "")
set(CPPTENSOR_AVX2_REQUIRED_FLAGS "")
set(CPPTENSOR_AVX512_REQUIRED_FLAGS "")

if(CPPTENSOR_TARGET_IS_X86)
    if(MSVC)
        set(CPPTENSOR_AVX2_COMPILE_FLAGS /arch:AVX2)
        set(CPPTENSOR_AVX512_COMPILE_FLAGS /arch:AVX512)
    else()
        set(CPPTENSOR_AVX2_COMPILE_FLAGS -mavx2 -mfma)
        set(CPPTENSOR_AVX512_COMPILE_FLAGS -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma)
    endif()

    string(JOIN " " CPPTENSOR_AVX2_REQUIRED_FLAGS ${CPPTENSOR_AVX2_COMPILE_FLAGS})
    string(JOIN " " CPPTENSOR_AVX512_REQUIRED_FLAGS ${CPPTENSOR_AVX512_COMPILE_FLAGS})

    cpptensor_check_source_compiles(
        "${CPPTENSOR_AVX2_REQUIRED_FLAGS}"
        "#include <immintrin.h>\nint main() { __m256 a = _mm256_set1_ps(1.0f); __m256 b = _mm256_fmadd_ps(a, a, a); return static_cast<int>(_mm256_cvtss_f32(b)); }"
        CPPTENSOR_COMPILER_SUPPORTS_AVX2)
    if(CPPTENSOR_COMPILER_SUPPORTS_AVX2)
        set(CPPTENSOR_AVX2_SUPPORTED TRUE)
        set(CPPTENSOR_AVX2_UNSUPPORTED_REASON "")
    else()
        set(CPPTENSOR_AVX2_UNSUPPORTED_REASON "compiler '${CMAKE_CXX_COMPILER_ID}' rejected AVX2/FMA flags '${CPPTENSOR_AVX2_REQUIRED_FLAGS}'")
    endif()

    cpptensor_check_source_compiles(
        "${CPPTENSOR_AVX512_REQUIRED_FLAGS}"
        "#include <immintrin.h>\nalignas(64) float out[16];\nint main() { __m512 a = _mm512_set1_ps(1.0f); __m512 b = _mm512_fmadd_ps(a, a, a); _mm512_store_ps(out, b); return static_cast<int>(out[0]); }"
        CPPTENSOR_COMPILER_SUPPORTS_AVX512)
    if(CPPTENSOR_COMPILER_SUPPORTS_AVX512)
        set(CPPTENSOR_AVX512_SUPPORTED TRUE)
        set(CPPTENSOR_AVX512_UNSUPPORTED_REASON "")
    else()
        set(CPPTENSOR_AVX512_UNSUPPORTED_REASON "compiler '${CMAKE_CXX_COMPILER_ID}' rejected AVX-512/FMA flags '${CPPTENSOR_AVX512_REQUIRED_FLAGS}'")
    endif()
endif()

set(CPPTENSOR_BUILD_AVX2_DEFAULT OFF)
if(CPPTENSOR_AVX2_SUPPORTED)
    set(CPPTENSOR_BUILD_AVX2_DEFAULT ON)
endif()
option(BUILD_AVX2 "Build AVX2 optimized kernels (object library)" ${CPPTENSOR_BUILD_AVX2_DEFAULT})

set(CPPTENSOR_BUILD_AVX512_DEFAULT OFF)
if(CPPTENSOR_AVX512_SUPPORTED)
    set(CPPTENSOR_BUILD_AVX512_DEFAULT ON)
endif()
option(BUILD_AVX512 "Build AVX-512 optimized kernels (object library)" ${CPPTENSOR_BUILD_AVX512_DEFAULT})

set(CPPTENSOR_ENABLE_AVX2 ${BUILD_AVX2})
if(BUILD_AVX2 AND NOT CPPTENSOR_AVX2_SUPPORTED)
    message(FATAL_ERROR "BUILD_AVX2=ON requested, but AVX2 kernels are unavailable: ${CPPTENSOR_AVX2_UNSUPPORTED_REASON}")
endif()

set(CPPTENSOR_ENABLE_AVX512 ${BUILD_AVX512})
if(BUILD_AVX512 AND NOT CPPTENSOR_AVX512_SUPPORTED)
    message(FATAL_ERROR "BUILD_AVX512=ON requested, but AVX-512 kernels are unavailable: ${CPPTENSOR_AVX512_UNSUPPORTED_REASON}")
endif()

message(STATUS "Target processor: ${CPPTENSOR_EFFECTIVE_SYSTEM_PROCESSOR}")
message(STATUS "AVX2 kernels: requested=${BUILD_AVX2}, enabled=${CPPTENSOR_ENABLE_AVX2}")
if(NOT CPPTENSOR_AVX2_SUPPORTED)
    message(STATUS "AVX2 unavailable: ${CPPTENSOR_AVX2_UNSUPPORTED_REASON}")
endif()
message(STATUS "AVX-512 kernels: requested=${BUILD_AVX512}, enabled=${CPPTENSOR_ENABLE_AVX512}")
if(NOT CPPTENSOR_AVX512_SUPPORTED)
    message(STATUS "AVX-512 unavailable: ${CPPTENSOR_AVX512_UNSUPPORTED_REASON}")
endif()
message(STATUS "Build profile options: profiling=${CPPTENSOR_ENABLE_PROFILING}, lto=${CPPTENSOR_ENABLE_LTO}, gperftools_examples=${CPPTENSOR_ENABLE_GPERFTOOLS}")

if(BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA support enabled.")
else()
    message(STATUS "CUDA support disabled.")
endif()

# ==== Optional development targets ====
if(CPPTENSOR_BUILD_TESTS)
    message(STATUS "cpptensor tests: ENABLED")
    add_subdirectory(third_party/Catch2)
else()
    message(STATUS "cpptensor tests: DISABLED")
endif()

if(CPPTENSOR_BUILD_BENCHMARKS)
    message(STATUS "cpptensor benchmarks: ENABLED")
    message(STATUS "Adding Google Benchmark from third_party/googlebenchmark")
    set(BENCHMARK_ENABLE_TESTING OFF) # disable GB's own tests
    set(BENCHMARK_ENABLE_INSTALL OFF)
    set(BENCHMARK_ENABLE_WERROR OFF)
    add_subdirectory(third_party/googlebenchmark)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        foreach(_benchmark_target benchmark benchmark_main)
            if(TARGET ${_benchmark_target})
                target_compile_options(${_benchmark_target} PRIVATE -Wno-c2y-extensions)
            endif()
        endforeach()
    endif()
else()
    message(STATUS "cpptensor benchmarks: DISABLED")
endif()

if(CPPTENSOR_BUILD_EXAMPLES)
    message(STATUS "cpptensor examples: ENABLED")
else()
    message(STATUS "cpptensor examples: DISABLED")
endif()

# ==== Optional OpenBLAS Detection ====
if(USE_OPENBLAS)
    find_package(OpenBLAS QUIET)

    if(OpenBLAS_FOUND)
        message(STATUS "✅ Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
        message(STATUS "    Includes: ${OpenBLAS_INCLUDE_DIRS}")
        set(HAVE_OPENBLAS TRUE)
    else()
        message(STATUS "⚠️ OpenBLAS not found. Building without BLAS support.")
        set(HAVE_OPENBLAS FALSE)
    endif()
else()
    set(HAVE_OPENBLAS FALSE)
endif()

# ==== Source Files ====
file(GLOB_RECURSE ALL_CPP CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
if(BUILD_CUDA)
    file(GLOB_RECURSE ALL_CUDA CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cu)
else()
    set(ALL_CUDA "")
endif()

set(ALL_SRC ${ALL_CPP} ${ALL_CUDA})

# ==== Explicitly exclude ISA source files from generic compilation ====
# (We will add them separately as object libraries based on the effective ISA configuration)
list(FILTER ALL_SRC EXCLUDE REGEX ".*/backend/isa/avx2.*\\.cpp$")
list(FILTER ALL_SRC EXCLUDE REGEX ".*/backend/isa/avx512.*\\.cpp$")

file(GLOB AVX2_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/backend/isa/*avx2.cpp)
file(GLOB AVX512_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/backend/isa/*avx512.cpp)

# ==== Create object libraries for ISA specializations ====
if(CPPTENSOR_ENABLE_AVX2 AND AVX2_SRC)
    add_library(cpptensor_isa_avx2 OBJECT ${AVX2_SRC})
    target_compile_options(cpptensor_isa_avx2 PRIVATE ${CPPTENSOR_AVX2_COMPILE_FLAGS})
    target_include_directories(cpptensor_isa_avx2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
    cpptensor_apply_build_profile(cpptensor_isa_avx2)
    cpptensor_apply_build_safety_flags(cpptensor_isa_avx2)
endif()

if(CPPTENSOR_ENABLE_AVX512 AND AVX512_SRC)
    add_library(cpptensor_isa_avx512 OBJECT ${AVX512_SRC})
    target_compile_options(cpptensor_isa_avx512 PRIVATE ${CPPTENSOR_AVX512_COMPILE_FLAGS})
    target_include_directories(cpptensor_isa_avx512 PRIVATE ${CMAKE_SOURCE_DIR}/include)
    cpptensor_apply_build_profile(cpptensor_isa_avx512)
    cpptensor_apply_build_safety_flags(cpptensor_isa_avx512)
endif()

# ==== Create final static library, including object libs if present ====
set(cpptensor_OBJ_SOURCES ${ALL_SRC})

# Construct the add_library call with OBJECTS expanded if they exist
if(CPPTENSOR_ENABLE_AVX2 AND AVX2_SRC)
    list(APPEND cpptensor_OBJ_SOURCES $<TARGET_OBJECTS:cpptensor_isa_avx2>)
endif()
if(CPPTENSOR_ENABLE_AVX512 AND AVX512_SRC)
    list(APPEND cpptensor_OBJ_SOURCES $<TARGET_OBJECTS:cpptensor_isa_avx512>)
endif()

# ==== Create Static Library ====
add_library(cpptensor STATIC ${cpptensor_OBJ_SOURCES})
add_library(cpptensor::cpptensor ALIAS cpptensor)
cpptensor_apply_build_safety_flags(cpptensor)

# ==== Link OpenBLAS ====
set(CPPTENSOR_PACKAGE_NEEDS_OPENBLAS OFF)
if(HAVE_OPENBLAS)
    if(TARGET OpenBLAS::OpenBLAS)
        target_link_libraries(cpptensor PUBLIC OpenBLAS::OpenBLAS)
        set(CPPTENSOR_PACKAGE_NEEDS_OPENBLAS ON)
    else()
        target_include_directories(cpptensor PUBLIC ${OpenBLAS_INCLUDE_DIRS})
        target_link_libraries(cpptensor PUBLIC ${OpenBLAS_LIBRARIES})
    endif()
endif()

# ==== Pass build options as compiler definitions ====
target_compile_definitions(cpptensor PUBLIC
        $<$<BOOL:${CPPTENSOR_ENABLE_AVX2}>:BUILD_AVX2>
        $<$<BOOL:${CPPTENSOR_ENABLE_AVX512}>:BUILD_AVX512>
        $<$<BOOL:${BUILD_CUDA}>:BUILD_CUDA>
        $<$<BOOL:${HAVE_OPENBLAS}>:USE_OPENBLAS>
        #$<$<BOOL:${USE_STD_SIMD}>:USE_STD_SIMD>
)

set_target_properties(cpptensor PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
cpptensor_apply_build_profile(cpptensor)

# ==== Include Directories ====
target_include_directories(cpptensor
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cpptensor>
)

## ==== Link Cuda Backend ====
if(BUILD_CUDA)
    set_target_properties(cpptensor PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(cpptensor PUBLIC CUDA::cudart)
endif()

# ==== Install + Package Export ====
install(TARGETS cpptensor
        EXPORT cpptensorTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(CPPTENSOR_PACKAGE_NEEDS_CUDA ${BUILD_CUDA})

configure_package_config_file(
        ${CMAKE_SOURCE_DIR}/cmake/cpptensorConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cpptensorConfig.cmake
        INSTALL_DESTINATION ${CPPTENSOR_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cpptensorConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cpptensorConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cpptensorConfigVersion.cmake
        DESTINATION ${CPPTENSOR_INSTALL_CMAKEDIR}
)

install(EXPORT cpptensorTargets
        FILE cpptensorTargets.cmake
        NAMESPACE cpptensor::
        DESTINATION ${CPPTENSOR_INSTALL_CMAKEDIR}
)

if(CPPTENSOR_INSTALL_PKGCONFIG)
    configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/cpptensor.pc.in
            ${CMAKE_CURRENT_BINARY_DIR}/cpptensor.pc
            @ONLY
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpptensor.pc
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

# ==== Enable Testing ====
if(CPPTENSOR_BUILD_TESTS)
    enable_testing()
endif()

# ==== Add optional subdirectories====
if(CPPTENSOR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(CPPTENSOR_BUILD_TESTS)
    add_subdirectory(test)
endif()

if(CPPTENSOR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
