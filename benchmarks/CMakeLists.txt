# ==== CPU benchmark ====
add_executable(cpptensor_bench_cpu cpu_bench.cpp)
target_link_libraries(cpptensor_bench_cpu PRIVATE cpptensor benchmark::benchmark benchmark::benchmark_main)
cpptensor_apply_build_profile(cpptensor_bench_cpu)
cpptensor_apply_build_safety_flags(cpptensor_bench_cpu)

# ==== AVX2 benchmark ====
if (CPPTENSOR_ENABLE_AVX2)
    add_executable(cpptensor_bench_avx2 avx2_bench.cpp)
    target_link_libraries(cpptensor_bench_avx2 PRIVATE cpptensor benchmark::benchmark benchmark::benchmark_main)
    target_compile_options(cpptensor_bench_avx2 PRIVATE ${CPPTENSOR_AVX2_COMPILE_FLAGS})
    cpptensor_apply_build_profile(cpptensor_bench_avx2)
    cpptensor_apply_build_safety_flags(cpptensor_bench_avx2)
endif()

# ==== AVX512 benchmark ====
if (CPPTENSOR_ENABLE_AVX512)
    add_executable(cpptensor_bench_avx512 avx512_bench.cpp)
    target_link_libraries(cpptensor_bench_avx512 PRIVATE cpptensor benchmark::benchmark benchmark::benchmark_main)
    target_compile_options(cpptensor_bench_avx512 PRIVATE ${CPPTENSOR_AVX512_COMPILE_FLAGS})
    cpptensor_apply_build_profile(cpptensor_bench_avx512)
    cpptensor_apply_build_safety_flags(cpptensor_bench_avx512)
endif()

# ==== CUDA benchmark ====
if (BUILD_CUDA)
    add_executable(cpptensor_bench_cuda cuda_bench.cu)
    target_link_libraries(cpptensor_bench_cuda PRIVATE cpptensor benchmark::benchmark benchmark::benchmark_main CUDA::cudart)
    set_target_properties(cpptensor_bench_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    cpptensor_apply_build_profile(cpptensor_bench_cuda)
    cpptensor_apply_build_safety_flags(cpptensor_bench_cuda)
endif()
