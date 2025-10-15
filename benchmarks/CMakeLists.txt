# ==== CPU benchmark ====
add_executable(cppgrad_bench_cpu cpu_bench.cpp)
target_link_libraries(cppgrad_bench_cpu PRIVATE cppgrad benchmark::benchmark benchmark::benchmark_main)
target_compile_options(cppgrad_bench_cpu PRIVATE -O3)

# ==== AVX2 benchmark ====
if (BUILD_AVX2)
    add_executable(cppgrad_bench_avx2 avx2_bench.cpp)
    target_link_libraries(cppgrad_bench_avx2 PRIVATE cppgrad benchmark::benchmark benchmark::benchmark_main)
    target_compile_options(cppgrad_bench_avx2 PRIVATE -O3 -mavx2 -mfma)
endif()

# ==== AVX512 benchmark ====
if (BUILD_AVX512)
    add_executable(cppgrad_bench_avx512 avx512_bench.cpp)
    target_link_libraries(cppgrad_bench_avx512 PRIVATE cppgrad benchmark::benchmark benchmark::benchmark_main)
    target_compile_options(cppgrad_bench_avx512 PRIVATE -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma)
endif()

# ==== CUDA benchmark ====
if (BUILD_CUDA)
    add_executable(cppgrad_bench_cuda cuda_bench.cu)
    target_link_libraries(cppgrad_bench_cuda PRIVATE cppgrad benchmark::benchmark benchmark::benchmark_main CUDA::cudart)
    set_target_properties(cppgrad_bench_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(cppgrad_bench_cuda PRIVATE -O3)
endif()

