#include <benchmark/benchmark.h>
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/backend/isa/avx2.hpp"

using namespace cpptensor;

static void BM_Add_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cpptensor::add_f32_avx2);
    Tensor A({2048, 2048}, 5.f, false, DeviceType::CPU);
    Tensor B({2048, 2048}, 5.f, false, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cpptensor::mul_f32_avx2);
    Tensor A({2048, 2048}, 5.f, false, DeviceType::CPU);
    Tensor B({2048, 2048}, 5.f, false, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_Add_AVX2);
BENCHMARK(BM_Mul_AVX2);
BENCHMARK_MAIN();
