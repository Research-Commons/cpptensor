#include <benchmark/benchmark.h>
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/backend/isa/avx512.hpp"
#include "cpptensor/ops/math/matmul.hpp"

using namespace cpptensor;

static void BM_Add_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::add_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::mul_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Matmul_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Matmul, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::gemm_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_Add_AVX512);
BENCHMARK(BM_Mul_AVX512);
BENCHMARK(BM_Matmul_AVX512);
BENCHMARK_MAIN();