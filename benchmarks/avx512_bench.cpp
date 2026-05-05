#include <benchmark/benchmark.h>
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/backend/isa/avx512.hpp"
#include "cpptensor/ops/linearAlgebra/dot.hpp"
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

static void BM_Dot_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Dot, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::dot_f32_avx512);
    Tensor A = Tensor::full({1048576}, 1.0f, DeviceType::CPU);  // 1M elements
    Tensor B = Tensor::full({1048576}, 1.0f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::dot(A, B);
        benchmark::DoNotOptimize(C);
    }
}

// Reduction Operations
static void BM_Sum_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Sum, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::sum_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.sum();  // Global sum
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sum_Dim_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Sum, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::sum_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.sum(0);  // Sum along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mean_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Mean, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::mean_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.mean();  // Global mean
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mean_Dim_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Mean, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::mean_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.mean(0);  // Mean along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Max_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Max, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::max_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.max();  // Global max
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Max_Dim_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Max, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::max_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.max(0);  // Max along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Min_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Min, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::min_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.min();  // Global min
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Min_Dim_AVX512(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Min, DeviceType::CPU, CpuIsa::AVX512, cpptensor::AVX512::min_f32_avx512);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.min(0);  // Min along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_Add_AVX512);
BENCHMARK(BM_Mul_AVX512);
BENCHMARK(BM_Matmul_AVX512);
BENCHMARK(BM_Dot_AVX512);
BENCHMARK(BM_Sum_AVX512);
BENCHMARK(BM_Sum_Dim_AVX512);
BENCHMARK(BM_Mean_AVX512);
BENCHMARK(BM_Mean_Dim_AVX512);
BENCHMARK(BM_Max_AVX512);
BENCHMARK(BM_Max_Dim_AVX512);
BENCHMARK(BM_Min_AVX512);
BENCHMARK(BM_Min_Dim_AVX512);
BENCHMARK_MAIN();