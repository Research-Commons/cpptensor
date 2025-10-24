#include <benchmark/benchmark.h>
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/backend/isa/avx2.hpp"
#include "cpptensor/ops/math/abs.hpp"
#include "cpptensor/ops/activation/relu.hpp"
#include "cpptensor/ops/activation/sigmoid.hpp"
#include "cpptensor/ops/arithmetic/add.hpp"
#include "cpptensor/ops/arithmetic/div.hpp"
#include "cpptensor/ops/arithmetic/neg.hpp"
#include "cpptensor/ops/arithmetic/mul.hpp"
#include "cpptensor/ops/arithmetic/pow.hpp"
#include "cpptensor/ops/arithmetic/sub.hpp"
#include "cpptensor/ops/math/abs.hpp"
#include "cpptensor/ops/math/cos.hpp"
#include "cpptensor/ops/math/log.hpp"
#include "cpptensor/ops/math/exp.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/ops/math/sin.hpp"
#include "cpptensor/ops/math/sqrt.hpp"
#include "cpptensor/ops/math/tan.hpp"

using namespace cpptensor;

static void BM_Add_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::add_f32_avx2);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::mul_f32_avx2);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Exp_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Exp, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::exp_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 1.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::exp(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Log_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Log, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::log_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 1.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::log(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Pow_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Pow, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::pow_f32_avx2);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::pow(A, B);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Abs_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Abs, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::abs_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, -5.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::abs(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sqrt_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Sqrt, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::sqrt_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::sqrt(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sin_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Sin, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::sin_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::sin(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Cos_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Cos, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::cos_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::cos(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Tan_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Tan, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::tan_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::tan(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sigmoid_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Sigmoid, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::sigmoid_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::sigmoid(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Relu_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Relu, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::relu_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::relu(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Matmul_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Matmul, DeviceType::CPU, CpuIsa::AVX2, cpptensor::AVX2::matmul_f32_avx2);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_Add_AVX2);
BENCHMARK(BM_Mul_AVX2);
BENCHMARK(BM_Exp_AVX2);
BENCHMARK(BM_Log_AVX2);
BENCHMARK(BM_Pow_AVX2);
BENCHMARK(BM_Abs_AVX2);
BENCHMARK(BM_Sqrt_AVX2);
BENCHMARK(BM_Sin_AVX2);
BENCHMARK(BM_Cos_AVX2);
BENCHMARK(BM_Tan_AVX2);
BENCHMARK(BM_Sigmoid_AVX2);
BENCHMARK(BM_Relu_AVX2);
BENCHMARK(BM_Matmul_AVX2);

BENCHMARK_MAIN();
