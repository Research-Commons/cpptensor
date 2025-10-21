#include <benchmark/benchmark.h>
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/backend/isa/avx2.hpp"
#include "cpptensor/ops/abs.hpp"
#include "cpptensor/ops/cos.hpp"
#include "cpptensor/ops/exp.hpp"
#include "cpptensor/ops/log.hpp"
#include "cpptensor/ops/pow.hpp"
#include "cpptensor/ops/relu.hpp"
#include "cpptensor/ops/sigmoid.hpp"
#include "cpptensor/ops/sin.hpp"
#include "cpptensor/ops/sqrt.hpp"
#include "cpptensor/ops/tan.hpp"

using namespace cpptensor;

static void BM_Add_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cpptensor::add_f32_avx2);
    Tensor A = Tensor::full({2048, 2048}, 5.f, false, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, false, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cpptensor::mul_f32_avx2);
    Tensor A = Tensor::full({2048, 2048}, 5.f, false, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, false, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Exp_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Exp, DeviceType::CPU, CpuIsa::AVX2, cpptensor::exp_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 1.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::exp(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Log_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Log, DeviceType::CPU, CpuIsa::AVX2, cpptensor::log_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 1.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::log(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Pow_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Pow, DeviceType::CPU, CpuIsa::AVX2, cpptensor::pow_f32_avx2);
    Tensor A = Tensor::full({2048, 2048}, 5.f, false, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, false, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::pow(A, B);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Abs_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Abs, DeviceType::CPU, CpuIsa::AVX2, cpptensor::abs_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, -5.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::abs(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sqrt_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Sqrt, DeviceType::CPU, CpuIsa::AVX2, cpptensor::sqrt_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::sqrt(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sin_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Sin, DeviceType::CPU, CpuIsa::AVX2, cpptensor::sin_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::sin(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Cos_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Cos, DeviceType::CPU, CpuIsa::AVX2, cpptensor::cos_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::cos(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Tan_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Tan, DeviceType::CPU, CpuIsa::AVX2, cpptensor::tan_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::tan(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sigmoid_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Sigmoid, DeviceType::CPU, CpuIsa::AVX2, cpptensor::sigmoid_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::sigmoid(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Relu_AVX2(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Relu, DeviceType::CPU, CpuIsa::AVX2, cpptensor::relu_f32_avx2);

    Tensor A = Tensor::full({2048, 2048}, 5.0f, false, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::relu(A);
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

BENCHMARK_MAIN();
