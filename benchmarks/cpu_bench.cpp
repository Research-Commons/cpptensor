#include <benchmark/benchmark.h>
#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"
#include "cppgrad/backend/cpu_backend.h"

using namespace cppgrad;

static void BM_Add_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CPU::addKernel);
    Tensor A({2048, 2048}, 5.f, false, DeviceType::CPU);
    Tensor B({2048, 2048}, 5.f, false, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CPU::mulKernel);
    Tensor A({2048, 2048}, 5.f, false, DeviceType::CPU);
    Tensor B({2048, 2048}, 5.f, false, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_Add_CPU);
BENCHMARK(BM_Mul_CPU);
BENCHMARK_MAIN();