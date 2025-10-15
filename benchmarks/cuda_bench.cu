#include <cuda_runtime_api.h>
#include <benchmark/benchmark.h>
#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/backend/cuda_backend.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"

using namespace cppgrad;

static void BM_Add_CUDA(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CUDA, CUDA::addKernel);
    Tensor A({2048, 2048}, 5.f, false, DeviceType::CUDA);
    Tensor B({2048, 2048}, 5.f, false, DeviceType::CUDA);
    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
        cudaDeviceSynchronize();
    }
}

static void BM_Mul_CUDA(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CUDA, CUDA::mulKernel);
    Tensor A({2048, 2048}, 5.f, false, DeviceType::CUDA);
    Tensor B({2048, 2048}, 5.f, false, DeviceType::CUDA);
    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
        cudaDeviceSynchronize();
    }
}

BENCHMARK(BM_Add_CUDA);
BENCHMARK(BM_Mul_CUDA);
BENCHMARK_MAIN();
