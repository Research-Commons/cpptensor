#include <benchmark/benchmark.h>
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/ops/math/abs.hpp"
#include "cpptensor/ops/activation/relu.hpp"
#include "cpptensor/ops/activation/sigmoid.hpp"
#include "cpptensor/ops/arithmetic/pow.hpp"
#include "cpptensor/ops/arithmetic/sub.hpp"
#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/ops/math/abs.hpp"
#include "cpptensor/ops/math/cos.hpp"
#include "cpptensor/ops/math/log.hpp"
#include "cpptensor/ops/math/exp.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/ops/math/sin.hpp"
#include "cpptensor/ops/math/sqrt.hpp"
#include "cpptensor/ops/math/tan.hpp"

using namespace cpptensor;

static void BM_Add_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Add, DeviceType::CPU, CPU::addKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A + B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mul_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Mul, DeviceType::CPU, CPU::mulKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A * B;
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Exp_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Exp, DeviceType::CPU, CPU::expKernel);

    Tensor A = Tensor::full({2048, 2048}, 1.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::exp(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Log_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(
        OpType::Log, DeviceType::CPU, CPU::logKernel);

    Tensor A = Tensor::full({2048, 2048}, 1.0f, DeviceType::CPU);

    for (auto _ : state) {
        Tensor C = cpptensor::log(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Pow_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Pow, DeviceType::CPU, CPU::powKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::pow(A, B);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Abs_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(OpType::Abs, DeviceType::CPU, CPU::absKernel);
    Tensor A = Tensor::full({2048, 2048}, -5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::abs(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sqrt_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(OpType::Sqrt, DeviceType::CPU, CPU::sqrtKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::sqrt(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sin_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(OpType::Sin, DeviceType::CPU, CPU::sinKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::sin(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Cos_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(OpType::Cos, DeviceType::CPU, CPU::cosKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::cos(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Tan_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(OpType::Tan, DeviceType::CPU, CPU::tanKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::tan(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sigmoid_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(OpType::Sigmoid, DeviceType::CPU, CPU::sigmoidKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::sigmoid(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Relu_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerUnaryKernel(OpType::Relu, DeviceType::CPU, CPU::reluKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::relu(A);
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Matmul_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Matmul, DeviceType::CPU, CPU::gemmf32kernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    Tensor B = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
}

// Performance Results (1M element vectors)
// Backend	Time (ns)	Speedup vs CPU
// CPU	372,199	1.0× (baseline)
// AVX2	119,997	3.1× faster
// AVX512	80,309	4.6× faster

static void BM_Dot_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerKernel(OpType::Dot, DeviceType::CPU, CPU::dotKernel);
    Tensor A = Tensor::full({1048576}, 1.0f, DeviceType::CPU);  // 1M elements
    Tensor B = Tensor::full({1048576}, 1.0f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = cpptensor::dot(A, B);
        benchmark::DoNotOptimize(C);
    }
}

// Reduction Operations
static void BM_Sum_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Sum, DeviceType::CPU, CPU::sumKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.sum();  // Global sum
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Sum_Dim_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Sum, DeviceType::CPU, CPU::sumKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.sum(0);  // Sum along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mean_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Mean, DeviceType::CPU, CPU::meanKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.mean();  // Global mean
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Mean_Dim_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Mean, DeviceType::CPU, CPU::meanKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.mean(0);  // Mean along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Max_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Max, DeviceType::CPU, CPU::maxKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.max();  // Global max
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Max_Dim_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Max, DeviceType::CPU, CPU::maxKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.max(0);  // Max along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Min_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Min, DeviceType::CPU, CPU::minKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.min();  // Global min
        benchmark::DoNotOptimize(C);
    }
}

static void BM_Min_Dim_CPU(benchmark::State& state) {
    KernelRegistry::instance().registerReductionKernel(OpType::Min, DeviceType::CPU, CPU::minKernel);
    Tensor A = Tensor::full({2048, 2048}, 5.f, DeviceType::CPU);
    for (auto _ : state) {
        Tensor C = A.min(0);  // Min along dimension 0
        benchmark::DoNotOptimize(C);
    }
}

BENCHMARK(BM_Add_CPU);
BENCHMARK(BM_Mul_CPU);
BENCHMARK(BM_Exp_CPU);
BENCHMARK(BM_Log_CPU);
BENCHMARK(BM_Pow_CPU);
BENCHMARK(BM_Abs_CPU);
BENCHMARK(BM_Sqrt_CPU);
BENCHMARK(BM_Sin_CPU);
BENCHMARK(BM_Cos_CPU);
BENCHMARK(BM_Tan_CPU);
BENCHMARK(BM_Sigmoid_CPU);
BENCHMARK(BM_Relu_CPU);
BENCHMARK(BM_Matmul_CPU);
BENCHMARK(BM_Dot_CPU);
BENCHMARK(BM_Sum_CPU);
BENCHMARK(BM_Sum_Dim_CPU);
BENCHMARK(BM_Mean_CPU);
BENCHMARK(BM_Mean_Dim_CPU);
BENCHMARK(BM_Max_CPU);
BENCHMARK(BM_Max_Dim_CPU);
BENCHMARK(BM_Min_CPU);
BENCHMARK(BM_Min_Dim_CPU);
BENCHMARK_MAIN();