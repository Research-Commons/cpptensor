#include <cmath>
#include <iostream>
#include <chrono>

#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/backend/backend_loader.hpp"

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
#include "cpptensor/ops/math/matmul.hpp"

//#include <gperftools/profiler.h>

using namespace cpptensor;

// Simple helper for timing
double benchmark_matmul(int M, int K, int N, int runs = 10) {
    std::cout << "\n===== Benchmark: Matmul (" << M << "x" << K << " × " << K << "x" << N << ") =====" << std::endl;

    // Create tensors
    Tensor A = Tensor::full({(size_t)M, (size_t)K}, 1.0f, DeviceType::CPU);
    Tensor B = Tensor::full({(size_t)K, (size_t)N}, 1.0f, DeviceType::CPU);

    // Warmup (to avoid cold cache or lazy init effects)
    for (int i = 0; i < 3; ++i) {
        Tensor C = cpptensor::matmul(A, B);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    Tensor C;
    for (int i = 0; i < runs; ++i) {
        C = cpptensor::matmul(A, B);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count() / runs;

    // Compute FLOPs and GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed * 1e9);

    std::cout << "Average time: " << (elapsed * 1000.0) << " ms" << std::endl;
    std::cout << "Performance:  " << gflops << " GFLOPS" << std::endl;

#ifdef USE_OPENBLAS
    std::cout << "Backend: OpenBLAS" << std::endl;
#elif defined(BUILD_AVX512)
    std::cout << "Backend: AVX512" << std::endl;
#elif defined(BUILD_AVX2)
    std::cout << "Backend: AVX2" << std::endl;
#else
    std::cout << "Backend: Scalar/CPU" << std::endl;
#endif

    return gflops;
}

int main() {

    initialize_kernels();

    //-----------------TESTING---------------------

    Tensor A({2,3}, std::vector<float>{1,2,3,4,5,6}, DeviceType::CPU);
    Tensor B({2,3}, std::vector<float>{6,5,4,3,2,1}, DeviceType::CPU);

    // ====== Binary Operations ======
    Tensor C1 = A + B;
    Tensor C2 = A * B;
    Tensor C3 = B - A;
    Tensor C4 = B / A;
    Tensor C5 = cpptensor::pow(A, B);      // A ^ B

    // ====== Unary Operations ======
    Tensor C7 = cpptensor::exp(A);         // e^A
    Tensor C8 = cpptensor::log(A);         // log(A)
    Tensor C9 = cpptensor::sqrt(A);        // sqrt(A)
    Tensor C10 = cpptensor::abs(-A); // | -A |
    Tensor C11 = cpptensor::sigmoid(A);    // 1 / (1 + exp(-A))
    Tensor C12 = cpptensor::relu(A);       // max(0, A)
    Tensor C13 = cpptensor::sin(A);        // sin(A)
    Tensor C14 = cpptensor::cos(A);        // cos(A)
    Tensor C15 = cpptensor::tan(A);        // tan(A)

    // ====== Linear Algebra: Matmul ======
    Tensor M1 = Tensor::full({32,64}, 5.f,  DeviceType::CPU);
    Tensor M2 = Tensor::full({64,32}, 5.f, DeviceType::CPU);

    Tensor M3 = cpptensor::matmul(M1, M2);

    // ====== Print Results ======
    std::cout << "\n===== Binary Ops =====" << std::endl;
    std::cout << "Add (A + B): ";   C1.print();
    std::cout << "Sub (A - B): ";   C2.print();
    std::cout << "Mul (A * B): ";   C3.print();
    std::cout << "Div (A / B): ";   C4.print();
    std::cout << "Pow (A ^ B): ";   C5.print();

    std::cout << "\n===== Unary Ops =====" << std::endl;
    std::cout << "Exp (e^A): ";         C7.print();
    std::cout << "Log (ln(A)): ";       C8.print();
    std::cout << "Sqrt (√A): ";         C9.print();
    std::cout << "Abs (|-A|): ";        C10.print();
    std::cout << "Sigmoid (σ(A)): ";    C11.print();
    std::cout << "ReLU (max(0,A)): ";   C12.print();
    std::cout << "Sin (sin(A)): ";      C13.print();
    std::cout << "Cos (cos(A)): ";      C14.print();
    std::cout << "Tan (tan(A)): ";      C15.print();

    std::cout << "\n===== Linear Algebra =====" << std::endl;
    std::cout << "Matmul (M1 × M2): ";  M3.print();

    std::cout << "=== cpptensor Matmul GFLOPS Benchmark ===\n";

    // Small correctness check
    {
        Tensor A = Tensor::full({32, 64}, 5.f, DeviceType::CPU);
        Tensor B = Tensor::full({64, 32}, 5.f, DeviceType::CPU);
        Tensor C = cpptensor::matmul(A, B);
        std::cout << "Small sanity test: 2x3 × 3x2 result:\n";
        C.print();
    }

    // Run performance tests
    benchmark_matmul(512, 512, 512);
    benchmark_matmul(1024, 1024, 1024);
    benchmark_matmul(2048, 2048, 2048);
    //benchmark_matmul(7700, 7700, 7700);

    //-----------------PROFILING---------------------
    //Run a bunch of tensor computations in a loop

    // ProfilerStart("profile.out");
    //
    // Tensor finalW;
    // for (int i = 0; i < 100000; ++i) {
    //     Tensor A({2,3}, std::vector<float>{1,2,3,4,5,6}, DeviceType::CPU);
    //     Tensor B({2,3}, std::vector<float>{6,5,4,3,2,1}, DeviceType::CPU);
    //
    //     // ====== Binary Operations ======
    //     Tensor C1 = A + B;
    //     Tensor C2 = A * B;
    //     Tensor C3 = B - A;
    //     Tensor C4 = B / A;
    //     Tensor C5 = cpptensor::pow(A, B);      // A ^ B
    //
    //     // ====== Unary Operations ======
    //     Tensor C7 = cpptensor::exp(A);         // e^A
    //     Tensor C8 = cpptensor::log(A);         // log(A)
    //     Tensor C9 = cpptensor::sqrt(A);        // sqrt(A)
    //     Tensor C10 = cpptensor::abs(-A); // | -A |
    //     Tensor C11 = cpptensor::sigmoid(A);    // 1 / (1 + exp(-A))
    //     Tensor C12 = cpptensor::relu(A);       // max(0, A)
    //     Tensor C13 = cpptensor::sin(A);        // sin(A)
    //     Tensor C14 = cpptensor::cos(A);        // cos(A)
    //     Tensor C15 = cpptensor::tan(A);        // tan(A)
    //
    //     // keep the result so compiler doesn’t optimize everything away
    //     finalW = C15;
    // }
    //
    //
    // ProfilerStop();

    return 0;
}

