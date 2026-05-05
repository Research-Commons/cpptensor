#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

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
#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/ops/math/abs.hpp"
#include "cpptensor/ops/math/cos.hpp"
#include "cpptensor/ops/math/log.hpp"
#include "cpptensor/ops/math/exp.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/ops/linearAlgebra/tensordot.hpp"
#include "cpptensor/ops/math/sin.hpp"
#include "cpptensor/ops/math/sqrt.hpp"
#include "cpptensor/ops/math/tan.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/ops/linearAlgebra/svd.hpp"
#include "cpptensor/ops/linearAlgebra/eig.hpp"
#include "cpptensor/ops/reduction/sum.hpp"
#include "cpptensor/ops/reduction/mean.hpp"
#include "cpptensor/ops/reduction/max.hpp"
#include "cpptensor/ops/reduction/min.hpp"
#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/ops/manipulation/stack.hpp"
#include "cpptensor/ops/comparison/eq.hpp"
#include "cpptensor/ops/comparison/ne.hpp"
#include "cpptensor/ops/comparison/gt.hpp"
#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/ops/comparison/ge.hpp"
#include "cpptensor/ops/comparison/le.hpp"



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

double benchmark_matmul_nd(const std::vector<size_t>& Ashape,
                           const std::vector<size_t>& Bshape,
                           int runs = 10) {
    Tensor A = Tensor::full(Ashape, 1.0f, DeviceType::CPU);
    Tensor B = Tensor::full(Bshape, 1.0f, DeviceType::CPU);
    for (int i = 0; i < 3; ++i) (void)cpptensor::matmul(A, B);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor C;
    for (int i = 0; i < runs; ++i) C = cpptensor::matmul(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count() / runs;

    const auto& Csh = C.shape();
    size_t M = Csh[Csh.size()-2], N = Csh[Csh.size()-1];
    size_t K = Ashape[Ashape.size()-1];

    size_t batch_count = 1;
    for (size_t i = 0; i + 2 < Csh.size(); ++i) batch_count *= Csh[i];

    double flops_total = 2.0 * M * N * K * batch_count;
    double gflops = flops_total / (elapsed * 1e9);

    double per_gemm_flops = 2.0 * M * N * K;
    double per_gemm_time_s = elapsed / std::max<size_t>(1, batch_count);
    double per_gemm_gflops = per_gemm_flops / (per_gemm_time_s * 1e9);

    std::cout << "\n===== Benchmark: ND Matmul =====\n";
    std::cout << "A shape: [ "; for (auto v: Ashape) std::cout << v << " "; std::cout << "]\n";
    std::cout << "B shape: [ "; for (auto v: Bshape) std::cout << v << " "; std::cout << "]\n";
    std::cout << "Average time (total): " << elapsed*1e3 << " ms\n";
    std::cout << "Batches: " << batch_count << "\n";
    std::cout << "Total FLOPs: " << flops_total/1e9 << " GFLOPs\n";
    std::cout << "Performance (total): " << gflops << " GFLOPS\n";
    std::cout << "Per-GEMM time: " << per_gemm_time_s*1e6 << " us\n";
    std::cout << "Per-GEMM perf: " << per_gemm_gflops << " GFLOPS\n";

#ifdef USE_OPENBLAS
    std::cout << "Backend: OpenBLAS\n";
#elif defined(BUILD_AVX512)
    std::cout << "Backend: AVX512\n";
#elif defined(BUILD_AVX2)
    std::cout << "Backend: AVX2\n";
#else
    std::cout << "Backend: Scalar/CPU\n";
#endif
    return gflops;
}

// Benchmark tensordot given shapes and axes; computes GFLOPS using the
// equivalent GEMM: [M,K] x [K,N] -> [M,N], where
//   M = prod(A_rest), K = prod(contracted), N = prod(B_rest)
double benchmark_tensordot(const std::vector<size_t>& Ashape,
                           const std::vector<size_t>& Bshape,
                           const std::vector<int>& axesA,
                           const std::vector<int>& axesB,
                           int runs = 10) {
    std::cout << "\n===== Benchmark: Tensordot =====\n";
    std::cout << "A shape: [ "; for (auto v: Ashape) std::cout << v << " "; std::cout << "]\n";
    std::cout << "B shape: [ "; for (auto v: Bshape) std::cout << v << " "; std::cout << "]\n";
    std::cout << "axesA: [ "; for (auto v: axesA) std::cout << v << " "; std::cout << "]\n";
    std::cout << "axesB: [ "; for (auto v: axesB) std::cout << v << " "; std::cout << "]\n";

    Tensor A = Tensor::full(Ashape, 1.0f, DeviceType::CPU);
    Tensor B = Tensor::full(Bshape, 1.0f, DeviceType::CPU);

    // Warmup
    for (int i = 0; i < 3; ++i) (void)cpptensor::tensordot(A, B, axesA, axesB);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor C;
    for (int i = 0; i < runs; ++i) C = cpptensor::tensordot(A, B, axesA, axesB);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count() / runs;

    // Compute M, K, N analogous to implementation
    auto norm_axes = [&](const std::vector<int>& ax, size_t rank){
        std::vector<int> out = ax; for (auto& a: out){ if (a < 0) a += (int)rank; }
        return out;
    };

    auto comp_axes = [&](size_t rank, const std::vector<int>& ax){
        std::vector<int> all(rank); for (size_t i=0;i<rank;++i) all[i]=(int)i;
        std::vector<int> sorted = ax; std::sort(sorted.begin(), sorted.end());
        std::vector<int> rest; rest.reserve(rank - sorted.size());
        size_t j=0; for (size_t i=0;i<rank;++i){ if (j<sorted.size() && (int)i==sorted[j]) {++j;} else rest.push_back((int)i);} return rest;
    };

    auto prodv = [](const std::vector<size_t>& v){ size_t p=1; for (auto x: v) p*=x; return p; };

    auto axesA_n = norm_axes(axesA, Ashape.size());
    auto axesB_n = norm_axes(axesB, Bshape.size());
    auto Arest = comp_axes(Ashape.size(), axesA_n);
    auto Brest = comp_axes(Bshape.size(), axesB_n);

    std::vector<size_t> Arest_sh; Arest_sh.reserve(Arest.size());
    for (auto i: Arest) Arest_sh.push_back(Ashape[(size_t)i]);
    std::vector<size_t> Ak_sh; Ak_sh.reserve(axesA_n.size());
    for (auto i: axesA_n) Ak_sh.push_back(Ashape[(size_t)i]);

    std::vector<size_t> Bk_sh; Bk_sh.reserve(axesB_n.size());
    for (auto i: axesB_n) Bk_sh.push_back(Bshape[(size_t)i]);
    std::vector<size_t> Brest_sh; Brest_sh.reserve(Brest.size());
    for (auto i: Brest) Brest_sh.push_back(Bshape[(size_t)i]);

    size_t M = prodv(Arest_sh);
    size_t K = prodv(Ak_sh); // equals prod(Bk_sh)
    size_t N = prodv(Brest_sh);

    double flops_total = 2.0 * (double)M * (double)K * (double)N; // total operations
    double gflops = flops_total / (elapsed * 1e9);

    std::cout << "Average time: " << (elapsed * 1000.0) << " ms\n";
    std::cout << "Total FLOPs: " << (flops_total/1e9) << " GFLOPs\n";
    std::cout << "Performance:  " << gflops << " GFLOPS\n";

#ifdef USE_OPENBLAS
    std::cout << "Backend (matmul core): OpenBLAS\n";
#elif defined(BUILD_AVX512)
    std::cout << "Backend (matmul core): AVX512\n";
#elif defined(BUILD_AVX2)
    std::cout << "Backend (matmul core): AVX2\n";
#else
    std::cout << "Backend (matmul core): Scalar/CPU\n";
#endif
    return gflops;
}

// Benchmark SVD given matrix shape and options
// SVD involves O(M*N^2) for M>=N or O(M^2*N) for M<N operations
double benchmark_svd(size_t M, size_t N, bool full_matrices = true, int runs = 10) {
#ifdef USE_OPENBLAS
    std::cout << "\n===== Benchmark: SVD =====" << std::endl;
    std::cout << "Matrix shape: [" << M << " × " << N << "]" << std::endl;
    std::cout << "Full matrices: " << (full_matrices ? "yes" : "no") << std::endl;

    Tensor A = Tensor::randn({M, N}, DeviceType::CPU);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        (void)cpptensor::svd(A, full_matrices, true);
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    cpptensor::SVDResult result;
    for (int i = 0; i < runs; ++i) {
        result = cpptensor::svd(A, full_matrices, true);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count() / runs;

    // SVD complexity: roughly O(M*N*min(M,N)) for the dominant term
    // More precisely: ~2*M*N*min(M,N) + 2*min(M,N)^3 FLOPs
    size_t min_dim = std::min(M, N);
    size_t max_dim = std::max(M, N);
    double flops = 2.0 * max_dim * min_dim * min_dim + 2.0 * min_dim * min_dim * min_dim;
    double gflops = flops / (elapsed * 1e9);

    std::cout << "Average time: " << (elapsed * 1000.0) << " ms" << std::endl;
    std::cout << "Estimated FLOPs: " << (flops / 1e9) << " GFLOPs" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "U shape: [" << result.U.shape()[0] << " × " << result.U.shape()[1] << "]" << std::endl;
    std::cout << "S shape: [" << result.S.shape()[0] << "]" << std::endl;
    std::cout << "Vt shape: [" << result.Vt.shape()[0] << " × " << result.Vt.shape()[1] << "]" << std::endl;
    std::cout << "Backend: OpenBLAS (LAPACK sgesvd)" << std::endl;

    return gflops;
#else
    std::cout << "\n===== SVD Benchmark not available (requires OpenBLAS) =====" << std::endl;
    return 0.0;
#endif
}

// Benchmark symmetric eigenvalue decomposition
// Complexity: O(N^3) for symmetric eigenvalue decomposition
double benchmark_eig_symmetric(size_t N, bool compute_eigenvectors = true, int runs = 10) {
#ifdef USE_OPENBLAS
    std::cout << "\n===== Benchmark: EIG Symmetric =====" << std::endl;
    std::cout << "Matrix shape: [" << N << " × " << N << "]" << std::endl;
    std::cout << "Compute eigenvectors: " << (compute_eigenvectors ? "yes" : "no") << std::endl;

    // Create a random symmetric matrix (A + A^T) / 2
    Tensor A = Tensor::randn({N, N}, DeviceType::CPU);
    auto a_data = A.data();
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            float avg = (a_data[i * N + j] + a_data[j * N + i]) / 2.0f;
            a_data[i * N + j] = avg;
            a_data[j * N + i] = avg;
        }
    }
    A = Tensor({N, N}, a_data, DeviceType::CPU);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        (void)cpptensor::eig_symmetric(A, compute_eigenvectors);
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    cpptensor::EigResult result;
    for (int i = 0; i < runs; ++i) {
        result = cpptensor::eig_symmetric(A, compute_eigenvectors);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count() / runs;

    // Symmetric eigenvalue decomposition complexity: ~(4/3)*N^3 FLOPs
    double flops = (4.0 / 3.0) * N * N * N;
    double gflops = flops / (elapsed * 1e9);

    std::cout << "Average time: " << (elapsed * 1000.0) << " ms" << std::endl;
    std::cout << "Estimated FLOPs: " << (flops / 1e9) << " GFLOPs" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Eigenvalues shape: [" << result.eigenvalues.shape()[0] << "]" << std::endl;
    if (compute_eigenvectors) {
        std::cout << "Eigenvectors shape: [" << result.eigenvectors.shape()[0] << " × "
                  << result.eigenvectors.shape()[1] << "]" << std::endl;
    }
    std::cout << "Backend: OpenBLAS (LAPACK ssyevd)" << std::endl;

    return gflops;
#else
    std::cout << "\n===== EIG Symmetric Benchmark not available (requires OpenBLAS) =====" << std::endl;
    return 0.0;
#endif
}

// Benchmark general eigenvalue decomposition
// Complexity: O(N^3) for general eigenvalue decomposition
double benchmark_eig(size_t N, bool compute_eigenvectors = true, int runs = 10) {
#ifdef USE_OPENBLAS
    std::cout << "\n===== Benchmark: EIG General =====" << std::endl;
    std::cout << "Matrix shape: [" << N << " × " << N << "]" << std::endl;
    std::cout << "Compute eigenvectors: " << (compute_eigenvectors ? "yes" : "no") << std::endl;

    Tensor A = Tensor::randn({N, N}, DeviceType::CPU);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        (void)cpptensor::eig(A, compute_eigenvectors);
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    cpptensor::EigResult result;
    for (int i = 0; i < runs; ++i) {
        result = cpptensor::eig(A, compute_eigenvectors);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count() / runs;

    // General eigenvalue decomposition complexity: ~10*N^3 FLOPs (QR algorithm)
    double flops = 10.0 * N * N * N;
    double gflops = flops / (elapsed * 1e9);

    std::cout << "Average time: " << (elapsed * 1000.0) << " ms" << std::endl;
    std::cout << "Estimated FLOPs: " << (flops / 1e9) << " GFLOPs" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Eigenvalues shape: [" << result.eigenvalues.shape()[0] << "]" << std::endl;
    std::cout << "Eigenvalues (imag) shape: [" << result.eigenvalues_imag.shape()[0] << "]" << std::endl;
    if (compute_eigenvectors) {
        std::cout << "Eigenvectors shape: [" << result.eigenvectors.shape()[0] << " × "
                  << result.eigenvectors.shape()[1] << "]" << std::endl;
    }
    std::cout << "Backend: OpenBLAS (LAPACK sgeev)" << std::endl;

    return gflops;
#else
    std::cout << "\n===== EIG General Benchmark not available (requires OpenBLAS) =====" << std::endl;
    return 0.0;
#endif
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
    //2D
    Tensor M1 = Tensor::full({32,64}, 5.f,  DeviceType::CPU);
    Tensor M2 = Tensor::full({64,32}, 5.f, DeviceType::CPU);
    Tensor M3 = cpptensor::matmul(M1, M2);

    //3D
    Tensor M4({2,2,3}, {
        // batch0
        1,2,3,
        4,5,6,
        // batch1
        6,5,4,
        3,2,1
    });
    Tensor M5({2,3,2}, {
        // batch0
        1,2,
        3,4,
        5,6,
        // batch1
        1,0,
        0,1,
        1,1
    });
    Tensor M6 = matmul(M4, M5);

    //4D
    // A: [2,1,2,3]
    Tensor M7({2,1,2,3}, {
        // batch 0
        1,2,3,
        4,5,6,
        // batch 1
        7,8,9,
        1,2,3
    });

    // B: [2,1,3,2]
    Tensor M8({2,1,3,2}, {
        // batch 0
        1,2,
        3,4,
        5,6,
        // batch 1
        2,1,
        0,1,
        1,0
    });
    Tensor M9 = matmul(M7, M8);

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
    std::cout << "Matmul 3D (M4 × M5): ";  M6.print();
    std::cout << "Matmul 4D (M7 × M8): ";  M9.print();

    // ====== Dot product examples ======
    std::cout << "\n===== Dot Product =====" << std::endl;
    {
        // Simple dot product: [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        Tensor v1({3}, {1.0f, 2.0f, 3.0f});
        Tensor v2({3}, {4.0f, 5.0f, 6.0f});
        Tensor result = cpptensor::dot(v1, v2);
        std::cout << "Dot([1,2,3], [4,5,6]) = " << result.data()[0] << " (expected 32)" << std::endl;
    }
    {
        // Orthogonal vectors: [1,0,0] · [0,1,0] = 0
        Tensor v1({3}, {1.0f, 0.0f, 0.0f});
        Tensor v2({3}, {0.0f, 1.0f, 0.0f});
        Tensor result = cpptensor::dot(v1, v2);
        std::cout << "Dot([1,0,0], [0,1,0]) = " << result.data()[0] << " (expected 0)" << std::endl;
    }
    {
        // Larger vector
        std::vector<float> data1(100);
        std::vector<float> data2(100);
        for (int i = 0; i < 100; ++i) {
            data1[i] = static_cast<float>(i);
            data2[i] = 1.0f;
        }
        Tensor v1({100}, data1);
        Tensor v2({100}, data2);
        Tensor result = cpptensor::dot(v1, v2);
        // Sum of 0+1+2+...+99 = 99*100/2 = 4950
        std::cout << "Dot([0..99], [1..1]) = " << result.data()[0] << " (expected 4950)" << std::endl;
    }

    // ====== Tensordot correctness checks ======
    {
        // Vector dot: tensordot with axes=1 should equal standard dot
        Tensor v1({3}, {1,2,3});
        Tensor v2({3}, {4,5,6});
        Tensor s = cpptensor::tensordot(v1, v2, 1);
        std::cout << "\nTensordot vector dot (expected 32): " << s.data()[0] << "\n";
    }
    {
        // Contract two axes: A[2,3,4], B[3,4,5] -> axesA={1,2}, axesB={0,1} => [2,5]
        Tensor A = Tensor::full({2,3,4}, 1.0f);
        Tensor B = Tensor::full({3,4,5}, 1.0f);
        Tensor O = cpptensor::tensordot(A, B, std::vector<int>{1,2}, std::vector<int>{0,1});
        // With all ones, each output entry equals product of contracted dims = 3*4 = 12
        std::cout << "Tensordot [2,3,4] x [3,4,5] over (1,2),(0,1) -> shape [2,5], expect all 12s\n";
        O.print();
    }

    std::cout << "\n\n\n=== cpptensor Matmul GFLOPS Benchmark ===\n";

    // Small correctness check
    {
        Tensor A = Tensor::full({32, 64}, 5.f, DeviceType::CPU);
        Tensor B = Tensor::full({64, 32}, 5.f, DeviceType::CPU);
        Tensor C = cpptensor::matmul(A, B);
        std::cout << "Small sanity test: 2x3 × 3x2 result:\n";
        C.print();
    }

    // ====== SVD examples ======
#ifdef USE_OPENBLAS
    std::cout << "\n===== Singular Value Decomposition (SVD) =====" << std::endl;
    {
        // Simple 3x2 matrix
        Tensor A({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        std::cout << "Input matrix A [3×2]:\n";
        A.print();

        // Full SVD
        auto [U, S, Vt] = cpptensor::svd(A, true, true);
        std::cout << "Full SVD:\n";
        std::cout << "  U [3×3]:  "; U.print();
        std::cout << "  S [2]:    "; S.print();
        std::cout << "  Vt [2×2]: "; Vt.print();

        // Verify reconstruction: A ≈ U @ diag(S) @ Vt
        // For simplicity, just show singular values are positive and sorted
        std::cout << "  Singular values (should be positive, descending): ";
        for (size_t i = 0; i < S.shape()[0]; ++i) {
            std::cout << S.data()[i] << " ";
        }
        std::cout << "\n";
    }
    {
        // Economy SVD (more memory efficient for tall matrices)
        Tensor A({4, 3}, {1,2,3, 4,5,6, 7,8,9, 10,11,12});
        std::cout << "\nEconomy SVD of [4×3] matrix:\n";
        auto result = cpptensor::svd(A, false, true);
        std::cout << "  U [4×3]:  shape = [" << result.U.shape()[0] << "×" << result.U.shape()[1] << "]\n";
        std::cout << "  S [3]:    "; result.S.print();
        std::cout << "  Vt [3×3]: shape = [" << result.Vt.shape()[0] << "×" << result.Vt.shape()[1] << "]\n";
    }
    {
        // Only compute singular values (fastest)
        Tensor A({5, 5}, {1,0,0,0,2,
                          0,0,3,0,0,
                          0,0,0,0,0,
                          0,4,0,0,0,
                          5,0,0,0,0});
        auto [_, S, __] = cpptensor::svd(A, false, false);
        std::cout << "\nSingular values only for [5×5] matrix:\n  S: ";
        S.print();
    }
#else
    std::cout << "\n===== SVD not available (requires OpenBLAS) =====\n";
#endif

    // ====== Eigenvalue Decomposition examples ======
#ifdef USE_OPENBLAS
    std::cout << "\n===== Eigenvalue Decomposition (EIG) =====" << std::endl;
    {
        // Symmetric matrix example
        std::cout << "\n-- Symmetric Matrix --" << std::endl;
        Tensor A({3, 3}, {4.0f, 1.0f, 2.0f,
                          1.0f, 3.0f, 1.0f,
                          2.0f, 1.0f, 4.0f});
        std::cout << "Input symmetric matrix A [3×3]:\n";
        A.print();

        auto [vals, vals_im, vecs] = cpptensor::eig_symmetric(A);
        std::cout << "Eigenvalues: ";
        vals.print();
        std::cout << "Eigenvectors (columns):\n";
        vecs.print();
        std::cout << "(All eigenvalues are real for symmetric matrices)\n";
    }
    {
        // General matrix with real eigenvalues
        std::cout << "\n-- General Matrix (real eigenvalues) --" << std::endl;
        Tensor A({3, 3}, {3.0f, 1.0f, 0.0f,
                          0.0f, 2.0f, 0.0f,
                          0.0f, 0.0f, 1.0f});
        std::cout << "Diagonal matrix (general eig):\n";
        A.print();

        auto [vals_re, vals_im, vecs] = cpptensor::eig(A);
        std::cout << "Eigenvalues (real): ";
        vals_re.print();
        std::cout << "Eigenvalues (imag): ";
        vals_im.print();
    }
    {
        // General matrix with complex eigenvalues
        std::cout << "\n-- General Matrix (complex eigenvalues) --" << std::endl;
        Tensor A({2, 2}, {0.0f, 1.0f,
                          -1.0f, 0.0f});
        std::cout << "Rotation matrix [2×2]:\n";
        A.print();

        auto [vals_re, vals_im, vecs] = cpptensor::eig(A);
        std::cout << "Eigenvalues (real): ";
        vals_re.print();
        std::cout << "Eigenvalues (imag): ";
        vals_im.print();
        std::cout << "(Complex eigenvalues: ±i)\n";
    }
#else
    std::cout << "\n===== EIG not available (requires OpenBLAS) =====\n";
#endif


    // ====== Tensor Manipulation Examples ======
    std::cout << "\n\n===== TENSOR MANIPULATION EXAMPLES =====" << std::endl;

    // 1. View Operations (zero-copy, shares data)
    std::cout << "\n--- 1. View Operations ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        std::cout << "Original A [2×3]: "; A.print();

        Tensor B = A.view({3, 2});  // Reshape without copying
        std::cout << "View B [3×2]: "; B.print();

        // Modifying B modifies A (shared data)
        B.data()[0] = 99.0f;
        std::cout << "After modifying B[0]: A[0]=" << A.data()[0] << " (data shared!)" << std::endl;
    }

    // 2. Reshape Operations (smart: view if contiguous, else copy)
    std::cout << "\n--- 2. Reshape Operations ---" << std::endl;
    {
        Tensor A = Tensor::full({2, 3, 4}, 1.0f);  // [2×3×4] = 24 elements
        std::cout << "Original A shape: [";
        for (auto s : A.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        Tensor B = A.reshape({6, 4});
        std::cout << "Reshaped B shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        Tensor C = A.reshape({24});
        std::cout << "Flattened C shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // 3. Flatten Operations
    std::cout << "\n--- 3. Flatten Operations ---" << std::endl;
    {
        Tensor A = Tensor::full({2, 3, 4}, 1.0f);
        std::cout << "Original A shape: [2, 3, 4]" << std::endl;

        Tensor B = A.flatten();  // Flatten all dimensions
        std::cout << "Fully flattened shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        Tensor C = A.flatten(1, 2);  // Flatten dims 1-2 only
        std::cout << "Partially flattened (dims 1-2) shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // 4. Squeeze Operations (remove size-1 dimensions)
    std::cout << "\n--- 4. Squeeze Operations ---" << std::endl;
    {
        Tensor A = Tensor::full({2, 1, 3, 1, 4}, 1.0f);
        std::cout << "Original A shape: [2, 1, 3, 1, 4]" << std::endl;

        Tensor B = A.squeeze();  // Remove all size-1 dims
        std::cout << "Squeezed (all) shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        Tensor C = A.squeeze(1);  // Remove specific dim
        std::cout << "Squeezed (dim 1) shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // 5. Unsqueeze Operations (add size-1 dimension)
    std::cout << "\n--- 5. Unsqueeze Operations ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        std::cout << "Original A shape: [2, 3]" << std::endl;

        Tensor B = A.unsqueeze(0);  // Add dim at position 0
        std::cout << "Unsqueezed (dim 0) shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        Tensor C = A.unsqueeze(2);  // Add dim at position 2
        std::cout << "Unsqueezed (dim 2) shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // 6. Permute Operations (arbitrary dimension reordering)
    std::cout << "\n--- 6. Permute Operations ---" << std::endl;
    {
        Tensor A = Tensor::full({2, 3, 4}, 1.0f);
        std::cout << "Original A shape: [2, 3, 4]" << std::endl;

        Tensor B = A.permute({2, 0, 1});  // Reorder to [4, 2, 3]
        std::cout << "Permuted (2,0,1) shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "], contiguous=" << (B.is_contiguous() ? "yes" : "no") << std::endl;

        // Permute changes memory layout, making it non-contiguous
        Tensor C = B.contiguous();  // Make contiguous again
        std::cout << "Made contiguous C: contiguous=" << (C.is_contiguous() ? "yes" : "no") << std::endl;
    }

    // 7. Transpose Operations
    std::cout << "\n--- 7. Transpose Operations ---" << std::endl;
    {
        Tensor A({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        std::cout << "Original A [3×4]:" << std::endl;
        A.print();

        Tensor B = A.transpose();  // Swap last two dims
        std::cout << "Transposed B [4×3]:" << std::endl;
        B.print();

        // 3D tensor transpose
        Tensor C = Tensor::full({2, 3, 4}, 1.0f);
        Tensor D = C.transpose(0, 2);  // Swap dims 0 and 2
        std::cout << "3D transpose (0,2): [2,3,4] -> [";
        for (auto s : D.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // 8. Contiguous Operations
    std::cout << "\n--- 8. Contiguous Operations ---" << std::endl;
    {
        Tensor A = Tensor::full({2, 3, 4}, 1.0f);
        std::cout << "Original A: contiguous=" << (A.is_contiguous() ? "yes" : "no") << std::endl;

        Tensor B = A.permute({2, 0, 1});
        std::cout << "After permute: contiguous=" << (B.is_contiguous() ? "yes" : "no") << std::endl;

        Tensor C = B.contiguous();
        std::cout << "After contiguous(): contiguous=" << (C.is_contiguous() ? "yes" : "no") << std::endl;
    }

    // 9. Clone Operations (deep copy)
    std::cout << "\n--- 9. Clone Operations ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B = A.clone();  // Independent copy

        B.data()[0] = 99.0f;
        std::cout << "After modifying clone: A[0]=" << A.data()[0]
                  << ", B[0]=" << B.data()[0] << " (independent!)" << std::endl;
    }

    // 10. Complex Manipulation Sequence
    std::cout << "\n--- 10. Complex Example: Image Batch Processing ---" << std::endl;
    {
        // Image batch: [batch=4, channels=3, height=64, width=64]
        Tensor images = Tensor::full({4, 3, 64, 64}, 1.0f);
        std::cout << "Image batch shape: [4, 3, 64, 64]" << std::endl;

        // Permute to [batch, height, width, channels] (NHWC format)
        Tensor nhwc = images.permute({0, 2, 3, 1});
        std::cout << "NHWC format shape: [";
        for (auto s : nhwc.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        // Flatten spatial dimensions: [batch, height*width, channels]
        Tensor flattened = nhwc.reshape({4, 64*64, 3});
        std::cout << "Flattened spatial shape: [";
        for (auto s : flattened.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // ====== Reduction Operations Examples ======
    std::cout << "\n\n===== REDUCTION OPERATIONS EXAMPLES =====" << std::endl;

    // 1. Sum Operations
    std::cout << "\n--- 1. Sum Operations ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        std::cout << "Original tensor A [2×3]:" << std::endl;
        A.print();

        // Sum all elements
        Tensor sum_all = A.sum();
        std::cout << "Sum of all elements: " << sum_all.data()[0] << " (expected: 21)" << std::endl;

        // Sum along dimension 0 (columns)
        Tensor sum_dim0 = A.sum(0);
        std::cout << "Sum along dim 0 (columns): ";
        sum_dim0.print();
        std::cout << "  Expected: [5, 7, 9]" << std::endl;

        // Sum along dimension 1 (rows)
        Tensor sum_dim1 = A.sum(1);
        std::cout << "Sum along dim 1 (rows): ";
        sum_dim1.print();
        std::cout << "  Expected: [6, 15]" << std::endl;

        // Sum with keepdim
        Tensor sum_keepdim = A.sum(0, true);
        std::cout << "Sum dim 0 with keepdim, shape: [";
        for (auto s : sum_keepdim.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // 2. Mean Operations
    std::cout << "\n--- 2. Mean Operations ---" << std::endl;
    {
        Tensor A({2, 3}, {2, 4, 6, 8, 10, 12});
        std::cout << "Original tensor A [2×3]:" << std::endl;
        A.print();

        // Mean of all elements
        Tensor mean_all = A.mean();
        std::cout << "Mean of all elements: " << mean_all.data()[0] << " (expected: 7)" << std::endl;

        // Mean along dimension 0
        Tensor mean_dim0 = A.mean(0);
        std::cout << "Mean along dim 0: ";
        mean_dim0.print();
        std::cout << "  Expected: [5, 7, 9]" << std::endl;

        // Mean along dimension 1
        Tensor mean_dim1 = A.mean(1);
        std::cout << "Mean along dim 1: ";
        mean_dim1.print();
        std::cout << "  Expected: [4, 10]" << std::endl;
    }

    // 3. Max Operations
    std::cout << "\n--- 3. Max Operations ---" << std::endl;
    {
        Tensor A({2, 3}, {3, 1, 4, 1, 5, 9});
        std::cout << "Original tensor A [2×3]:" << std::endl;
        A.print();

        // Max of all elements
        Tensor max_all = A.max();
        std::cout << "Max of all elements: " << max_all.data()[0] << " (expected: 9)" << std::endl;

        // Max along dimension 0
        Tensor max_dim0 = A.max(0);
        std::cout << "Max along dim 0: ";
        max_dim0.print();
        std::cout << "  Expected: [3, 5, 9]" << std::endl;

        // Max along dimension 1
        Tensor max_dim1 = A.max(1);
        std::cout << "Max along dim 1: ";
        max_dim1.print();
        std::cout << "  Expected: [4, 9]" << std::endl;

        // Max with keepdim
        Tensor max_keepdim = A.max(1, true);
        std::cout << "Max dim 1 with keepdim, shape: [";
        for (auto s : max_keepdim.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
    }

    // 4. Min Operations
    std::cout << "\n--- 4. Min Operations ---" << std::endl;
    {
        Tensor A({2, 3}, {3, 1, 4, 1, 5, 9});
        std::cout << "Original tensor A [2×3]:" << std::endl;
        A.print();

        // Min of all elements
        Tensor min_all = A.min();
        std::cout << "Min of all elements: " << min_all.data()[0] << " (expected: 1)" << std::endl;

        // Min along dimension 0
        Tensor min_dim0 = A.min(0);
        std::cout << "Min along dim 0: ";
        min_dim0.print();
        std::cout << "  Expected: [1, 1, 4]" << std::endl;

        // Min along dimension 1
        Tensor min_dim1 = A.min(1);
        std::cout << "Min along dim 1: ";
        min_dim1.print();
        std::cout << "  Expected: [1, 1]" << std::endl;
    }

    // 5. 3D Tensor Reductions
    std::cout << "\n--- 5. 3D Tensor Reductions ---" << std::endl;
    {
        Tensor A = Tensor::full({2, 3, 4}, 1.0f);
        // Add some variation
        auto& data = A.data();
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i);
        }

        std::cout << "3D tensor A [2×3×4], values 0-23" << std::endl;

        // Sum along different dimensions
        Tensor sum_d0 = A.sum(0);
        std::cout << "Sum along dim 0, result shape: [";
        for (auto s : sum_d0.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 4])" << std::endl;

        Tensor sum_d1 = A.sum(1);
        std::cout << "Sum along dim 1, result shape: [";
        for (auto s : sum_d1.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 4])" << std::endl;

        Tensor sum_d2 = A.sum(2);
        std::cout << "Sum along dim 2, result shape: [";
        for (auto s : sum_d2.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3])" << std::endl;

        // All reductions
        float sum_total = A.sum().data()[0];
        float mean_total = A.mean().data()[0];
        float max_total = A.max().data()[0];
        float min_total = A.min().data()[0];

        std::cout << "All reductions on 3D tensor:" << std::endl;
        std::cout << "  Sum:  " << sum_total << " (expected: 276 = 0+1+...+23)" << std::endl;
        std::cout << "  Mean: " << mean_total << " (expected: 11.5)" << std::endl;
        std::cout << "  Max:  " << max_total << " (expected: 23)" << std::endl;
        std::cout << "  Min:  " << min_total << " (expected: 0)" << std::endl;
    }

    // 6. Real-world Example: Batch Statistics
    std::cout << "\n--- 6. Real-world Example: Batch Statistics ---" << std::endl;
    {
        // Simulate batch of images: [batch=4, channels=3, height=8, width=8]
        Tensor batch = Tensor::randn({4, 3, 8, 8});

        // Compute statistics across batch
        Tensor batch_mean = batch.mean(0);  // Mean across batch dimension
        std::cout << "Batch mean shape: [";
        for (auto s : batch_mean.shape()) std::cout << s << " ";
        std::cout << "] (per-channel mean)" << std::endl;

        // Global statistics
        float global_mean = batch.mean().data()[0];
        float global_max = batch.max().data()[0];
        float global_min = batch.min().data()[0];

        std::cout << "Global batch statistics:" << std::endl;
        std::cout << "  Mean: " << global_mean << std::endl;
        std::cout << "  Max:  " << global_max << std::endl;
        std::cout << "  Min:  " << global_min << std::endl;
        std::cout << "  Range: [" << global_min << ", " << global_max << "]" << std::endl;
    }

    // 7. Chaining Reductions
    std::cout << "\n--- 7. Chaining Multiple Reductions ---" << std::endl;
    {
        Tensor A = Tensor::randn({4, 5, 6});

        // Multi-step reduction
        Tensor step1 = A.sum(2);     // Sum over last dim: [4, 5, 6] -> [4, 5]
        Tensor step2 = step1.mean(1); // Mean over dim 1:    [4, 5] -> [4]
        Tensor step3 = step2.max();   // Max of all:         [4] -> scalar

        std::cout << "Chained reductions A.sum(2).mean(1).max():" << std::endl;
        std::cout << "  After sum(2):  shape [";
        for (auto s : step1.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
        std::cout << "  After mean(1): shape [";
        for (auto s : step2.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
        std::cout << "  After max():   scalar value = " << step3.data()[0] << std::endl;
    }

    // 8. Comparison of All Reduction Operations
    std::cout << "\n--- 8. Side-by-side Comparison ---" << std::endl;
    {
        Tensor A({3, 4}, {
            1.5f,  2.0f,  -1.0f,  3.5f,
            0.5f,  4.0f,   2.5f, -0.5f,
            -2.0f, 1.0f,   3.0f,  2.0f
        });
        std::cout << "Test tensor A [3×4]:" << std::endl;
        A.print();

        std::cout << "\nAll reductions along dim=1 (rows):" << std::endl;
        Tensor sum = A.sum(1);
        Tensor mean = A.mean(1);
        Tensor max = A.max(1);
        Tensor min = A.min(1);

        std::cout << "  Sum:  "; sum.print();
        std::cout << "  Mean: "; mean.print();
        std::cout << "  Max:  "; max.print();
        std::cout << "  Min:  "; min.print();
    }

    // NEW: Test overloaded reduction methods
    std::cout << "\n=== TESTING NEW OVERLOADED REDUCTION METHODS ===" << std::endl;
    {
        Tensor A({2, 3, 4}, {
            1.0f,  2.0f,  3.0f,  4.0f,   // [0, :, :]
            5.0f,  6.0f,  7.0f,  8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,

            13.0f, 14.0f, 15.0f, 16.0f,  // [1, :, :]
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f
        });

        std::cout << "\n--- Testing Global Reductions (no dim parameter) ---" << std::endl;
        std::cout << "Tensor A: shape [2, 3, 4], values 1-24" << std::endl;

        // Test global reductions with new overload
        auto sum_global = A.sum();
        std::cout << "A.sum() [global]:     " << sum_global.data()[0] << " (expected: 300)" << std::endl;

        auto mean_global = A.mean();
        std::cout << "A.mean() [global]:    " << mean_global.data()[0] << " (expected: 12.5)" << std::endl;

        auto max_global = A.max();
        std::cout << "A.max() [global]:     " << max_global.data()[0] << " (expected: 24)" << std::endl;

        auto min_global = A.min();
        std::cout << "A.min() [global]:     " << min_global.data()[0] << " (expected: 1)" << std::endl;

        std::cout << "\n--- Testing Dimensional Reductions (with dim parameter) ---" << std::endl;

        // Test dimensional reductions
        auto sum_dim0 = A.sum(0);
        std::cout << "A.sum(0) shape: [";
        for (auto s : sum_dim0.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 4])" << std::endl;

        auto sum_dim2 = A.sum(2);
        std::cout << "A.sum(2) shape: [";
        for (auto s : sum_dim2.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3])" << std::endl;

        // Test negative indexing
        auto sum_neg1 = A.sum(-1);
        std::cout << "A.sum(-1) shape: [";
        for (auto s : sum_neg1.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3], same as sum(2))" << std::endl;

        auto sum_neg2 = A.sum(-2);
        std::cout << "A.sum(-2) shape: [";
        for (auto s : sum_neg2.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 4], same as sum(1))" << std::endl;

        // Verify negative indexing works correctly
        bool neg_indexing_works = (sum_neg1.shape() == sum_dim2.shape());
        std::cout << "\n✓ Negative indexing verification: "
                  << (neg_indexing_works ? "PASS" : "FAIL") << std::endl;

        // Test keepdim with global reduction
        auto sum_keepdim = A.sum(true);
        std::cout << "\nA.sum(true) [global with keepdim] shape: [";
        for (auto s : sum_keepdim.shape()) std::cout << s << " ";
        std::cout << "] (expected: [1, 1, 1])" << std::endl;

        // Test keepdim with dimensional reduction
        auto sum_dim1_keepdim = A.sum(1, true);
        std::cout << "A.sum(1, true) [dim with keepdim] shape: [";
        for (auto s : sum_dim1_keepdim.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 1, 4])" << std::endl;

        std::cout << "\n✓ All overload tests completed successfully!" << std::endl;
    }

    // EDGE CASE TESTS
    std::cout << "\n=== EDGE CASE TESTS FOR REDUCTION OPERATIONS ===" << std::endl;

    // Test 1: Single element tensor
    std::cout << "\n--- Test 1: Single Element Tensor ---" << std::endl;
    {
        Tensor single({1}, std::vector<float>{42.0f});
        std::cout << "Single element tensor: [42.0]" << std::endl;
        std::cout << "sum():  " << single.sum().data()[0] << " (expected: 42)" << std::endl;
        std::cout << "mean(): " << single.mean().data()[0] << " (expected: 42)" << std::endl;
        std::cout << "max():  " << single.max().data()[0] << " (expected: 42)" << std::endl;
        std::cout << "min():  " << single.min().data()[0] << " (expected: 42)" << std::endl;
    }

    // Test 2: All zeros
    std::cout << "\n--- Test 2: All Zeros ---" << std::endl;
    {
        Tensor zeros({2, 3}, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        std::cout << "Tensor of all zeros [2x3]" << std::endl;
        std::cout << "sum():  " << zeros.sum().data()[0] << " (expected: 0)" << std::endl;
        std::cout << "mean(): " << zeros.mean().data()[0] << " (expected: 0)" << std::endl;
        std::cout << "max():  " << zeros.max().data()[0] << " (expected: 0)" << std::endl;
        std::cout << "min():  " << zeros.min().data()[0] << " (expected: 0)" << std::endl;
    }

    // Test 3: All same values
    std::cout << "\n--- Test 3: All Same Values ---" << std::endl;
    {
        Tensor same({2, 4}, std::vector<float>{5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f, 5.5f});
        std::cout << "Tensor of all 5.5 [2x4]" << std::endl;
        std::cout << "sum():  " << same.sum().data()[0] << " (expected: 44)" << std::endl;
        std::cout << "mean(): " << same.mean().data()[0] << " (expected: 5.5)" << std::endl;
        std::cout << "max():  " << same.max().data()[0] << " (expected: 5.5)" << std::endl;
        std::cout << "min():  " << same.min().data()[0] << " (expected: 5.5)" << std::endl;
    }

    // Test 4: Negative values
    std::cout << "\n--- Test 4: Negative Values ---" << std::endl;
    {
        Tensor negative({2, 3}, std::vector<float>{-5.0f, -2.0f, -8.0f, -1.0f, -3.0f, -6.0f});
        std::cout << "Tensor with negative values: [-5, -2, -8, -1, -3, -6]" << std::endl;
        std::cout << "sum():  " << negative.sum().data()[0] << " (expected: -25)" << std::endl;
        std::cout << "mean(): " << negative.mean().data()[0] << " (expected: -4.1667)" << std::endl;
        std::cout << "max():  " << negative.max().data()[0] << " (expected: -1)" << std::endl;
        std::cout << "min():  " << negative.min().data()[0] << " (expected: -8)" << std::endl;
    }

    // Test 5: Mixed positive and negative
    std::cout << "\n--- Test 5: Mixed Positive and Negative ---" << std::endl;
    {
        Tensor mixed({2, 3}, std::vector<float>{-3.0f, 2.0f, -1.0f, 5.0f, -4.0f, 1.0f});
        std::cout << "Mixed tensor: [-3, 2, -1, 5, -4, 1]" << std::endl;
        std::cout << "sum():  " << mixed.sum().data()[0] << " (expected: 0)" << std::endl;
        std::cout << "mean(): " << mixed.mean().data()[0] << " (expected: 0)" << std::endl;
        std::cout << "max():  " << mixed.max().data()[0] << " (expected: 5)" << std::endl;
        std::cout << "min():  " << mixed.min().data()[0] << " (expected: -4)" << std::endl;
    }

    // Test 6: Large dimension size (1D tensor)
    std::cout << "\n--- Test 6: 1D Tensor (100 elements) ---" << std::endl;
    {
        std::vector<float> data_1d(100);
        for (int i = 0; i < 100; i++) data_1d[i] = static_cast<float>(i + 1);
        Tensor large_1d({100}, data_1d);

        float expected_sum = 5050.0f; // 1+2+...+100 = 100*101/2
        float expected_mean = 50.5f;

        std::cout << "1D tensor with values 1-100" << std::endl;
        std::cout << "sum():  " << large_1d.sum().data()[0] << " (expected: " << expected_sum << ")" << std::endl;
        std::cout << "mean(): " << large_1d.mean().data()[0] << " (expected: " << expected_mean << ")" << std::endl;
        std::cout << "max():  " << large_1d.max().data()[0] << " (expected: 100)" << std::endl;
        std::cout << "min():  " << large_1d.min().data()[0] << " (expected: 1)" << std::endl;
    }

    // Test 7: Reduction along first/last dimensions of high-rank tensor
    std::cout << "\n--- Test 7: 4D Tensor Reductions ---" << std::endl;
    {
        std::vector<float> data_4d(2*3*4*5);
        for (size_t i = 0; i < data_4d.size(); i++) data_4d[i] = static_cast<float>(i);
        Tensor tensor_4d({2, 3, 4, 5}, data_4d);

        std::cout << "4D tensor shape: [2, 3, 4, 5]" << std::endl;

        auto sum_dim0 = tensor_4d.sum(0);
        std::cout << "sum(0) shape: [";
        for (auto s : sum_dim0.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 4, 5])" << std::endl;

        auto sum_dim3 = tensor_4d.sum(3);
        std::cout << "sum(3) shape: [";
        for (auto s : sum_dim3.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3, 4])" << std::endl;

        auto sum_neg1 = tensor_4d.sum(-1);
        std::cout << "sum(-1) shape: [";
        for (auto s : sum_neg1.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3, 4], same as sum(3))" << std::endl;

        auto sum_neg4 = tensor_4d.sum(-4);
        std::cout << "sum(-4) shape: [";
        for (auto s : sum_neg4.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 4, 5], same as sum(0))" << std::endl;
    }

    // Test 8: Very small floating point values
    std::cout << "\n--- Test 8: Very Small Floating Point Values ---" << std::endl;
    {
        Tensor tiny({3}, std::vector<float>{1e-6f, 2e-6f, 3e-6f});
        std::cout << "Tensor with tiny values: [1e-6, 2e-6, 3e-6]" << std::endl;
        std::cout << "sum():  " << tiny.sum().data()[0] << " (expected: 6e-6)" << std::endl;
        std::cout << "mean(): " << tiny.mean().data()[0] << " (expected: 2e-6)" << std::endl;
    }

    // Test 9: Verify sum reduction with keepdim on each dimension
    std::cout << "\n--- Test 9: keepdim=true for all dimensions ---" << std::endl;
    {
        Tensor A({2, 3, 4}, std::vector<float>(24, 1.0f)); // All ones

        auto sum_d0_keep = A.sum(0, true);
        std::cout << "sum(0, keepdim=true) shape: [";
        for (auto s : sum_d0_keep.shape()) std::cout << s << " ";
        std::cout << "] (expected: [1, 3, 4])" << std::endl;

        auto sum_d1_keep = A.sum(1, true);
        std::cout << "sum(1, keepdim=true) shape: [";
        for (auto s : sum_d1_keep.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 1, 4])" << std::endl;

        auto sum_d2_keep = A.sum(2, true);
        std::cout << "sum(2, keepdim=true) shape: [";
        for (auto s : sum_d2_keep.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3, 1])" << std::endl;
    }

    // Test 10: Verify all operations give consistent results
    std::cout << "\n--- Test 10: Consistency Check Across Operations ---" << std::endl;
    {
        Tensor data({3, 4}, {
            10.0f, 20.0f, 30.0f, 40.0f,
            50.0f, 60.0f, 70.0f, 80.0f,
            90.0f, 100.0f, 110.0f, 120.0f
        });

        std::cout << "Tensor [3x4] with values 10, 20, ..., 120" << std::endl;

        // Global reductions
        auto sum_g = data.sum();
        auto mean_g = data.mean();
        float expected_sum = 780.0f; // 10+20+...+120
        float expected_mean = 65.0f; // 780/12

        std::cout << "Global sum:  " << sum_g.data()[0] << " (expected: " << expected_sum << ")" << std::endl;
        std::cout << "Global mean: " << mean_g.data()[0] << " (expected: " << expected_mean << ")" << std::endl;

        // Verify: sum/count == mean
        float computed_mean = sum_g.data()[0] / 12.0f;
        bool mean_check = std::abs(computed_mean - mean_g.data()[0]) < 1e-5f;
        std::cout << "sum/count == mean: " << (mean_check ? "✓ PASS" : "✗ FAIL") << std::endl;

        // Dimensional sum along dim=1
        auto sum_d1 = data.sum(1);
        std::cout << "sum(1) values: [";
        for (int i = 0; i < 3; i++) std::cout << sum_d1.data()[i] << " ";
        std::cout << "] (expected: [100, 260, 420])" << std::endl;

        // Verify max >= min
        auto max_g = data.max();
        auto min_g = data.min();
        bool max_min_check = max_g.data()[0] >= min_g.data()[0];
        std::cout << "max >= min: " << (max_min_check ? "✓ PASS" : "✗ FAIL");
        std::cout << " (max=" << max_g.data()[0] << ", min=" << min_g.data()[0] << ")" << std::endl;
    }

    std::cout << "\n✓ All edge case tests completed!" << std::endl;

    std::cout << "\n===== END OF REDUCTION EXAMPLES =====" << std::endl;

    // ========== CAT (CONCATENATE) OPERATION EXAMPLES ==========
    std::cout << "\n\n===== CAT (CONCATENATE) OPERATION EXAMPLES =====" << std::endl;

    // Test 1: Basic concatenation along dim 0
    std::cout << "\n--- 1. Concatenate 2D tensors along dim 0 (rows) ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, {7, 8, 9, 10, 11, 12});

        std::cout << "Tensor A [2x3]: [1, 2, 3, 4, 5, 6]" << std::endl;
        std::cout << "Tensor B [2x3]: [7, 8, 9, 10, 11, 12]" << std::endl;

        Tensor C = cat({A, B}, 0);
        std::cout << "cat([A, B], dim=0) shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 3])" << std::endl;

        std::cout << "Result values: [";
        for (int i = 0; i < 12; i++) std::cout << C.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected:      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]" << std::endl;
    }

    // Test 2: Concatenation along dim 1
    std::cout << "\n--- 2. Concatenate 2D tensors along dim 1 (columns) ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B({2, 2}, {7, 8, 9, 10});

        std::cout << "Tensor A [2x3]: [1, 2, 3, 4, 5, 6]" << std::endl;
        std::cout << "Tensor B [2x2]: [7, 8, 9, 10]" << std::endl;

        Tensor C = cat({A, B}, 1);
        std::cout << "cat([A, B], dim=1) shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 5])" << std::endl;

        std::cout << "Result row 0: [";
        for (int i = 0; i < 5; i++) std::cout << C.data()[i] << " ";
        std::cout << "] (expected: [1, 2, 3, 7, 8])" << std::endl;

        std::cout << "Result row 1: [";
        for (int i = 5; i < 10; i++) std::cout << C.data()[i] << " ";
        std::cout << "] (expected: [4, 5, 6, 9, 10])" << std::endl;
    }

    // Test 3: Negative indexing
    std::cout << "\n--- 3. Negative dimension indexing ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, {7, 8, 9, 10, 11, 12});

        Tensor C_neg1 = cat({A, B}, -1);  // Last dimension
        Tensor C_neg2 = cat({A, B}, -2);  // Second-to-last dimension

        std::cout << "cat([A, B], dim=-1) shape: [";
        for (auto s : C_neg1.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 6] - concatenate columns)" << std::endl;

        std::cout << "cat([A, B], dim=-2) shape: [";
        for (auto s : C_neg2.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 3] - concatenate rows)" << std::endl;
    }

    // Test 4: Multiple tensors
    std::cout << "\n--- 4. Concatenate multiple tensors ---" << std::endl;
    {
        Tensor A({2, 2}, {1, 2, 3, 4});
        Tensor B({2, 2}, {5, 6, 7, 8});
        Tensor C({2, 2}, {9, 10, 11, 12});

        Tensor result = cat({A, B, C}, 0);
        std::cout << "cat([A, B, C], dim=0) shape: [";
        for (auto s : result.shape()) std::cout << s << " ";
        std::cout << "] (expected: [6, 2])" << std::endl;

        std::cout << "Result values: [";
        for (int i = 0; i < 12; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected:      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]" << std::endl;
    }

    // Test 5: 3D tensors
    std::cout << "\n--- 5. 3D Tensor Concatenation ---" << std::endl;
    {
        Tensor A({2, 3, 4}, std::vector<float>(24, 1.0f));
        Tensor B({2, 3, 4}, std::vector<float>(24, 2.0f));

        std::cout << "Tensor A [2x3x4]: all 1.0" << std::endl;
        std::cout << "Tensor B [2x3x4]: all 2.0" << std::endl;

        // Concatenate along dim 0
        Tensor C_dim0 = cat({A, B}, 0);
        std::cout << "cat([A, B], dim=0) shape: [";
        for (auto s : C_dim0.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 3, 4])" << std::endl;

        // Verify first and second halves
        bool first_half_ok = true;
        for (int i = 0; i < 24; i++) {
            if (std::abs(C_dim0.data()[i] - 1.0f) > 1e-5f) first_half_ok = false;
        }
        bool second_half_ok = true;
        for (int i = 24; i < 48; i++) {
            if (std::abs(C_dim0.data()[i] - 2.0f) > 1e-5f) second_half_ok = false;
        }
        std::cout << "First 24 elements are 1.0: " << (first_half_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
        std::cout << "Next 24 elements are 2.0:  " << (second_half_ok ? "✓ PASS" : "✗ FAIL") << std::endl;

        // Concatenate along dim 1
        Tensor C_dim1 = cat({A, B}, 1);
        std::cout << "cat([A, B], dim=1) shape: [";
        for (auto s : C_dim1.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 6, 4])" << std::endl;

        // Concatenate along dim 2 (last dimension)
        Tensor C_dim2 = cat({A, B}, 2);
        std::cout << "cat([A, B], dim=2) shape: [";
        for (auto s : C_dim2.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3, 8])" << std::endl;
    }

    // Test 6: 1D tensors
    std::cout << "\n--- 6. 1D Tensor Concatenation ---" << std::endl;
    {
        Tensor A({3}, {1, 2, 3});
        Tensor B({2}, {4, 5});
        Tensor C({4}, {6, 7, 8, 9});

        Tensor result = cat({A, B, C}, 0);
        std::cout << "cat([A, B, C], dim=0) shape: [";
        for (auto s : result.shape()) std::cout << s << " ";
        std::cout << "] (expected: [9])" << std::endl;

        std::cout << "Result: [";
        for (int i = 0; i < 9; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9]" << std::endl;
    }

    // Test 7: Non-contiguous tensors
    std::cout << "\n--- 7. Non-contiguous Tensor Handling ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B({3, 2}, {7, 8, 9, 10, 11, 12});

        Tensor B_T = B.transpose();  // [2, 3], non-contiguous
        std::cout << "Tensor A [2x3]: contiguous" << std::endl;
        std::cout << "Tensor B_T [2x3]: non-contiguous (transposed)" << std::endl;
        std::cout << "B_T is contiguous: " << (B_T.is_contiguous() ? "yes" : "no") << std::endl;

        Tensor C = cat({A, B_T}, 0);
        std::cout << "cat([A, B_T], dim=0) shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 3])" << std::endl;
        std::cout << "Result is contiguous: " << (C.is_contiguous() ? "✓ yes" : "✗ no") << std::endl;
    }

    // Test 8: Different sizes in concat dimension
    std::cout << "\n--- 8. Different Sizes in Concatenation Dimension ---" << std::endl;
    {
        Tensor A({2, 2}, {1, 2, 3, 4});
        Tensor B({3, 2}, {5, 6, 7, 8, 9, 10});
        Tensor C({1, 2}, {11, 12});

        Tensor result = cat({A, B, C}, 0);
        std::cout << "Tensors with shapes [2,2], [3,2], [1,2]" << std::endl;
        std::cout << "cat result shape: [";
        for (auto s : result.shape()) std::cout << s << " ";
        std::cout << "] (expected: [6, 2])" << std::endl;

        std::cout << "Result: [";
        for (int i = 0; i < 12; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]" << std::endl;
    }

    // Test 9: Real-world use case - batch concatenation
    std::cout << "\n--- 9. Real-world Use Case: Batch Concatenation ---" << std::endl;
    {
        // Simulate concatenating mini-batches
        Tensor batch1({32, 128}, std::vector<float>(32 * 128, 1.0f));  // Batch of 32 samples
        Tensor batch2({16, 128}, std::vector<float>(16 * 128, 2.0f));  // Batch of 16 samples
        Tensor batch3({24, 128}, std::vector<float>(24 * 128, 3.0f));  // Batch of 24 samples

        Tensor combined = cat({batch1, batch2, batch3}, 0);

        std::cout << "Combining batches of sizes 32, 16, 24 with 128 features each" << std::endl;
        std::cout << "Combined shape: [";
        for (auto s : combined.shape()) std::cout << s << " ";
        std::cout << "] (expected: [72, 128])" << std::endl;

        // Verify some values
        bool batch1_ok = std::abs(combined.data()[0] - 1.0f) < 1e-5f;
        bool batch2_ok = std::abs(combined.data()[32 * 128] - 2.0f) < 1e-5f;
        bool batch3_ok = std::abs(combined.data()[(32 + 16) * 128] - 3.0f) < 1e-5f;

        std::cout << "Batch 1 values correct: " << (batch1_ok ? "✓" : "✗") << std::endl;
        std::cout << "Batch 2 values correct: " << (batch2_ok ? "✓" : "✗") << std::endl;
        std::cout << "Batch 3 values correct: " << (batch3_ok ? "✓" : "✗") << std::endl;
    }

    // Test 10: Single tensor (edge case)
    std::cout << "\n--- 10. Edge Case: Single Tensor ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B = cat({A}, 0);

        std::cout << "cat([A], dim=0) shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3])" << std::endl;

        // Should be a copy, not the same tensor
        B.data()[0] = 99.0f;
        bool is_copy = (A.data()[0] != 99.0f);
        std::cout << "Result is a copy (not same tensor): " << (is_copy ? "✓ yes" : "✗ no") << std::endl;
    }

    std::cout << "\n===== END OF CAT EXAMPLES =====" << std::endl;

    // ========================================
    // Stack Examples
    // ========================================
    std::cout << "\n===== STACK EXAMPLES =====" << std::endl;

    // Example 1: Basic 2D stack along dim 0 (prepend - creates new first dimension)
    {
        std::cout << "\n--- Stack Example 1: 2D stack along dim 0 ---" << std::endl;
        Tensor A({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, std::vector<float>{7, 8, 9, 10, 11, 12});

        Tensor result = cpptensor::stack({A, B}, 0);
        std::cout << "A shape: ["; for (auto s : A.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "B shape: ["; for (auto s : B.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B], dim=0) shape: ["; for (auto s : result.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Expected shape: [2 2 3]" << std::endl;
        std::cout << "Result values: [";
        for (int i = 0; i < 12; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: First [1,2,3,4,5,6] then [7,8,9,10,11,12]" << std::endl;
    }

    // Example 2: Stack along dim 1 (insert middle dimension)
    {
        std::cout << "\n--- Stack Example 2: 2D stack along dim 1 ---" << std::endl;
        Tensor A({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, std::vector<float>{7, 8, 9, 10, 11, 12});

        Tensor result = cpptensor::stack({A, B}, 1);
        std::cout << "A shape: ["; for (auto s : A.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "B shape: ["; for (auto s : B.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B], dim=1) shape: ["; for (auto s : result.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Expected shape: [2 2 3]" << std::endl;
        std::cout << "Result values: [";
        for (int i = 0; i < 12; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: For each row, first from A then from B" << std::endl;
    }

    // Example 3: Stack along dim 2 (append - creates new last dimension)
    {
        std::cout << "\n--- Stack Example 3: 2D stack along dim 2 ---" << std::endl;
        Tensor A({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, std::vector<float>{7, 8, 9, 10, 11, 12});

        Tensor result = cpptensor::stack({A, B}, 2);
        std::cout << "A shape: ["; for (auto s : A.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "B shape: ["; for (auto s : B.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B], dim=2) shape: ["; for (auto s : result.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Expected shape: [2 3 2]" << std::endl;
        std::cout << "Result values: [";
        for (int i = 0; i < 12; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: For each position [i,j], result[i][j] = [A[i][j], B[i][j]]" << std::endl;
    }

    // Example 4: Negative dimension indexing
    {
        std::cout << "\n--- Stack Example 4: Negative dimension indexing ---" << std::endl;
        Tensor A({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, std::vector<float>{7, 8, 9, 10, 11, 12});

        Tensor result1 = cpptensor::stack({A, B}, -1);  // Same as dim=2 for 2D tensors
        Tensor result2 = cpptensor::stack({A, B}, -2);  // Same as dim=1 for 2D tensors
        Tensor result3 = cpptensor::stack({A, B}, -3);  // Same as dim=0 for 2D tensors

        std::cout << "stack([A, B], dim=-1) shape: ["; for (auto s : result1.shape()) std::cout << s << " "; std::cout << "] (same as dim=2)" << std::endl;
        std::cout << "stack([A, B], dim=-2) shape: ["; for (auto s : result2.shape()) std::cout << s << " "; std::cout << "] (same as dim=1)" << std::endl;
        std::cout << "stack([A, B], dim=-3) shape: ["; for (auto s : result3.shape()) std::cout << s << " "; std::cout << "] (same as dim=0)" << std::endl;
    }

    // Example 5: Multiple tensors (3+ stacked)
    {
        std::cout << "\n--- Stack Example 5: Multiple tensors (4 tensors) ---" << std::endl;
        Tensor A({2, 2}, std::vector<float>{1, 2, 3, 4});
        Tensor B({2, 2}, std::vector<float>{5, 6, 7, 8});
        Tensor C({2, 2}, std::vector<float>{9, 10, 11, 12});
        Tensor D({2, 2}, std::vector<float>{13, 14, 15, 16});

        Tensor result = cpptensor::stack({A, B, C, D}, 0);
        std::cout << "Input shape: ["; for (auto s : A.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B, C, D], dim=0) shape: ["; for (auto s : result.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Expected shape: [4 2 2]" << std::endl;
        std::cout << "Result values: [";
        for (int i = 0; i < 16; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: 4 [2,2] matrices stacked: [1-4],[5-8],[9-12],[13-16]" << std::endl;
    }

    // Example 6: 3D tensor stacking (all dimensions)
    {
        std::cout << "\n--- Stack Example 6: 3D tensor stacking ---" << std::endl;
        Tensor A({2, 2, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
        Tensor B({2, 2, 2}, std::vector<float>{9, 10, 11, 12, 13, 14, 15, 16});

        Tensor result0 = cpptensor::stack({A, B}, 0);
        Tensor result1 = cpptensor::stack({A, B}, 1);
        Tensor result2 = cpptensor::stack({A, B}, 2);
        Tensor result3 = cpptensor::stack({A, B}, 3);

        std::cout << "Input shape: ["; for (auto s : A.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B], dim=0) shape: ["; for (auto s : result0.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B], dim=1) shape: ["; for (auto s : result1.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B], dim=2) shape: ["; for (auto s : result2.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A, B], dim=3) shape: ["; for (auto s : result3.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "All create [2 2 2 2] but with different element ordering" << std::endl;
    }

    // Example 7: 1D tensor stacking (vector stacking)
    {
        std::cout << "\n--- Stack Example 7: 1D tensor stacking ---" << std::endl;
        Tensor v1({4}, std::vector<float>{1, 2, 3, 4});
        Tensor v2({4}, std::vector<float>{5, 6, 7, 8});
        Tensor v3({4}, std::vector<float>{9, 10, 11, 12});

        Tensor result0 = cpptensor::stack({v1, v2, v3}, 0);  // Creates [3, 4]
        Tensor result1 = cpptensor::stack({v1, v2, v3}, 1);  // Creates [4, 3]

        std::cout << "Input shape: ["; for (auto s : v1.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([v1, v2, v3], dim=0) shape: ["; for (auto s : result0.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Stack dim=0 values: [";
        for (int i = 0; i < 12; i++) std::cout << result0.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: 3 rows [1,2,3,4], [5,6,7,8], [9,10,11,12]" << std::endl;

        std::cout << "\nstack([v1, v2, v3], dim=1) shape: ["; for (auto s : result1.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Stack dim=1 values: [";
        for (int i = 0; i < 12; i++) std::cout << result1.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: 4 rows with columns [1,5,9], [2,6,10], [3,7,11], [4,8,12]" << std::endl;
    }

    // Example 8: Single tensor edge case
    {
        std::cout << "\n--- Stack Example 8: Single tensor edge case ---" << std::endl;
        Tensor A({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});

        Tensor result = cpptensor::stack({A}, 0);
        std::cout << "Input shape: ["; for (auto s : A.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "stack([A], dim=0) shape: ["; for (auto s : result.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Expected shape: [1 2 3]" << std::endl;
        std::cout << "Result values: [";
        for (int i = 0; i < 6; i++) std::cout << result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: Just adds a dimension of size 1" << std::endl;
    }

    // Example 9: Real-world use case - image batch stacking
    {
        std::cout << "\n--- Stack Example 9: Image batch stacking ---" << std::endl;
        // Simulate 3 RGB images of size 32x32
        Tensor img1 = Tensor::full({3, 32, 32}, 1.0f);  // Channels, Height, Width
        Tensor img2 = Tensor::full({3, 32, 32}, 2.0f);
        Tensor img3 = Tensor::full({3, 32, 32}, 3.0f);

        Tensor batch = cpptensor::stack({img1, img2, img3}, 0);
        std::cout << "Single image shape: ["; for (auto s : img1.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Batch shape: ["; for (auto s : batch.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "Expected shape: [3 3 32 32] (batch_size, channels, height, width)" << std::endl;
        std::cout << "Batch[0] first element: " << batch.data()[0] << " (expected: 1.0)" << std::endl;
        std::cout << "Batch[1] first element: " << batch.data()[3*32*32] << " (expected: 2.0)" << std::endl;
        std::cout << "Batch[2] first element: " << batch.data()[2*3*32*32] << " (expected: 3.0)" << std::endl;
    }

    // Example 10: Compare stack vs cat behavior
    {
        std::cout << "\n--- Stack Example 10: Stack vs Cat comparison ---" << std::endl;
        Tensor A({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, std::vector<float>{7, 8, 9, 10, 11, 12});

        // Cat along dim 0: [2,3] + [2,3] -> [4,3]
        Tensor cat_result = cpptensor::cat({A, B}, 0);
        // Stack along dim 0: [2,3] + [2,3] -> [2,2,3]
        Tensor stack_result = cpptensor::stack({A, B}, 0);

        std::cout << "Input shapes: ["; for (auto s : A.shape()) std::cout << s << " "; std::cout << "], ["; for (auto s : B.shape()) std::cout << s << " "; std::cout << "]" << std::endl;
        std::cout << "cat([A, B], dim=0) shape: ["; for (auto s : cat_result.shape()) std::cout << s << " "; std::cout << "] (concatenates along existing dim)" << std::endl;
        std::cout << "stack([A, B], dim=0) shape: ["; for (auto s : stack_result.shape()) std::cout << s << " "; std::cout << "] (creates NEW dimension)" << std::endl;

        std::cout << "\nCat result: [";
        for (int i = 0; i < 12; i++) std::cout << cat_result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Stack result: [";
        for (int i = 0; i < 12; i++) std::cout << stack_result.data()[i] << " ";
        std::cout << "]" << std::endl;

        std::cout << "\nKey difference: Cat joins along existing dimension (increases that dimension's size)" << std::endl;
        std::cout << "                Stack creates NEW dimension and joins there" << std::endl;
    }

    std::cout << "\n===== END OF STACK EXAMPLES =====" << std::endl;

    // ====== SLICING OPERATIONS EXAMPLES ======
    std::cout << "\n\n===== SLICING OPERATIONS EXAMPLES =====" << std::endl;

    // 1. Basic Slicing - Simple Range
    std::cout << "\n--- 1. Basic Slicing - Simple Range ---" << std::endl;
    {
        Tensor A({5, 4}, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
           13, 14, 15, 16,
           17, 18, 19, 20
        });
        std::cout << "Original tensor A [5×4]:" << std::endl;
        A.print();

        // Slice rows [1:4)
        Tensor B = A.slice(0, 1, 4);
        std::cout << "\nA.slice(0, 1, 4) - rows [1:4), shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
        B.print();
        std::cout << "Expected: rows 1-3 (indices 1,2,3)" << std::endl;

        // Slice columns [1:3)
        Tensor C = A.slice(1, 1, 3);
        std::cout << "\nA.slice(1, 1, 3) - cols [1:3), shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
        C.print();
        std::cout << "Expected: columns 1-2" << std::endl;
    }

    // 2. Zero-Copy Verification
    std::cout << "\n--- 2. Zero-Copy Verification (Slice Shares Data) ---" << std::endl;
    {
        Tensor A({3, 4}, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12
        });
        std::cout << "Original A:" << std::endl;
        A.print();

        Tensor B = A.slice(0, 1, 3);  // Rows [1:3)
        std::cout << "\nSliced B = A.slice(0, 1, 3):" << std::endl;
        B.print();

        // Modify slice - should modify original
        B.data()[0] = 999.0f;
        std::cout << "\nAfter modifying B.data()[0] = 999:" << std::endl;
        std::cout << "B: "; B.print();
        std::cout << "A: "; A.print();
        std::cout << "✓ Data is shared (zero-copy view)!" << std::endl;
    }

    // 3. Negative Indices
    std::cout << "\n--- 3. Negative Indices (Python-style) ---" << std::endl;
    {
        Tensor A({5, 6}, std::vector<float>(30));
        for (size_t i = 0; i < 30; ++i) A.data()[i] = static_cast<float>(i);

        std::cout << "Tensor A [5×6] with values 0-29" << std::endl;

        // Last 3 rows: [-3:end]
        Tensor B = A.slice(0, -3, std::nullopt);
        std::cout << "\nA.slice(0, -3, nullopt) - last 3 rows, shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
        std::cout << "First element: " << B.impl()->data_ptr()[0] << " (expected: 12 = element at row 2)" << std::endl;

        // Last 2 columns: [-2:end]
        Tensor C = A.slice(1, -2, std::nullopt);
        std::cout << "\nA.slice(1, -2, nullopt) - last 2 cols, shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        // From start to -2 (exclude last 2 rows)
        Tensor D = A.slice(0, std::nullopt, -2);
        std::cout << "\nA.slice(0, nullopt, -2) - all but last 2 rows, shape: [";
        for (auto s : D.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 6])" << std::endl;
    }

    // 4. Step/Stride Parameter
    std::cout << "\n--- 4. Step/Stride Parameter ---" << std::endl;
    {
        Tensor A({4, 8}, std::vector<float>(32));
        for (size_t i = 0; i < 32; ++i) A.data()[i] = static_cast<float>(i);

        std::cout << "Tensor A [4×8] with values 0-31" << std::endl;

        // Every 2nd row
        Tensor B = A.slice(0, std::nullopt, std::nullopt, 2);
        std::cout << "\nA.slice(0, nullopt, nullopt, 2) - every 2nd row, shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 8])" << std::endl;
        std::cout << "First row values: ";
        for (int i = 0; i < 8; ++i) std::cout << B.impl()->data_ptr()[i * B.stride()[1]] << " ";
        std::cout << "(expected: 0-7)" << std::endl;
        std::cout << "Second row values: ";
        for (int i = 0; i < 8; ++i) std::cout << B.impl()->data_ptr()[B.stride()[0] + i * B.stride()[1]] << " ";
        std::cout << "(expected: 16-23)" << std::endl;

        // Every 3rd column
        Tensor C = A.slice(1, 0, std::nullopt, 3);
        std::cout << "\nA.slice(1, 0, nullopt, 3) - every 3rd col, shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 3])" << std::endl;

        // Stride with range
        Tensor D = A.slice(1, 1, 7, 2);  // cols [1:7:2] = [1,3,5]
        std::cout << "\nA.slice(1, 1, 7, 2) - cols [1:7:2], shape: [";
        for (auto s : D.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 3])" << std::endl;
    }

    // 5. Edge Cases
    std::cout << "\n--- 5. Edge Cases ---" << std::endl;
    {
        Tensor A({3, 5}, std::vector<float>(15, 1.0f));

        // Empty slice (start >= end)
        Tensor B = A.slice(0, 2, 2);
        std::cout << "\nA.slice(0, 2, 2) - empty slice, shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "] (expected: [0, 5])" << std::endl;

        // Single element slice
        Tensor C = A.slice(0, 1, 2);
        std::cout << "\nA.slice(0, 1, 2) - single row, shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "] (expected: [1, 5])" << std::endl;

        // Entire dimension (default params)
        Tensor D = A.slice(0);
        std::cout << "\nA.slice(0) - entire dim 0, shape: [";
        for (auto s : D.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 5], same as original)" << std::endl;

        // Out of bounds clamping
        Tensor E = A.slice(0, -10, 100);
        std::cout << "\nA.slice(0, -10, 100) - clamped to bounds, shape: [";
        for (auto s : E.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 5], entire range)" << std::endl;
    }

    // 6. 3D Tensor Slicing
    std::cout << "\n--- 6. 3D Tensor Slicing ---" << std::endl;
    {
        Tensor A({4, 5, 6}, std::vector<float>(120));
        for (size_t i = 0; i < 120; ++i) A.data()[i] = static_cast<float>(i);

        std::cout << "3D Tensor A [4×5×6] with values 0-119" << std::endl;

        // Slice first dimension
        Tensor B = A.slice(0, 1, 3);
        std::cout << "\nA.slice(0, 1, 3) - batch [1:3), shape: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 5, 6])" << std::endl;

        // Slice middle dimension
        Tensor C = A.slice(1, 2, 5);
        std::cout << "\nA.slice(1, 2, 5) - middle dim [2:5), shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 3, 6])" << std::endl;

        // Slice last dimension
        Tensor D = A.slice(2, 1, 5);
        std::cout << "\nA.slice(2, 1, 5) - last dim [1:5), shape: [";
        for (auto s : D.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 5, 4])" << std::endl;

        // Negative indexing on 3D
        Tensor E = A.slice(-1, -3, -1);  // Last dim, last 2 elements
        std::cout << "\nA.slice(-1, -3, -1) - last dim [-3:-1), shape: [";
        for (auto s : E.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 5, 2])" << std::endl;
    }

    // 7. Chained Slicing
    std::cout << "\n--- 7. Chained Slicing Operations ---" << std::endl;
    {
        Tensor A({10, 20, 30}, std::vector<float>(6000));
        for (size_t i = 0; i < 6000; ++i) A.data()[i] = static_cast<float>(i);

        std::cout << "Original A [10×20×30]" << std::endl;

        // Chain multiple slices
        Tensor B = A.slice(0, 2, 8);        // [6, 20, 30]
        Tensor C = B.slice(1, 5, 15);       // [6, 10, 30]
        Tensor D = C.slice(2, 10, 25, 2);   // [6, 10, 8]

        std::cout << "After A.slice(0,2,8): [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        std::cout << "After .slice(1,5,15): [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        std::cout << "After .slice(2,10,25,2): [";
        for (auto s : D.shape()) std::cout << s << " ";
        std::cout << "] (expected: [6, 10, 8])" << std::endl;

        // Verify data sharing through the chain
        // D[0,0,0] corresponds to A[2, 5, 10]
        // Read original value first
        size_t expected_idx = 2 * 20 * 30 + 5 * 30 + 10;
        float original_val = A.data()[expected_idx];
        std::cout << "\nOriginal A[2,5,10] = A.data()[" << expected_idx << "] = " << original_val << std::endl;

        // Modify through D's data_ptr (which accounts for offset properly)
        float* d_ptr = D.impl()->data_ptr();
        d_ptr[0] = 7777.0f;

        std::cout << "After modifying D[0,0,0] via data_ptr:" << std::endl;
        std::cout << "  A.data()[" << expected_idx << "] = " << A.data()[expected_idx] << " (expected: 7777)" << std::endl;
        std::cout << "✓ Chained slices maintain zero-copy property!" << std::endl;
    }

    // 8. Slicing with Permute/Transpose
    std::cout << "\n--- 8. Slicing Combined with Permute/Transpose ---" << std::endl;
    {
        Tensor A({3, 4, 5}, std::vector<float>(60));
        for (size_t i = 0; i < 60; ++i) A.data()[i] = static_cast<float>(i);

        std::cout << "Original A [3×4×5]" << std::endl;

        // Permute then slice
        Tensor B = A.permute({2, 0, 1});  // [5, 3, 4]
        Tensor C = B.slice(0, 1, 4);       // [3, 3, 4]
        std::cout << "A.permute({2,0,1}).slice(0,1,4) shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "] (expected: [3, 3, 4])" << std::endl;

        // Slice then transpose
        Tensor D = A.slice(0, 1, 3);       // [2, 4, 5]
        Tensor E = D.transpose(1, 2);      // [2, 5, 4]
        std::cout << "A.slice(0,1,3).transpose(1,2) shape: [";
        for (auto s : E.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 5, 4])" << std::endl;

        // Complex chain
        Tensor F = A.slice(1, 1, 3)        // [3, 2, 5]
                     .permute({2, 1, 0})    // [5, 2, 3]
                     .slice(0, 0, 4, 2)     // [2, 2, 3]
                     .transpose();          // [2, 3, 2] (swap last 2 dims)
        std::cout << "Complex chain final shape: [";
        for (auto s : F.shape()) std::cout << s << " ";
        std::cout << "] (expected: [2, 3, 2])" << std::endl;
    }

    // 9. Real-World Use Case: Batch Processing
    std::cout << "\n--- 9. Real-World Use Case: Batch Processing ---" << std::endl;
    {
        // Simulate video data: [batch=8, frames=100, height=64, width=64, channels=3]
        Tensor video({8, 100, 64, 64, 3}, std::vector<float>(8*100*64*64*3, 1.0f));

        std::cout << "Video tensor [8, 100, 64, 64, 3]" << std::endl;
        std::cout << "  (8 videos, 100 frames each, 64×64 resolution, RGB)" << std::endl;

        // Extract middle 60 frames from all videos
        Tensor middle_frames = video.slice(1, 20, 80);
        std::cout << "\nExtract frames [20:80): [";
        for (auto s : middle_frames.shape()) std::cout << s << " ";
        std::cout << "] (expected: [8, 60, 64, 64, 3])" << std::endl;

        // Extract subset of batches
        Tensor batch_subset = video.slice(0, 2, 6);
        std::cout << "Extract batches [2:6): [";
        for (auto s : batch_subset.shape()) std::cout << s << " ";
        std::cout << "] (expected: [4, 100, 64, 64, 3])" << std::endl;

        // Downsample spatially (every 2nd pixel)
        Tensor downsampled_h = video.slice(2, 0, std::nullopt, 2);  // Height
        Tensor downsampled = downsampled_h.slice(3, 0, std::nullopt, 2);  // Width
        std::cout << "Spatial downsampling (every 2nd pixel): [";
        for (auto s : downsampled.shape()) std::cout << s << " ";
        std::cout << "] (expected: [8, 100, 32, 32, 3])" << std::endl;

        // Temporal downsampling (every 5th frame)
        Tensor temporal_down = video.slice(1, 0, std::nullopt, 5);
        std::cout << "Temporal downsampling (every 5th frame): [";
        for (auto s : temporal_down.shape()) std::cout << s << " ";
        std::cout << "] (expected: [8, 20, 64, 64, 3])" << std::endl;
    }

    // 10. Error Handling
    std::cout << "\n--- 10. Error Handling and Validation ---" << std::endl;
    {
        Tensor A({3, 4, 5}, std::vector<float>(60, 1.0f));

        std::cout << "Testing error conditions on tensor [3, 4, 5]..." << std::endl;

        // Test dimension out of range
        try {
            Tensor B = A.slice(5, 0, 2);
            std::cout << "✗ FAIL: Should have thrown for dim=5" << std::endl;
        } catch (const std::runtime_error& e) {
            std::cout << "✓ PASS: Caught dimension out of range: " << e.what() << std::endl;
        }

        // Test invalid step (non-positive)
        try {
            Tensor B = A.slice(0, 0, 2, 0);
            std::cout << "✗ FAIL: Should have thrown for step=0" << std::endl;
        } catch (const std::runtime_error& e) {
            std::cout << "✓ PASS: Caught invalid step: " << e.what() << std::endl;
        }

        try {
            Tensor B = A.slice(0, 0, 2, -1);
            std::cout << "✗ FAIL: Should have thrown for step=-1" << std::endl;
        } catch (const std::runtime_error& e) {
            std::cout << "✓ PASS: Caught negative step: " << e.what() << std::endl;
        }

        // Valid cases that should work
        try {
            Tensor B = A.slice(0, 10, 20);  // Out of bounds, should clamp
            std::cout << "✓ PASS: Out-of-bounds indices clamped successfully, shape: [";
            for (auto s : B.shape()) std::cout << s << " ";
            std::cout << "]" << std::endl;
        } catch (...) {
            std::cout << "✗ FAIL: Should have clamped out-of-bounds indices" << std::endl;
        }
    }

    // 11. Contiguous() After Slicing with Stride
    std::cout << "\n--- 11. Making Sliced Strided Tensor Contiguous ---" << std::endl;
    {
        Tensor A({6, 8}, std::vector<float>(48));
        for (size_t i = 0; i < 48; ++i) A.data()[i] = static_cast<float>(i);

        // Slice with stride
        Tensor B = A.slice(0, 0, std::nullopt, 2);  // Every 2nd row
        std::cout << "A [6×8], slice every 2nd row: [";
        for (auto s : B.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;
        std::cout << "B is contiguous: " << (B.is_contiguous() ? "yes" : "no") << std::endl;

        // Make contiguous
        Tensor C = B.contiguous();
        std::cout << "After contiguous(): " << (C.is_contiguous() ? "yes" : "no") << std::endl;
        std::cout << "C shape: [";
        for (auto s : C.shape()) std::cout << s << " ";
        std::cout << "]" << std::endl;

        // Verify data values
        std::cout << "First row of C: ";
        for (int i = 0; i < 8; ++i) std::cout << C.data()[i] << " ";
        std::cout << "\nExpected: 0-7" << std::endl;
        std::cout << "Second row of C: ";
        for (int i = 8; i < 16; ++i) std::cout << C.data()[i] << " ";
        std::cout << "\nExpected: 16-23" << std::endl;
    }

    // 12. Performance Consideration Example
    std::cout << "\n--- 12. Performance: Zero-Copy vs Contiguous ---" << std::endl;
    {
        std::cout << "Slicing is O(1) - instant view creation" << std::endl;
        std::cout << "No data copying until contiguous() is called" << std::endl;
        std::cout << "Use slice() for quick sub-tensor access" << std::endl;
        std::cout << "Call contiguous() only when needed for operations requiring it" << std::endl;

        Tensor A = Tensor::randn({1000, 1000});

        auto t1 = std::chrono::high_resolution_clock::now();
        Tensor B = A.slice(0, 100, 900);
        auto t2 = std::chrono::high_resolution_clock::now();

        auto slice_time = std::chrono::duration<double, std::micro>(t2 - t1).count();
        std::cout << "\nSlice large tensor [1000×1000] → [800×1000]: "
                  << slice_time << " μs (instant!)" << std::endl;
    }

    std::cout << "\n===== END OF SLICING EXAMPLES =====" << std::endl;

    // ========== COMPARISON OPERATION EXAMPLES ==========
    std::cout << "\n\n===== COMPARISON OPERATION EXAMPLES =====" << std::endl;

    // Test 1: Basic equality comparisons
    std::cout << "\n--- 1. Basic Equality Comparisons (==) ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 2, 5, 3});
        Tensor B({2, 3}, {1, 0, 3, 2, 1, 4});

        std::cout << "Tensor A: [1, 2, 3, 2, 5, 3]" << std::endl;
        std::cout << "Tensor B: [1, 0, 3, 2, 1, 4]" << std::endl;

        // Using function API
        Tensor eq_result = cpptensor::eq(A, B);
        std::cout << "eq(A, B): [";
        for (int i = 0; i < 6; i++) std::cout << eq_result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 0, 1, 1, 0, 0] (1.0=true, 0.0=false)" << std::endl;

        // Using operator overload
        Tensor eq_op = A == B;
        std::cout << "A == B:   [";
        for (int i = 0; i < 6; i++) std::cout << eq_op.data()[i] << " ";
        std::cout << "] (same result via operator)" << std::endl;
    }

    // Test 2: Inequality comparisons
    std::cout << "\n--- 2. Inequality Comparisons (!=) ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 2, 5, 3});
        Tensor B({2, 3}, {1, 0, 3, 2, 1, 4});

        Tensor ne_result = cpptensor::ne(A, B);
        std::cout << "ne(A, B): [";
        for (int i = 0; i < 6; i++) std::cout << ne_result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 1, 0, 0, 1, 1] (opposite of ==)" << std::endl;

        Tensor ne_op = A != B;
        std::cout << "A != B:   [";
        for (int i = 0; i < 6; i++) std::cout << ne_op.data()[i] << " ";
        std::cout << "] (same result via operator)" << std::endl;
    }

    // Test 3: Greater-than comparisons
    std::cout << "\n--- 3. Greater-Than Comparisons (>) ---" << std::endl;
    {
        Tensor A({2, 3}, {5, 2, 8, 1, 9, 3});
        Tensor B({2, 3}, {3, 4, 8, 2, 6, 3});

        std::cout << "Tensor A: [5, 2, 8, 1, 9, 3]" << std::endl;
        std::cout << "Tensor B: [3, 4, 8, 2, 6, 3]" << std::endl;

        Tensor gt_result = cpptensor::gt(A, B);
        std::cout << "gt(A, B): [";
        for (int i = 0; i < 6; i++) std::cout << gt_result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 0, 0, 0, 1, 0] (5>3, 2<4, 8=8, 1<2, 9>6, 3=3)" << std::endl;

        Tensor gt_op = A > B;
        std::cout << "A > B:    [";
        for (int i = 0; i < 6; i++) std::cout << gt_op.data()[i] << " ";
        std::cout << "] (same result via operator)" << std::endl;
    }

    // Test 4: Less-than comparisons
    std::cout << "\n--- 4. Less-Than Comparisons (<) ---" << std::endl;
    {
        Tensor A({2, 3}, {5, 2, 8, 1, 9, 3});
        Tensor B({2, 3}, {3, 4, 8, 2, 6, 3});

        Tensor lt_result = cpptensor::lt(A, B);
        std::cout << "lt(A, B): [";
        for (int i = 0; i < 6; i++) std::cout << lt_result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 1, 0, 1, 0, 0] (opposite pattern of >)" << std::endl;

        Tensor lt_op = A < B;
        std::cout << "A < B:    [";
        for (int i = 0; i < 6; i++) std::cout << lt_op.data()[i] << " ";
        std::cout << "] (same result via operator)" << std::endl;
    }

    // Test 5: Greater-or-equal comparisons
    std::cout << "\n--- 5. Greater-or-Equal Comparisons (>=) ---" << std::endl;
    {
        Tensor A({2, 3}, {5, 2, 8, 1, 9, 3});
        Tensor B({2, 3}, {3, 4, 8, 2, 6, 3});

        Tensor ge_result = cpptensor::ge(A, B);
        std::cout << "ge(A, B): [";
        for (int i = 0; i < 6; i++) std::cout << ge_result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 0, 1, 0, 1, 1] (>= includes equality)" << std::endl;

        Tensor ge_op = A >= B;
        std::cout << "A >= B:   [";
        for (int i = 0; i < 6; i++) std::cout << ge_op.data()[i] << " ";
        std::cout << "] (same result via operator)" << std::endl;
    }

    // Test 6: Less-or-equal comparisons
    std::cout << "\n--- 6. Less-or-Equal Comparisons (<=) ---" << std::endl;
    {
        Tensor A({2, 3}, {5, 2, 8, 1, 9, 3});
        Tensor B({2, 3}, {3, 4, 8, 2, 6, 3});

        Tensor le_result = cpptensor::le(A, B);
        std::cout << "le(A, B): [";
        for (int i = 0; i < 6; i++) std::cout << le_result.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 1, 1, 1, 0, 1] (<= includes equality)" << std::endl;

        Tensor le_op = A <= B;
        std::cout << "A <= B:   [";
        for (int i = 0; i < 6; i++) std::cout << le_op.data()[i] << " ";
        std::cout << "] (same result via operator)" << std::endl;
    }

    // Test 7: Scalar comparisons
    std::cout << "\n--- 7. Scalar Comparisons ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 5, 3, 7, 2, 6});
        std::cout << "Tensor A: [1, 5, 3, 7, 2, 6]" << std::endl;

        // Compare with scalar
        Tensor gt_scalar = A > 4.0f;
        std::cout << "A > 4.0:  [";
        for (int i = 0; i < 6; i++) std::cout << gt_scalar.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 1, 0, 1, 0, 1] (values > 4)" << std::endl;

        Tensor eq_scalar = A == 3.0f;
        std::cout << "A == 3.0: [";
        for (int i = 0; i < 6; i++) std::cout << eq_scalar.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 0, 1, 0, 0, 0] (only third element is 3)" << std::endl;

        Tensor le_scalar = A <= 3.0f;
        std::cout << "A <= 3.0: [";
        for (int i = 0; i < 6; i++) std::cout << le_scalar.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 0, 1, 0, 1, 0] (values <= 3)" << std::endl;

        // Scalar on left side
        Tensor scalar_gt = 5.0f > A;
        std::cout << "5.0 > A:  [";
        for (int i = 0; i < 6; i++) std::cout << scalar_gt.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 0, 1, 0, 1, 0] (5 > each value)" << std::endl;
    }

    // Test 8: Broadcasting support
    std::cout << "\n--- 8. Broadcasting Support ---" << std::endl;
    {
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B({1, 3}, {2, 3, 4});  // Will broadcast to [2, 3]

        std::cout << "Tensor A [2x3]: [1, 2, 3, 4, 5, 6]" << std::endl;
        std::cout << "Tensor B [1x3]: [2, 3, 4] (broadcasts to [2x3])" << std::endl;

        Tensor ge_broadcast = A >= B;
        std::cout << "A >= B:   [";
        for (int i = 0; i < 6; i++) std::cout << ge_broadcast.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 0, 0, 1, 1, 1]" << std::endl;
        std::cout << "  Row 0: [1,2,3] >= [2,3,4] = [0,0,0]" << std::endl;
        std::cout << "  Row 1: [4,5,6] >= [2,3,4] = [1,1,1]" << std::endl;

        // Column vector broadcasting
        Tensor C({2, 1}, {3, 5});  // Will broadcast to [2, 3]
        std::cout << "\nTensor C [2x1]: [3, 5] (broadcasts to [2x3])" << std::endl;

        Tensor lt_broadcast = A < C;
        std::cout << "A < C:    [";
        for (int i = 0; i < 6; i++) std::cout << lt_broadcast.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [1, 1, 0, 1, 0, 0]" << std::endl;
        std::cout << "  Row 0: [1,2,3] < 3 = [1,1,0]" << std::endl;
        std::cout << "  Row 1: [4,5,6] < 5 = [1,0,0]" << std::endl;
    }

    // Test 9: Chaining comparisons with logical operations
    std::cout << "\n--- 9. Chaining Multiple Comparisons ---" << std::endl;
    {
        Tensor A({1, 6}, {1, 3, 5, 7, 9, 11});
        std::cout << "Tensor A: [1, 3, 5, 7, 9, 11]" << std::endl;

        // Find values in range [4, 8]
        Tensor ge_4 = A >= 4.0f;
        Tensor le_8 = A <= 8.0f;
        Tensor in_range = ge_4 * le_8;  // Element-wise AND via multiplication

        std::cout << "A >= 4:   [";
        for (int i = 0; i < 6; i++) std::cout << ge_4.data()[i] << " ";
        std::cout << "]" << std::endl;

        std::cout << "A <= 8:   [";
        for (int i = 0; i < 6; i++) std::cout << le_8.data()[i] << " ";
        std::cout << "]" << std::endl;

        std::cout << "In [4,8]: [";
        for (int i = 0; i < 6; i++) std::cout << in_range.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 0, 1, 1, 0, 0] (5 and 7 are in range)" << std::endl;
    }

    // Test 10: Real-world use case - thresholding
    std::cout << "\n--- 10. Real-World Use Case: Image Thresholding ---" << std::endl;
    {
        // Simulate grayscale image values [0-255]
        Tensor image({2, 4}, {50, 100, 150, 200, 75, 125, 175, 225});
        std::cout << "Image values: [50, 100, 150, 200, 75, 125, 175, 225]" << std::endl;

        float threshold = 128.0f;
        Tensor binary = image > threshold;

        std::cout << "Binary threshold (>128): [";
        for (int i = 0; i < 8; i++) std::cout << binary.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 0, 1, 1, 0, 0, 1, 1]" << std::endl;
        std::cout << "Creates binary mask: 0=dark pixels, 1=bright pixels" << std::endl;
    }

    // Test 11: Finding extremes
    std::cout << "\n--- 11. Finding Extremes (Min/Max Detection) ---" << std::endl;
    {
        Tensor data({2, 5}, {3, 1, 7, 2, 9, 4, 9, 5, 1, 8});
        std::cout << "Data: [3, 1, 7, 2, 9, 4, 9, 5, 1, 8]" << std::endl;

        float max_val = data.max().data()[0];
        float min_val = data.min().data()[0];
        std::cout << "Max value: " << max_val << ", Min value: " << min_val << std::endl;

        // Find all max positions
        Tensor is_max = data == max_val;
        std::cout << "Is max:   [";
        for (int i = 0; i < 10; i++) std::cout << is_max.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 0, 0, 0, 1, 0, 1, 0, 0, 0] (two 9s)" << std::endl;

        // Find all min positions
        Tensor is_min = data == min_val;
        std::cout << "Is min:   [";
        for (int i = 0; i < 10; i++) std::cout << is_min.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0] (two 1s)" << std::endl;
    }

    // Test 12: Counting with comparisons
    std::cout << "\n--- 12. Counting Elements via Comparisons ---" << std::endl;
    {
        Tensor scores({3, 4}, {45, 78, 92, 65, 88, 54, 73, 95, 61, 82, 77, 90});
        std::cout << "Test scores [3x4]: [45, 78, 92, 65, 88, 54, 73, 95, 61, 82, 77, 90]" << std::endl;

        // Count passing grades (>= 70)
        Tensor passing = scores >= 70.0f;
        float count_passing = passing.sum().data()[0];
        std::cout << "Passing (>=70): " << count_passing << " students" << std::endl;
        std::cout << "Expected: 9 students passed" << std::endl;

        // Count excellent grades (>= 90)
        Tensor excellent = scores >= 90.0f;
        float count_excellent = excellent.sum().data()[0];
        std::cout << "Excellent (>=90): " << count_excellent << " students" << std::endl;
        std::cout << "Expected: 3 students got 90+ (92, 95, 90)" << std::endl;

        // Count failing grades (< 60)
        Tensor failing = scores < 60.0f;
        float count_failing = failing.sum().data()[0];
        std::cout << "Failing (<60): " << count_failing << " students" << std::endl;
        std::cout << "Expected: 2 students failed (45, 54)" << std::endl;
    }

    // Test 13: All six comparison operators side-by-side
    std::cout << "\n--- 13. All Comparison Operators Side-by-Side ---" << std::endl;
    {
        Tensor A({1, 4}, {5, 3, 3, 1});
        Tensor B({1, 4}, {3, 3, 5, 2});
        std::cout << "Tensor A: [5, 3, 3, 1]" << std::endl;
        std::cout << "Tensor B: [3, 3, 5, 2]" << std::endl;

        std::cout << "A == B: [";
        Tensor eq = A == B;
        for (int i = 0; i < 4; i++) std::cout << eq.data()[i] << " ";
        std::cout << "] (only position 1 equal)" << std::endl;

        std::cout << "A != B: [";
        Tensor ne = A != B;
        for (int i = 0; i < 4; i++) std::cout << ne.data()[i] << " ";
        std::cout << "] (complement of ==)" << std::endl;

        std::cout << "A > B:  [";
        Tensor gt = A > B;
        for (int i = 0; i < 4; i++) std::cout << gt.data()[i] << " ";
        std::cout << "] (position 0: 5>3)" << std::endl;

        std::cout << "A < B:  [";
        Tensor lt = A < B;
        for (int i = 0; i < 4; i++) std::cout << lt.data()[i] << " ";
        std::cout << "] (positions 2,3: 3<5, 1<2)" << std::endl;

        std::cout << "A >= B: [";
        Tensor ge = A >= B;
        for (int i = 0; i < 4; i++) std::cout << ge.data()[i] << " ";
        std::cout << "] (>= includes equality)" << std::endl;

        std::cout << "A <= B: [";
        Tensor le = A <= B;
        for (int i = 0; i < 4; i++) std::cout << le.data()[i] << " ";
        std::cout << "] (<= includes equality)" << std::endl;

        std::cout << "\nVerification: (A >= B) + (A < B) should be all 1s" << std::endl;
        Tensor combined = ge + lt;
        std::cout << "ge + lt: [";
        for (int i = 0; i < 4; i++) std::cout << combined.data()[i] << " ";
        std::cout << "] ✓" << std::endl;
    }

    // Test 14: Negative numbers
    std::cout << "\n--- 14. Comparisons with Negative Numbers ---" << std::endl;
    {
        Tensor A({2, 3}, {-5, -2, 0, 2, 5, -3});
        Tensor B({2, 3}, {-3, -2, -1, 1, 4, -5});
        std::cout << "Tensor A: [-5, -2, 0, 2, 5, -3]" << std::endl;
        std::cout << "Tensor B: [-3, -2, -1, 1, 4, -5]" << std::endl;

        Tensor gt_neg = A > B;
        std::cout << "A > B: [";
        for (int i = 0; i < 6; i++) std::cout << gt_neg.data()[i] << " ";
        std::cout << "]" << std::endl;
        std::cout << "Expected: [0, 0, 1, 1, 1, 1]" << std::endl;
        std::cout << "  -5 > -3? No, -2 > -2? No, 0 > -1? Yes, 2 > 1? Yes, 5 > 4? Yes, -3 > -5? Yes" << std::endl;

        Tensor lt_zero = A < 0.0f;
        std::cout << "A < 0: [";
        for (int i = 0; i < 6; i++) std::cout << lt_zero.data()[i] << " ";
        std::cout << "] (negative values)" << std::endl;
        std::cout << "Expected: [1, 1, 0, 0, 0, 1]" << std::endl;
    }

    // Test 15: 3D tensor comparisons
    std::cout << "\n--- 15. 3D Tensor Comparisons ---" << std::endl;
    {
        Tensor A = Tensor::full({2, 2, 3}, 1.0f);
        // Modify some values
        A.data()[0] = 5.0f;
        A.data()[3] = 5.0f;
        A.data()[6] = 5.0f;
        A.data()[9] = 5.0f;

        std::cout << "3D tensor A [2x2x3] with some values = 5.0, others = 1.0" << std::endl;

        Tensor is_five = A == 5.0f;
        float count_fives = is_five.sum().data()[0];
        std::cout << "Count of 5.0 values: " << count_fives << " (expected: 4)" << std::endl;

        Tensor greater_than_3 = A > 3.0f;
        float count_gt3 = greater_than_3.sum().data()[0];
        std::cout << "Count of values > 3.0: " << count_gt3 << " (expected: 4)" << std::endl;

        std::cout << "Result shape: [";
        for (auto s : is_five.shape()) std::cout << s << " ";
        std::cout << "] (preserves input shape)" << std::endl;
    }

    // Test 16: Function vs Operator API
    std::cout << "\n--- 16. Function API vs Operator API ---" << std::endl;
    {
        Tensor A({1, 3}, {1, 2, 3});
        Tensor B({1, 3}, {2, 2, 2});

        // Function API
        Tensor func_eq = cpptensor::eq(A, B);
        Tensor func_ne = cpptensor::ne(A, B);
        Tensor func_gt = cpptensor::gt(A, B);
        Tensor func_lt = cpptensor::lt(A, B);
        Tensor func_ge = cpptensor::ge(A, B);
        Tensor func_le = cpptensor::le(A, B);

        // Operator API
        Tensor op_eq = A == B;
        Tensor op_ne = A != B;
        Tensor op_gt = A > B;
        Tensor op_lt = A < B;
        Tensor op_ge = A >= B;
        Tensor op_le = A <= B;

        std::cout << "Function API: eq(), ne(), gt(), lt(), ge(), le()" << std::endl;
        std::cout << "Operator API: ==, !=, >, <, >=, <=" << std::endl;
        std::cout << "Both produce identical results:" << std::endl;

        bool all_match = true;
        for (int i = 0; i < 3; i++) {
            if (func_eq.data()[i] != op_eq.data()[i] ||
                func_ne.data()[i] != op_ne.data()[i] ||
                func_gt.data()[i] != op_gt.data()[i] ||
                func_lt.data()[i] != op_lt.data()[i] ||
                func_ge.data()[i] != op_ge.data()[i] ||
                func_le.data()[i] != op_le.data()[i]) {
                all_match = false;
                break;
            }
        }
        std::cout << "  Verification: " << (all_match ? "✓ PASS - All match" : "✗ FAIL") << std::endl;
    }

    std::cout << "\n===== END OF COMPARISON EXAMPLES =====" << std::endl;

    std::cout << "\n===== END OF EXAMPLES =====" << std::endl;


    // Run performance tests
    benchmark_matmul(512, 512, 512);
    benchmark_matmul(1024, 1024, 1024);
    benchmark_matmul(2048, 2048, 2048);
    //benchmark_matmul(7700, 7700, 7700);

    benchmark_matmul_nd({2, 3, 4, 64, 64},  // M=64, K=64
                        {2, 3, 4, 64, 64}); // K=64, N=64

    benchmark_matmul_nd({4, 8, 3, 128, 256},
                        {4, 8, 3, 256, 128});

    // Tensordot benchmarks
    benchmark_tensordot({64,128,256}, {256,128,64}, std::vector<int>{1,2}, std::vector<int>{1,0}); // [64]x[64]
    benchmark_tensordot({16,32,64,128}, {64,128,32,16}, std::vector<int>{2,3}, std::vector<int>{0,1});

    //SVD benchmarks
    benchmark_svd(512, 512, true);     // Square matrix, full SVD
    benchmark_svd(1024, 512, false);   // Tall matrix, economy SVD
    benchmark_svd(512, 1024, false);   // Wide matrix, economy SVD

    // EIG benchmarks
    benchmark_eig_symmetric(256, true);   // Small symmetric matrix
    benchmark_eig_symmetric(512, true);   // Medium symmetric matrix
    benchmark_eig_symmetric(1024, false); // Large symmetric, eigenvalues only

    benchmark_eig(256, true);             // Small general matrix
    benchmark_eig(512, true);             // Medium general matrix
    benchmark_eig(1024, false);           // Large general, eigenvalues only



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

