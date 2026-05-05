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

