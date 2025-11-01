// #include "cpptensor/ops/linearAlgebra/svd.hpp"
// #include <stdexcept>
// #include <algorithm>
// #include <cstring>
//
// #ifdef USE_OPENBLAS
//     extern "C" {
//         /**
//          * LAPACK single-precision SVD routine
//          *
//          * Computes the singular value decomposition of a real M-by-N matrix A:
//          *   A = U * SIGMA * V^T
//          *
//          * Arguments:
//          *   JOBU   - char: 'A' (all M cols of U), 'S' (min(M,N) cols), 'N' (no U)
//          *   JOBVT  - char: 'A' (all N rows of V^T), 'S' (min(M,N) rows), 'N' (no V^T)
//          *   M      - int: number of rows of A
//          *   N      - int: number of columns of A
//          *   A      - float[M*N]: input matrix (destroyed on output)
//          *   LDA    - int: leading dimension of A (>= M)
//          *   S      - float[min(M,N)]: singular values in descending order
//          *   U      - float[LDU*UCOL]: left singular vectors
//          *   LDU    - int: leading dimension of U
//          *   VT     - float[LDVT*N]: right singular vectors (transposed)
//          *   LDVT   - int: leading dimension of VT
//          *   WORK   - float[LWORK]: workspace array
//          *   LWORK  - int: size of WORK (-1 for query)
//          *   INFO   - int: output status (0=success, <0=illegal arg, >0=convergence failure)
//          */
//         void sgesvd_(
//             const char* jobu,
//             const char* jobvt,
//             const int* m,
//             const int* n,
//             float* a,
//             const int* lda,
//             float* s,
//             float* u,
//             const int* ldu,
//             float* vt,
//             const int* ldvt,
//             float* work,
//             const int* lwork,
//             int* info
//         );
//     }
// #endif
//
// namespace cpptensor {
//
//     SVDResult svd(const Tensor& A, bool full_matrices, bool compute_uv) {
//         // ===== Step 1: Validate Input =====
//         if (A.device_type() != DeviceType::CPU) {
//             throw std::runtime_error("svd: only CPU tensors supported");
//         }
//
//         const auto& shape = A.shape();
//         if (shape.size() != 2) {
//             throw std::runtime_error("svd: input must be 2D matrix, got " +
//                                     std::to_string(shape.size()) + "D tensor");
//         }
//
//         int M = static_cast<int>(shape[0]);  // Number of rows
//         int N = static_cast<int>(shape[1]);  // Number of columns
//         int K = std::min(M, N);              // Number of singular values
//
//         if (M == 0 || N == 0) {
//             throw std::runtime_error("svd: matrix dimensions cannot be zero");
//         }
//
//     #ifdef USE_OPENBLAS
//         // ===== Step 2: Prepare Input (LAPACK destroys input matrix) =====
//         std::vector<float> a_copy = A.data();
//
//         // ===== Step 3: Determine Job Specifications =====
//         // JOBU:  'A' = all M columns of U, 'S' = first min(M,N) columns, 'N' = no U
//         // JOBVT: 'A' = all N rows of V^T, 'S' = first min(M,N) rows, 'N' = no V^T
//         char jobu  = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
//         char jobvt = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
//
//         // ===== Step 4: Allocate Output Arrays =====
//         // Singular values (always computed)
//         std::vector<float> s(K);
//
//         // Left singular vectors U
//         int u_rows = M;
//         int u_cols = compute_uv ? (full_matrices ? M : K) : 1;  // At least 1 for LAPACK
//         std::vector<float> u_data(u_rows * u_cols);
//
//         // Right singular vectors V^T (transposed)
//         int vt_rows = compute_uv ? (full_matrices ? N : K) : 1;  // At least 1 for LAPACK
//         int vt_cols = N;
//         std::vector<float> vt_data(vt_rows * vt_cols);
//
//         // Leading dimensions (for column-major layout)
//         int lda  = M;
//         int ldu  = M;
//         int ldvt = compute_uv ? vt_rows : 1;
//
//         // ===== Step 5: Workspace Query =====
//         // First call with lwork=-1 to determine optimal workspace size
//         int lwork = -1;
//         float work_query;
//         int info;
//
//         sgesvd_(&jobu, &jobvt, &M, &N,
//                 a_copy.data(), &lda,
//                 s.data(),
//                 u_data.data(), &ldu,
//                 vt_data.data(), &ldvt,
//                 &work_query, &lwork,
//                 &info);
//
//         if (info != 0) {
//             throw std::runtime_error("svd: workspace query failed with info=" + std::to_string(info));
//         }
//
//         // Allocate optimal workspace
//         lwork = static_cast<int>(work_query);
//         std::vector<float> work(lwork);
//
//         // ===== Step 6: Perform SVD Computation =====
//         sgesvd_(&jobu, &jobvt, &M, &N,
//                 a_copy.data(), &lda,
//                 s.data(),
//                 u_data.data(), &ldu,
//                 vt_data.data(), &ldvt,
//                 work.data(), &lwork,
//                 &info);
//
//         // ===== Step 7: Check for Errors =====
//         if (info < 0) {
//             throw std::runtime_error("svd: LAPACK sgesvd illegal argument at position " +
//                                     std::to_string(-info));
//         } else if (info > 0) {
//             throw std::runtime_error("svd: LAPACK sgesvd failed to converge. " +
//                                     std::to_string(info) + " superdiagonals did not converge to zero");
//         }
//
//         // ===== Step 8: Construct Result Tensors =====
//         SVDResult result;
//
//         // Singular values (always present)
//         result.S = Tensor({static_cast<size_t>(K)}, s, DeviceType::CPU);
//
//         // Left and right singular vectors (if requested)
//         if (compute_uv) {
//             result.U = Tensor({static_cast<size_t>(u_rows), static_cast<size_t>(u_cols)},
//                              u_data, DeviceType::CPU);
//             result.Vt = Tensor({static_cast<size_t>(vt_rows), static_cast<size_t>(vt_cols)},
//                               vt_data, DeviceType::CPU);
//         } else {
//             // Return empty tensors when U and Vt not computed
//             result.U = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
//             result.Vt = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
//         }
//
//         return result;
//
//     #else
//         // ===== No LAPACK Available =====
//         throw std::runtime_error(
//             "svd: requires OpenBLAS/LAPACK library.\n"
//             "Please rebuild with: cmake -DUSE_OPENBLAS=ON ..\n"
//             "Make sure OpenBLAS is installed on your system."
//         );
//     #endif
//     }
//
// } // namespace cpptensor

#include "cpptensor/ops/linearAlgebra/svd.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifdef USE_OPENBLAS
// Use LAPACKE (C interface) which supports row-major layout directly
#include <lapacke.h>
#endif

namespace cpptensor {

SVDResult svd(const Tensor& A, bool full_matrices, bool compute_uv) {
    // ===== Step 1: Validate Input =====
    if (A.device_type() != DeviceType::CPU) {
        throw std::runtime_error("svd: only CPU tensors supported");
    }

    const auto& shape = A.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("svd: input must be 2D matrix, got " +
                                std::to_string(shape.size()) + "D tensor");
    }

    int M = static_cast<int>(shape[0]);  // Number of rows
    int N = static_cast<int>(shape[1]);  // Number of columns
    int K = std::min(M, N);              // Number of singular values

    if (M == 0 || N == 0) {
        throw std::runtime_error("svd: matrix dimensions cannot be zero");
    }

#ifdef USE_OPENBLAS
    // ===== Step 2: Prepare Input =====
    // LAPACKE supports row-major layout directly, so no transpose needed!
    std::vector<float> a_copy = A.data();

    // ===== Step 3: Determine Job Specifications =====
    // JOBU:  'A' = all M columns of U, 'S' = first min(M,N) columns, 'N' = no U
    // JOBVT: 'A' = all N rows of V^T, 'S' = first min(M,N) rows, 'N' = no V^T
    char jobu  = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
    char jobvt = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';

    // ===== Step 4: Allocate Output Arrays =====
    // Singular values (always computed)
    std::vector<float> s(K);

    // Left singular vectors U
    int u_rows = M;
    int u_cols = compute_uv ? (full_matrices ? M : K) : 1;
    std::vector<float> u_data(u_rows * u_cols);

    // Right singular vectors V^T (transposed)
    int vt_rows = compute_uv ? (full_matrices ? N : K) : 1;
    int vt_cols = N;
    std::vector<float> vt_data(vt_rows * vt_cols);

    // ===== Step 5: Allocate Superb (for LAPACKE interface) =====
    std::vector<float> superb(std::min(M, N) - 1);

    // ===== Step 6: Perform SVD using LAPACKE (row-major interface) =====
    int info = LAPACKE_sgesvd(
        LAPACK_ROW_MAJOR,   // Row-major layout (C/C++ convention)
        jobu,               // Compute U?
        jobvt,              // Compute V^T?
        M,                  // Number of rows
        N,                  // Number of columns
        a_copy.data(),      // Input matrix (destroyed)
        N,                  // Leading dimension (for row-major: number of columns)
        s.data(),           // Singular values output
        u_data.data(),      // U output
        u_cols,             // Leading dimension of U (for row-major)
        vt_data.data(),     // V^T output
        vt_cols,            // Leading dimension of V^T (for row-major)
        superb.data()       // Superdiagonal elements (for divide-and-conquer)
    );

    // ===== Step 7: Check for Errors =====
    if (info < 0) {
        throw std::runtime_error("svd: LAPACKE_sgesvd illegal argument at position " +
                                std::to_string(-info));
    } else if (info > 0) {
        throw std::runtime_error("svd: LAPACKE_sgesvd failed to converge. " +
                                std::to_string(info) + " superdiagonals did not converge to zero");
    }

    // ===== Step 8: Construct Result Tensors =====
    SVDResult result;

    // Singular values (always present)
    result.S = Tensor({static_cast<size_t>(K)}, s, DeviceType::CPU);

    // Left and right singular vectors (if requested)
    if (compute_uv) {
        result.U = Tensor({static_cast<size_t>(u_rows), static_cast<size_t>(u_cols)},
                         u_data, DeviceType::CPU);
        result.Vt = Tensor({static_cast<size_t>(vt_rows), static_cast<size_t>(vt_cols)},
                          vt_data, DeviceType::CPU);
    } else {
        // Return empty tensors when U and Vt not computed
        result.U = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
        result.Vt = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
    }

    return result;

#else
    // ===== No LAPACK Available =====
    throw std::runtime_error(
        "svd: requires OpenBLAS/LAPACK library.\n"
        "Please rebuild with: cmake -DUSE_OPENBLAS=ON ..\n"
        "Make sure OpenBLAS is installed on your system."
    );
#endif
}

} // namespace cpptensor


