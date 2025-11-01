#pragma once

#include <vector>
#include <tuple>
#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

    /**
     * @brief Result structure for Singular Value Decomposition
     *
     * For matrix A [M×N], the SVD is: A = U * diag(S) * V^T
     * where:
     *   - U:  Left singular vectors (orthonormal columns)
     *   - S:  Singular values (non-negative, sorted descending)
     *   - Vt: Right singular vectors transposed (orthonormal rows)
     */
    struct SVDResult {
        Tensor U;      // Left singular vectors  [M×K] or [M×M] if full_matrices=true
        Tensor S;      // Singular values (1D)   [K] where K=min(M,N)
        Tensor Vt;     // Right singular vectors [K×N] or [N×N] if full_matrices=true
    };

    /**
     * @brief Singular Value Decomposition
     *
     * Computes the SVD of a 2D matrix: A = U * diag(S) * V^T
     *
     * Uses LAPACK's sgesvd routine (requires OpenBLAS/LAPACK).
     * Falls back to error if LAPACK is not available.
     *
     * @param A Input matrix [M×N], must be 2D
     * @param full_matrices
     *        - If true:  U is [M×M],    Vt is [N×N] (complete orthonormal bases)
     *        - If false: U is [M×K],    Vt is [K×N] (economy mode, K=min(M,N))
     *        Default: true (matches NumPy default)
     * @param compute_uv
     *        - If true:  Compute both U and Vt
     *        - If false: Only compute singular values S (faster)
     *        Default: true
     *
     * @return SVDResult containing U, S, and Vt tensors
     *
     * @throws std::runtime_error if:
     *         - Input is not 2D
     *         - Input is not on CPU
     *         - LAPACK is not available (build with -DUSE_OPENBLAS=ON)
     *         - SVD computation fails
     *
     * @example
     *   Tensor A = Tensor::randn({100, 50});
     *
     *   // Full SVD
     *   auto [U, S, Vt] = svd(A, true, true);
     *   // U: [100×100], S: [50], Vt: [50×50]
     *
     *   // Economy SVD (memory efficient)
     *   auto result = svd(A, false, true);
     *   // result.U: [100×50], result.S: [50], result.Vt: [50×50]
     *
     *   // Only singular values
     *   auto [_, S_only, __] = svd(A, false, false);
     */
    SVDResult svd(const Tensor& A, bool full_matrices = true, bool compute_uv = true);

} // namespace cpptensor
