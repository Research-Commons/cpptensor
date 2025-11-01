#pragma once

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {

/**
 * @brief Result structure for eigenvalue decomposition
 *
 * For symmetric matrices:
 *   - eigenvalues: real eigenvalues [N]
 *   - eigenvalues_imag: empty tensor
 *   - eigenvectors: real eigenvectors [N, N] (columns are eigenvectors)
 *
 * For general matrices:
 *   - eigenvalues: real parts of eigenvalues [N]
 *   - eigenvalues_imag: imaginary parts of eigenvalues [N]
 *   - eigenvectors: eigenvectors [N, N] in special LAPACK format
 *     If eigenvalues_imag[j] = 0: column j is real eigenvector
 *     If eigenvalues_imag[j] != 0: columns j and j+1 form complex conjugate pair
 */
struct EigResult {
    Tensor eigenvalues;      // Real part of eigenvalues [N]
    Tensor eigenvalues_imag; // Imaginary part (empty for symmetric case)
    Tensor eigenvectors;     // Eigenvectors [N, N]
};

/**
 * @brief Compute eigenvalues and eigenvectors of a symmetric real matrix
 *
 * Computes the eigenvalue decomposition of a real symmetric matrix A:
 *   A = V * diag(λ) * V^T
 *
 * where V is the matrix of eigenvectors (columns) and λ are eigenvalues.
 *
 * For symmetric matrices:
 * - All eigenvalues are real
 * - Eigenvectors are orthogonal
 * - More stable and faster than general eig()
 *
 * @param A Input square symmetric matrix [N, N]
 * @param compute_eigenvectors If true, compute eigenvectors; if false, eigenvalues only
 * @return EigResult containing eigenvalues and eigenvectors
 *
 * @throws std::runtime_error if A is not 2D, not square, not on CPU, or computation fails
 *
 * @note User is responsible for ensuring A is symmetric. No symmetry check is performed.
 * @note Requires OpenBLAS/LAPACK (USE_OPENBLAS=ON)
 *
 * Example:
 * @code
 *   Tensor A({3, 3}, {4, 1, 2,
 *                     1, 3, 1,
 *                     2, 1, 4});
 *   auto [vals, _, vecs] = cpptensor::eig_symmetric(A);
 *   // vals contains real eigenvalues
 *   // vecs contains orthogonal eigenvectors as columns
 * @endcode
 */
EigResult eig_symmetric(const Tensor& A, bool compute_eigenvectors = true);

/**
 * @brief Compute eigenvalues and eigenvectors of a general real matrix
 *
 * Computes the eigenvalue decomposition of a general real matrix A:
 *   A * V = V * diag(λ)
 *
 * where V is the matrix of right eigenvectors and λ are eigenvalues.
 *
 * For general matrices:
 * - Eigenvalues may be complex (stored as real + imaginary parts)
 * - Eigenvectors may be complex (stored in special LAPACK format)
 *
 * @param A Input square matrix [N, N]
 * @param compute_eigenvectors If true, compute eigenvectors; if false, eigenvalues only
 * @return EigResult containing eigenvalues and eigenvectors
 *
 * @throws std::runtime_error if A is not 2D, not square, not on CPU, or computation fails
 *
 * @note Complex eigenvectors are stored in LAPACK's packed format:
 *       - If eigenvalues_imag[j] == 0: column j of eigenvectors is real
 *       - If eigenvalues_imag[j] != 0: columns j and j+1 form complex conjugate pair
 *         (real part in column j, imaginary part in column j+1)
 * @note Requires OpenBLAS/LAPACK (USE_OPENBLAS=ON)
 *
 * Example:
 * @code
 *   Tensor A({3, 3}, {0, 1, 0,
 *                     0, 0, 1,
 *                     1, 0, 0});
 *   auto [vals_re, vals_im, vecs] = cpptensor::eig(A);
 *   // vals_re, vals_im contain real and imaginary parts
 *   // vecs contains eigenvectors in packed format
 * @endcode
 */
EigResult eig(const Tensor& A, bool compute_eigenvectors = true);

} // namespace cpptensor
