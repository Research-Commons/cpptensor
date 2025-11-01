#include "cpptensor/ops/linearAlgebra/eig.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifdef USE_OPENBLAS
// Use LAPACKE (C interface) which supports row-major layout directly
#include <lapacke.h>
#endif

namespace cpptensor {

EigResult eig_symmetric(const Tensor& A, bool compute_eigenvectors) {
    // ===== Step 1: Validate Input =====
    if (A.device_type() != DeviceType::CPU) {
        throw std::runtime_error("eig_symmetric: only CPU tensors supported");
    }

    const auto& shape = A.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("eig_symmetric: input must be 2D matrix, got " +
                                std::to_string(shape.size()) + "D tensor");
    }

    int N = static_cast<int>(shape[0]);
    if (shape[0] != shape[1]) {
        throw std::runtime_error("eig_symmetric: matrix must be square, got [" +
                                std::to_string(shape[0]) + " × " + std::to_string(shape[1]) + "]");
    }

    if (N == 0) {
        throw std::runtime_error("eig_symmetric: matrix dimension cannot be zero");
    }

#ifdef USE_OPENBLAS
    // ===== Step 2: Prepare Input =====
    // LAPACKE_ssyevd destroys the input matrix and returns eigenvectors in it
    std::vector<float> a_copy = A.data();

    // ===== Step 3: Allocate Output =====
    std::vector<float> w(N);  // Eigenvalues

    // ===== Step 4: Compute Eigenvalues/Eigenvectors =====
    // LAPACKE_ssyevd: symmetric eigenvalue decomposition (divide-and-conquer)
    // This is 2-10× faster than ssyev for matrices larger than ~100×100
    // - 'V' = compute eigenvectors, 'N' = eigenvalues only
    // - 'U' = use upper triangle of matrix
    // - Matrix is overwritten with eigenvectors (column-wise)
    char jobz = compute_eigenvectors ? 'V' : 'N';

    int info = LAPACKE_ssyevd(
        LAPACK_ROW_MAJOR,     // Row-major layout (C-style)
        jobz,                 // 'V' for eigenvectors, 'N' for values only
        'U',                  // Use upper triangle
        N,                    // Matrix dimension
        a_copy.data(),        // Input/output matrix
        N,                    // Leading dimension
        w.data()              // Output eigenvalues
    );

    // ===== Step 5: Check for Errors =====
    if (info < 0) {
        throw std::runtime_error("eig_symmetric: LAPACKE_ssyevd illegal argument at position " +
                                std::to_string(-info));
    } else if (info > 0) {
        throw std::runtime_error("eig_symmetric: LAPACKE_ssyevd failed to converge. " +
                                std::to_string(info) + " off-diagonal elements did not converge to zero");
    }

    // ===== Step 6: Construct Result =====
    EigResult result;

    // Eigenvalues (always computed)
    result.eigenvalues = Tensor({static_cast<size_t>(N)}, w, DeviceType::CPU);

    // Eigenvectors (if requested)
    if (compute_eigenvectors) {
        result.eigenvectors = Tensor({static_cast<size_t>(N), static_cast<size_t>(N)},
                                     a_copy, DeviceType::CPU);
    } else {
        result.eigenvectors = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
    }

    // Imaginary part is empty for symmetric case (eigenvalues are always real)
    result.eigenvalues_imag = Tensor({0}, std::vector<float>{}, DeviceType::CPU);

    return result;

#else
    // ===== No LAPACK Available =====
    throw std::runtime_error(
        "eig_symmetric: requires OpenBLAS/LAPACK library.\n"
        "Please rebuild with: cmake -DUSE_OPENBLAS=ON ..\n"
        "Make sure OpenBLAS is installed on your system."
    );
#endif
}

EigResult eig(const Tensor& A, bool compute_eigenvectors) {
    // ===== Step 1: Validate Input =====
    if (A.device_type() != DeviceType::CPU) {
        throw std::runtime_error("eig: only CPU tensors supported");
    }

    const auto& shape = A.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("eig: input must be 2D matrix, got " +
                                std::to_string(shape.size()) + "D tensor");
    }

    int N = static_cast<int>(shape[0]);
    if (shape[0] != shape[1]) {
        throw std::runtime_error("eig: matrix must be square, got [" +
                                std::to_string(shape[0]) + " × " + std::to_string(shape[1]) + "]");
    }

    if (N == 0) {
        throw std::runtime_error("eig: matrix dimension cannot be zero");
    }

#ifdef USE_OPENBLAS
    // ===== Step 2: Prepare Input =====
    std::vector<float> a_copy = A.data();

    // ===== Step 3: Allocate Output =====
    std::vector<float> wr(N);   // Real part of eigenvalues
    std::vector<float> wi(N);   // Imaginary part of eigenvalues
    std::vector<float> vl(1);   // Left eigenvectors (not computed)
    std::vector<float> vr(compute_eigenvectors ? N * N : 1);  // Right eigenvectors

    // ===== Step 4: Compute Eigenvalues/Eigenvectors =====
    // LAPACKE_sgeev: general eigenvalue decomposition
    // - First 'N' = don't compute left eigenvectors
    // - Second 'V'/'N' = compute/don't compute right eigenvectors
    char jobvl = 'N';  // Don't compute left eigenvectors
    char jobvr = compute_eigenvectors ? 'V' : 'N';  // Compute right eigenvectors

    int info = LAPACKE_sgeev(
        LAPACK_ROW_MAJOR,     // Row-major layout
        jobvl,                // Left eigenvectors: 'N' = don't compute
        jobvr,                // Right eigenvectors: 'V' = compute, 'N' = don't
        N,                    // Matrix dimension
        a_copy.data(),        // Input matrix (destroyed on output)
        N,                    // Leading dimension of A
        wr.data(),            // Real part of eigenvalues
        wi.data(),            // Imaginary part of eigenvalues
        vl.data(),            // Left eigenvectors (not used)
        1,                    // Leading dimension of VL
        vr.data(),            // Right eigenvectors
        compute_eigenvectors ? N : 1  // Leading dimension of VR
    );

    // ===== Step 5: Check for Errors =====
    if (info < 0) {
        throw std::runtime_error("eig: LAPACKE_sgeev illegal argument at position " +
                                std::to_string(-info));
    } else if (info > 0) {
        throw std::runtime_error("eig: LAPACKE_sgeev failed to converge. "
                                "The QR algorithm failed to compute all eigenvalues");
    }

    // ===== Step 6: Construct Result =====
    EigResult result;

    // Eigenvalues (always computed)
    result.eigenvalues = Tensor({static_cast<size_t>(N)}, wr, DeviceType::CPU);
    result.eigenvalues_imag = Tensor({static_cast<size_t>(N)}, wi, DeviceType::CPU);

    // Eigenvectors (if requested)
    if (compute_eigenvectors) {
        result.eigenvectors = Tensor({static_cast<size_t>(N), static_cast<size_t>(N)},
                                     vr, DeviceType::CPU);
    } else {
        result.eigenvectors = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
    }

    return result;

#else
    // ===== No LAPACK Available =====
    throw std::runtime_error(
        "eig: requires OpenBLAS/LAPACK library.\n"
        "Please rebuild with: cmake -DUSE_OPENBLAS=ON ..\n"
        "Make sure OpenBLAS is installed on your system."
    );
#endif
}

} // namespace cpptensor
