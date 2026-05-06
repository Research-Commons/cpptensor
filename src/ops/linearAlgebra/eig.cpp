#include "cpptensor/ops/linearAlgebra/eig.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"
#include <algorithm>
#include <string>
#include <stdexcept>
#include <vector>

#ifdef USE_OPENBLAS
#include <lapacke.h>
#endif

namespace cpptensor {
namespace {

std::vector<float> copy_matrix_to_row_major_buffer(const Tensor& matrix) {
    const auto& shape = matrix.shape();
    const auto& stride = matrix.stride();
    const float* data_ptr = matrix.impl()->data_ptr();

    const size_t rows = shape[0];
    const size_t cols = shape[1];

    std::vector<float> buffer(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            buffer[row * cols + col] = data_ptr[row * stride[0] + col * stride[1]];
        }
    }
    return buffer;
}

} // namespace

EigResult eig_symmetric(const Tensor& A, bool compute_eigenvectors) {
    autograd::throw_if_requires_grad(A, "eig_symmetric");
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
    std::vector<float> a_copy = copy_matrix_to_row_major_buffer(A);

    std::vector<float> w(N);

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

    if (info < 0) {
        throw std::runtime_error("eig_symmetric: LAPACKE_ssyevd illegal argument at position " +
                                std::to_string(-info));
    } else if (info > 0) {
        throw std::runtime_error("eig_symmetric: LAPACKE_ssyevd failed to converge. " +
                                std::to_string(info) + " off-diagonal elements did not converge to zero");
    }

    EigResult result;
    result.eigenvalues = Tensor({static_cast<size_t>(N)}, w, DeviceType::CPU);
    if (compute_eigenvectors) {
        result.eigenvectors = Tensor({static_cast<size_t>(N), static_cast<size_t>(N)},
                                     a_copy, DeviceType::CPU);
    } else {
        result.eigenvectors = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
    }

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
    autograd::throw_if_requires_grad(A, "eig");
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
    std::vector<float> a_copy = copy_matrix_to_row_major_buffer(A);

    std::vector<float> wr(N);
    std::vector<float> wi(N);
    std::vector<float> vl(1);
    std::vector<float> vr(compute_eigenvectors ? N * N : 1);

    char jobvl = 'N';
    char jobvr = compute_eigenvectors ? 'V' : 'N';

    int info = LAPACKE_sgeev(
        LAPACK_ROW_MAJOR,
        jobvl,
        jobvr,
        N,
        a_copy.data(),
        N,
        wr.data(),
        wi.data(),
        vl.data(),
        1,
        vr.data(),
        compute_eigenvectors ? N : 1
    );

    if (info < 0) {
        throw std::runtime_error("eig: LAPACKE_sgeev illegal argument at position " +
                                std::to_string(-info));
    } else if (info > 0) {
        throw std::runtime_error("eig: LAPACKE_sgeev failed to converge. "
                                "The QR algorithm failed to compute all eigenvalues");
    }

    EigResult result;
    result.eigenvalues = Tensor({static_cast<size_t>(N)}, wr, DeviceType::CPU);
    result.eigenvalues_imag = Tensor({static_cast<size_t>(N)}, wi, DeviceType::CPU);
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
