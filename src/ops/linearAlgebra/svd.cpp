#include "cpptensor/ops/linearAlgebra/svd.hpp"
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

SVDResult svd(const Tensor& A, bool full_matrices, bool compute_uv) {
    autograd::throw_if_requires_grad(A, "svd");
    if (A.device_type() != DeviceType::CPU) {
        throw std::runtime_error("svd: only CPU tensors supported");
    }

    const auto& shape = A.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("svd: input must be 2D matrix, got " +
                                 std::to_string(shape.size()) + "D tensor");
    }

    const int M = static_cast<int>(shape[0]);
    const int N = static_cast<int>(shape[1]);
    const int K = std::min(M, N);

    if (M == 0 || N == 0) {
        throw std::runtime_error("svd: matrix dimensions cannot be zero");
    }

#ifdef USE_OPENBLAS
    std::vector<float> a_copy = copy_matrix_to_row_major_buffer(A);
    std::vector<float> singular_values(K);

    const char jobz = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';

    const int result_u_rows = compute_uv ? M : 0;
    const int result_u_cols = compute_uv ? (full_matrices ? M : K) : 0;
    const int result_vt_rows = compute_uv ? (full_matrices ? N : K) : 0;
    const int result_vt_cols = compute_uv ? N : 0;

    const int lapack_u_cols = compute_uv ? result_u_cols : 1;
    const int lapack_vt_cols = compute_uv ? result_vt_cols : 1;

    std::vector<float> u_data(compute_uv ? static_cast<size_t>(result_u_rows * result_u_cols) : 1, 0.0f);
    std::vector<float> vt_data(compute_uv ? static_cast<size_t>(result_vt_rows * result_vt_cols) : 1, 0.0f);

    const int info = LAPACKE_sgesdd(
        LAPACK_ROW_MAJOR,
        jobz,
        M,
        N,
        a_copy.data(),
        N,
        singular_values.data(),
        u_data.data(),
        lapack_u_cols,
        vt_data.data(),
        lapack_vt_cols
    );

    if (info < 0) {
        throw std::runtime_error("svd: LAPACKE_sgesdd illegal argument at position " +
                                 std::to_string(-info));
    }
    if (info > 0) {
        throw std::runtime_error("svd: LAPACKE_sgesdd failed to converge. "
                                 "The divide-and-conquer bidiagonal solver did not converge");
    }

    SVDResult result;
    result.S = Tensor({static_cast<size_t>(K)}, singular_values, DeviceType::CPU);

    if (compute_uv) {
        result.U = Tensor({static_cast<size_t>(result_u_rows), static_cast<size_t>(result_u_cols)},
                          u_data, DeviceType::CPU);
        result.Vt = Tensor({static_cast<size_t>(result_vt_rows), static_cast<size_t>(result_vt_cols)},
                           vt_data, DeviceType::CPU);
    } else {
        result.U = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
        result.Vt = Tensor({0, 0}, std::vector<float>{}, DeviceType::CPU);
    }

    return result;
#else
    throw std::runtime_error(
        "svd: requires OpenBLAS/LAPACK library.\n"
        "Please rebuild with: cmake -DUSE_OPENBLAS=ON ..\n"
        "Make sure OpenBLAS is installed on your system."
    );
#endif
}

} // namespace cpptensor
