#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/utils/broadcastUtils.hpp"
#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/backend/cpu_backend.h"

#include <stdexcept>
#include <vector>
#include <numeric>

#include "ops/helperOps.hpp"
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

namespace cpptensor {

    // =============== Helper Functions for Optimization ===============

    namespace {
        float* raw_data(Tensor& tensor) {
            return tensor.impl()->data_ptr();
        }

        Tensor compact_blas_input(const Tensor& tensor) {
            return tensor.is_contiguous() ? tensor : tensor.contiguous();
        }

        bool is_simple_transposed_view(const Tensor& tensor) {
            if (tensor.ndim() != 2 || tensor.is_contiguous()) {
                return false;
            }

            const auto& shape = tensor.shape();
            const auto& stride = tensor.stride();
            return stride.size() == 2 && stride[0] == 1 && stride[1] == shape[0];
        }

        Tensor make_batched_matrix_view(const Tensor& tensor, size_t base_offset) {
            const auto shape = tensor.shape();
            const auto stride = tensor.stride();
            const size_t rank = shape.size();
            const size_t rows = shape[rank - 2];
            const size_t cols = shape[rank - 1];

            if (stride[rank - 1] == 1 && stride[rank - 2] == cols) {
                float* data_ptr = const_cast<float*>(tensor.impl()->data_ptr() + base_offset);
                return Tensor::from_ptr({rows, cols}, data_ptr, tensor.impl(), tensor.device_type());
            }

            const float* src = tensor.impl()->data_ptr();
            std::vector<float> compact(rows * cols);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    compact[row * cols + col] =
                        src[base_offset + row * stride[rank - 2] + col * stride[rank - 1]];
                }
            }

            return Tensor({rows, cols}, compact, tensor.device_type());
        }
    }

    Tensor matmul(const Tensor& A, const Tensor& B) {
        if (A.device_type() != B.device_type()) {
            throw std::runtime_error("matmul: device mismatch");
        }

        const auto& Ash = A.shape();
        const auto& Bsh = B.shape();

        const size_t A_ndim = Ash.size();
        const size_t B_ndim = Bsh.size();

        // Case 1: 1D × 1D → dot product
        if (A_ndim == 1 && B_ndim == 1) {
            return dot(A, B);
        }

        //Case 2: 2D × 1D → matrix-vector product
        if (A_ndim == 2 && B_ndim == 1) {
            return gemv(A, B);
        }

        //Case 3: 1D × 2D → vector-matrix product
        if (A_ndim == 1 && B_ndim == 2) {
            // PyTorch behavior: treats 1D as row vector [1, N]
            // Result: [1, N] × [N, M] → [1, M], then squeeze to [M]
            //
            // Implementation: Use gemm() with view() (RECOMMENDED)
            // - view() operations are zero-copy (no memory allocation)
            // - gemm() with [1,N] shape is well-optimized by BLAS
            // - Keeps data contiguous in row-major order

            const size_t N = Ash[0];  // Vector length
            const size_t M = Bsh[1];  // Output size

            if (N != Bsh[0]) {
                throw std::runtime_error("matmul: 1D×2D dimension mismatch (vec.size != matrix.rows)");
            }

            // OPTION 1: Use gemm with view (CURRENT - RECOMMENDED)
            // Reshape A from [N] to [1, N] for gemm (zero-copy view)
            Tensor A_reshaped = A.view({1, N});
            Tensor result = gemm(A_reshaped, B);  // [1, N] × [N, M] → [1, M]

            // Squeeze result from [1, M] to [M] (zero-copy view)
            return result.view({M});

            // OPTION 2: Use gemv with transpose (COMMENTED OUT - FOR TESTING)
            // Mathematical: y = Bᵀ * x where x=[N], Bᵀ=[M,N], y=[M]
            // WARNING: transpose() creates non-contiguous view with column-major strides
            // BLAS functions assume row-major, so you MUST call .contiguous() first
            // This adds memory copy overhead, making it slower than Option 1
            //
            // Tensor B_T = B.transpose().contiguous();  // ⚠️ .contiguous() required!
            // return gemv(B_T, A);                      // [M, N] × [N] → [M]
            //
            // Without .contiguous(), gemv() will read wrong memory locations because:
            // - transpose() only swaps strides: [3,1] → [1,3] (column-major)
            // - gemv() assumes row-major with lda=N
            // - Results in incorrect output (e.g., [5,11,17] instead of [9,12,15])
        }

        //Case 4: 2D × 2D and higher-dimensional cases
        if (Ash.size() < 2 || Bsh.size() < 2)
            throw std::runtime_error("matmul: tensors must have at least 1 dim (already handled above)");

        //if tensor is 2D dont waste computation
        if (Ash.size() == 2 && Bsh.size() == 2) {
            return gemm(A, B);
        }

        const size_t M  = Ash[Ash.size() - 2]; // no of row in a
        const size_t K  = Ash[Ash.size() - 1]; // no of col in a
        const size_t KB = Bsh[Bsh.size() - 2]; // row in b
        const size_t N  = Bsh[Bsh.size() - 1]; // col in b

        //last dim of first tensor not equal to second last dim of second tensor
        if (K != KB)
            throw std::runtime_error("matmul: inner dims mismatch (A[...,-1] != B[...,-2])");

        // batch dims (all but last two). req for broadcast checking
        std::vector<size_t> Abatch(Ash.begin(), Ash.end() - 2);
        std::vector<size_t> Bbatch(Bsh.begin(), Bsh.end() - 2);

        // compute broadcasting for the two batches
        std::vector<size_t> out_batch = computeBroadcastShape(Abatch, Bbatch);

        // create a tensor with broadcasted values and last 2 matrices filled
        std::vector<size_t> out_shape = out_batch;
        out_shape.push_back(M);
        out_shape.push_back(N);
        Tensor C = Tensor::full(out_shape, 0.0f, A.device_type());

        // no work to do, result is all zeros so skip
        if (M == 0 || N == 0 || K == 0) {
            return C;
        }

        //stride is used later to help us calculate how far a slice is from the batch
        const auto& Astride = A.stride();
        const auto& Bstride = B.stride();
        const auto& Cstride = C.stride();

        const size_t LA = Abatch.size(); // A batch rank
        const size_t LB = Bbatch.size(); // B batch rank
        const size_t LO = out_batch.size(); // output batch rank

        // calc the offsets when aligning A/B batches to output batch
        const size_t offA = LO - LA; // where A batch dims begin inside out_batch
        const size_t offB = LO - LB; // where B batch dims begin inside out_batch

        // total num of matmul calls based on batch dims in output
        size_t batch_count = 1;
        for (size_t d : out_batch) batch_count *= d;

        //Helper Lambda that maps a multi-dimensional batch index to a linear memory offset.
        auto compute_base_offset = [](const std::vector<size_t>& batch_index, // full output batch index
                                      const std::vector<size_t>& t_batch_shape, // tensor A/B batch shape
                                      const std::vector<size_t>& t_stride, // tensor A/B stride
                                      size_t rank_t_batch, // number of batch dims in A/B
                                      size_t align_offset // how many leading output dims to skip
                                      ) -> size_t {
            // if no batch dims, the only slice is the matrix at offset 0
            if (rank_t_batch == 0) return 0;

            size_t off = 0;
            for (size_t d = 0; d < rank_t_batch; ++d) {
                const size_t out_d = align_offset + d; // corresponding dim in output batch
                const size_t dim   = t_batch_shape[d];
                const size_t idx   = (dim == 1) ? 0 : batch_index[out_d];
                off += idx * t_stride[d]; // stride[d] corresponds to batch dim d
            }
            return off;
        };

        // for each batch_index in batch_count:
        // find which slice of A to pick
        // find which slice of B to pick
        // GEMM that pair
        // write result into the correct slice of C

        //for each batch
        for (size_t b = 0; b < batch_count; ++b) {
            // expand flat index b into multi-index over out_batch dims. Basically, find the slice from b

            //batch_index tells which small matrix from A and B to mul in this turn
            std::vector<size_t> batch_index(LO, 0);
            size_t tmp = b;
            for (int i = static_cast<int>(LO) - 1; i >= 0; --i) {
                const size_t dim = out_batch[(size_t)i];
                batch_index[(size_t)i] = (dim == 0) ? 0 : (tmp % dim);
                tmp = (dim == 0) ? tmp : (tmp / dim);
            }

            // compute base offsets into A/B/C for this batch (last two dims start at 0)
            //where in A's memory does this slice start?
            const size_t baseA = compute_base_offset(batch_index, Abatch, Astride, LA, offA);
            const size_t baseB = compute_base_offset(batch_index, Bbatch, Bstride, LB, offB);
            //where in C's memory should I write the result?
            const size_t baseC = compute_base_offset(batch_index, out_batch, Cstride, LO, 0);

            Tensor A2D = make_batched_matrix_view(A, baseA);
            Tensor B2D = make_batched_matrix_view(B, baseB);

            // Call gemm on the 2D slices
            Tensor C2D = gemm(A2D, B2D);

            // copy result back into the correct batch region of C
            float* C_ptr = raw_data(C) + baseC;
            const auto& C2Ddata = C2D.data();
            std::copy(C2Ddata.begin(), C2Ddata.end(), C_ptr);
        }

        return C;
    }

    Tensor gemv(const Tensor& A, const Tensor& x) {
        // Matrix-vector product: y = A * x
        // A: [M, N] matrix
        // x: [N] vector
        // Returns: [M] vector

        if (A.device_type() != x.device_type()) {
            throw std::runtime_error("gemv: device mismatch");
        }

        const auto& Ash = A.shape();
        const auto& xsh = x.shape();

        if (Ash.size() != 2) {
            throw std::runtime_error("gemv: A must be a 2D matrix");
        }
        if (xsh.size() != 1) {
            throw std::runtime_error("gemv: x must be a 1D vector");
        }

        const size_t M = Ash[0];  // rows of A
        const size_t N = Ash[1];  // cols of A
        const size_t xN = xsh[0]; // length of x

        if (N != xN) {
            throw std::runtime_error("gemv: dimension mismatch (A.cols != x.size)");
        }

        // Create output vector
        Tensor y = Tensor::full({M}, 0.0f, A.device_type());

    #ifdef USE_OPENBLAS
        const bool A_is_simple_transpose = is_simple_transposed_view(A);
        Tensor A_blas = (!A.is_contiguous() && !A_is_simple_transpose) ? A.contiguous() : A;
        Tensor x_blas = compact_blas_input(x);

        // ===== Use OpenBLAS SGEMV =====
        //
        // SGEMV performs: y = alpha * A * x + beta * y
        //
        // Parameters:
        // - Layout: CblasRowMajor (row-major storage)
        // - TransA: depends on whether A is a transposed logical view
        // - M, N: physical matrix dimensions passed to BLAS
        // - alpha: scaling factor (1.0 for y = A*x)
        // - A, lda: matrix and leading dimension
        // - x, incx: input vector and stride
        // - beta: scaling for existing y (0.0 to overwrite)
        // - y, incy: output vector and stride

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const int rows = static_cast<int>(A_is_simple_transpose ? N : M);
        const int cols = static_cast<int>(A_is_simple_transpose ? M : N);
        const CBLAS_TRANSPOSE transA = A_is_simple_transpose ? CblasTrans : CblasNoTrans;
        const int lda = A_is_simple_transpose ? static_cast<int>(M) : static_cast<int>(N);

        const float* Adata = raw_data(A_blas);
        const float* xdata = raw_data(x_blas);
        float* ydata = raw_data(y);

        cblas_sgemv(
            CblasRowMajor,           // row-major storage
            transA,                  // logical transpose handling for view-backed matrices
            rows,                    // physical rows of the BLAS input matrix
            cols,                    // physical cols of the BLAS input matrix
            alpha,                   // scaling factor for A*x
            Adata,                   // matrix A
            lda,                     // leading dimension of the physical matrix
            xdata,                   // vector x
            1,                       // stride in x (contiguous)
            beta,                    // scaling factor for y (0 = overwrite)
            ydata,                   // output vector y
            1                        // stride in y (contiguous)
        );
    #else
        // Fallback: use the generic CPU GEMV kernel.
        // Reading through Tensor::data() keeps sliced and transposed inputs
        // logically correct even when BLAS is unavailable.
        cpptensor::CPU::gemvKernel(A, x, y);
    #endif

        return y;
    }

    Tensor gemm(const Tensor& A, const Tensor& B) {
        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t KB = B.shape()[0];
        size_t N = B.shape()[1];

        if (K != KB) {
            throw std::runtime_error("matmul: dimension mismatch (A.cols != B.rows)");
        }

        Tensor C = Tensor::full({M, N}, 0.0f, A.device_type());

    #ifdef USE_OPENBLAS
        const bool A_is_simple_transpose = is_simple_transposed_view(A);
        const bool B_is_simple_transpose = is_simple_transposed_view(B);

        Tensor A_blas = (!A.is_contiguous() && !A_is_simple_transpose) ? A.contiguous() : A;
        Tensor B_blas = (!B.is_contiguous() && !B_is_simple_transpose) ? B.contiguous() : B;

        // ===== Use OpenBLAS SGEMM on logical inputs =====
        //
        // SGEMM performs: C = alpha * op(A) * op(B) + beta * C
        // where op(X) = X or X^T depending on transpose flags.
        // Simple transposed views can be represented with BLAS transpose flags;
        // all other non-contiguous layouts are materialized logically first.

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const float* Adata = raw_data(A_blas);
        const float* Bdata = raw_data(B_blas);
        float* Cdata = raw_data(C);

        const CBLAS_TRANSPOSE transA = A_is_simple_transpose ? CblasTrans : CblasNoTrans;
        const CBLAS_TRANSPOSE transB = B_is_simple_transpose ? CblasTrans : CblasNoTrans;
        const int lda = A_is_simple_transpose ? static_cast<int>(M) : static_cast<int>(K);
        const int ldb = B_is_simple_transpose ? static_cast<int>(K) : static_cast<int>(N);
        const int ldc = static_cast<int>(N);

        cblas_sgemm(
            CblasRowMajor,       // row-major storage
            transA,              // logical transpose handling for A views
            transB,              // logical transpose handling for B views
            static_cast<int>(M), // rows of op(A) and C
            static_cast<int>(N), // cols of op(B) and C
            static_cast<int>(K), // shared dimension
            alpha,               // scaling for op(A) * op(B)
            Adata, lda,          // A with the correct physical leading dimension
            Bdata, ldb,          // B with the correct physical leading dimension
            beta,                // scaling for existing C
            Cdata, ldc           // C, leading dimension = N
        );
    #else
        KernelRegistry::instance().getKernel(OpType::Matmul, A.device_type())(A, B, C);
    #endif
        return C;
    }

} // namespace cpptensor
