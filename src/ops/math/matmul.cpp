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
        /**
         * @brief Check if tensor represents a transposed view
         *
         * Detects if a 2D tensor has column-major stride pattern (transposed).
         * For row-major: stride[0] > stride[1] (e.g., [3, 1] for [2×3])
         * For col-major: stride[0] < stride[1] (e.g., [1, 2] for [2×3])
         */
        bool is_transposed(const Tensor& T) {
            if (T.ndim() != 2) return false;
            auto st = T.stride();
            // Transposed: stride[0] < stride[1] (column-major)
            return st[0] < st[1];
        }

        /**
         * @brief Check if a batch slice is contiguous in memory
         *
         * Determines whether extracting a batch slice requires copying or
         * can be done with a zero-copy view.
         */
        bool is_batch_slice_contiguous(const Tensor& T) {
            auto st = T.stride();
            auto sh = T.shape();
            size_t ndim = sh.size();

            if (ndim < 2) return false;

            // Check last two dims are contiguous (row-major)
            if (st[ndim-1] != 1) return false;
            if (st[ndim-2] != sh[ndim-1]) return false;

            return true;
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
        const size_t offA = (LO >= LA) ? (LO - LA) : 0;
        const size_t offB = (LO >= LB) ? (LO - LB) : 0;

        //how many matmuls we will have to do. product of all batch sizes
        size_t batch_count = 1;
        for (auto d : out_batch) batch_count *= d;

        // Helper lambda: compute base offset into a tensor for given out_batch index
        // basically where does a slice start in 1D memory
        auto compute_base_offset = [&](const std::vector<size_t>& batch_index,
                                       const std::vector<size_t>& t_batch_shape,
                                       const std::vector<size_t>& t_stride,
                                       size_t t_batch_rank,
                                       size_t align_off) -> size_t {
            size_t off = 0;
            // Map output batch index to tensor's batch index (broadcast aware)
            for (size_t d = 0; d < t_batch_rank; ++d) {
                const size_t out_d = align_off + d; // aligned to the right
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

            Tensor A2D, B2D;

            // Check if we can create zero-copy views
            bool A_is_contiguous = is_batch_slice_contiguous(A);
            bool B_is_contiguous = is_batch_slice_contiguous(B);
            const auto& A_impl = *A.impl();
            const auto& B_impl = *B.impl();

            if (A_is_contiguous) {
                // Zero-copy view using raw pointer
                float* A_ptr = const_cast<float*>(A_impl.data_ptr() + baseA);
                A2D = Tensor::from_ptr({M, K}, A_ptr, A.impl(), A.device_type());
            } else {
                // Need to copy (non-contiguous batch slice)
                const float* A_ptr = A_impl.data_ptr() + baseA;
                std::vector<float> A_block(M * K);
                std::copy(A_ptr, A_ptr + (M * K), A_block.begin());
                A2D = Tensor({M, K}, A_block, A.device_type());
            }

            if (B_is_contiguous) {
                // Zero-copy view using raw pointer
                float* B_ptr = const_cast<float*>(B_impl.data_ptr() + baseB);
                B2D = Tensor::from_ptr({K, N}, B_ptr, B.impl(), B.device_type());
            } else {
                // Need to copy (non-contiguous batch slice)
                const float* B_ptr = B_impl.data_ptr() + baseB;
                std::vector<float> B_block(K * N);
                std::copy(B_ptr, B_ptr + (K * N), B_block.begin());
                B2D = Tensor({K, N}, B_block, B.device_type());
            }

            // Call gemm on the 2D slices
            Tensor C2D = gemm(A2D, B2D);

            // copy result back into the correct batch region of C
            float* C_ptr = C.data().data() + baseC;
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
        // ===== Use OpenBLAS SGEMV =====
        //
        // SGEMV performs: y = alpha * A * x + beta * y
        //
        // Parameters:
        // - Layout: CblasRowMajor (row-major storage)
        // - TransA: CblasNoTrans (use A as-is, not transposed)
        // - M, N: matrix dimensions
        // - alpha: scaling factor (1.0 for y = A*x)
        // - A, lda: matrix and leading dimension
        // - x, incx: input vector and stride
        // - beta: scaling for existing y (0.0 to overwrite)
        // - y, incy: output vector and stride

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const float* Adata = A.impl()->data_ptr();
        const float* xdata = x.impl()->data_ptr();
        float* ydata = y.data().data();

        cblas_sgemv(
            CblasRowMajor,           // row-major storage
            CblasNoTrans,            // don't transpose A
            static_cast<int>(M),     // rows of A
            static_cast<int>(N),     // cols of A
            alpha,                   // scaling factor for A*x
            Adata,                   // matrix A
            static_cast<int>(N),     // leading dimension (lda = N for row-major)
            xdata,                   // vector x
            1,                       // stride in x (contiguous)
            beta,                    // scaling factor for y (0 = overwrite)
            ydata,                   // output vector y
            1                        // stride in y (contiguous)
        );
    #else
        // Fallback: use CPU kernel with SIMD optimization
        // The gemvKernel will automatically dispatch to AVX512/AVX2 if available
        //TODO : switch to proper kernel dispatcher
        cpptensor::CPU::gemvKernel(A, x, y);
    #endif

        return y;
    }

    Tensor gemm(const Tensor& A, const Tensor& B) {
        // OPTIMIZATION: Detect transpose to use BLAS flags instead of copying
        bool A_trans = is_transposed(A);
        bool B_trans = is_transposed(B);

        // Get actual dimensions (accounting for transpose)
        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t KB = B.shape()[0];
        size_t N = B.shape()[1];

        if (K != KB) {
            throw std::runtime_error("matmul: dimension mismatch (A.cols != B.rows)");
        }

        Tensor C = Tensor::full({M, N}, 0.0f, A.device_type());

    #ifdef USE_OPENBLAS
        // ===== Use OpenBLAS SGEMM with transpose detection =====
        //
        // SGEMM performs: C = alpha * op(A) * op(B) + beta * C
        // where op(X) = X or X^T depending on transpose flags
        //
        // If tensor is transposed (column-major strides), we can use
        // CblasTrans flag instead of forcing a contiguous() copy.

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const float* Adata = A.impl()->data_ptr();
        const float* Bdata = B.impl()->data_ptr();
        float* Cdata = C.data().data();

        // Set transpose flags based on stride pattern
        CBLAS_TRANSPOSE transA = A_trans ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE transB = B_trans ? CblasTrans : CblasNoTrans;

        // Leading dimensions depend on actual memory layout
        // For transposed matrices, leading dim is the other dimension
        int lda = A_trans ? static_cast<int>(M) : static_cast<int>(K);
        int ldb = B_trans ? static_cast<int>(K) : static_cast<int>(N);
        int ldc = static_cast<int>(N);

        cblas_sgemm(
            CblasRowMajor,    // row-major storage
            transA,           // Use detected transpose flag for A
            transB,           // Use detected transpose flag for B
            static_cast<int>(M), // rows of op(A) and C
            static_cast<int>(N), // cols of op(B) and C
            static_cast<int>(K), // shared dimension
            alpha,             // scaling for op(A) * op(B)
            Adata, lda,        // A with correct leading dimension
            Bdata, ldb,        // B with correct leading dimension
            beta,              // scaling for existing C
            Cdata, ldc         // C, leading dimension = N
        );
    #else
        KernelRegistry::instance().getKernel(OpType::Matmul, A.device_type())(A, B, C);
    #endif
        return C;
    }

} // namespace cpptensor
