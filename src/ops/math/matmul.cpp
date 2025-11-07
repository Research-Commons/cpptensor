#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/utils/broadcastUtils.hpp"

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

        if (Ash.size() < 2 || Bsh.size() < 2)
            throw std::runtime_error("matmul: tensors must have at least 2 dims");

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

            if (A_is_contiguous) {
                // Zero-copy view using raw pointer
                float* A_ptr = const_cast<float*>(A.data().data() + baseA);
                A2D = Tensor::from_ptr({M, K}, A_ptr, A.impl(), A.device_type());
            } else {
                // Need to copy (non-contiguous batch slice)
                const float* A_ptr = A.data().data() + baseA;
                std::vector<float> A_block(M * K);
                std::copy(A_ptr, A_ptr + (M * K), A_block.begin());
                A2D = Tensor({M, K}, A_block, A.device_type());
            }

            if (B_is_contiguous) {
                // Zero-copy view using raw pointer
                float* B_ptr = const_cast<float*>(B.data().data() + baseB);
                B2D = Tensor::from_ptr({K, N}, B_ptr, B.impl(), B.device_type());
            } else {
                // Need to copy (non-contiguous batch slice)
                const float* B_ptr = B.data().data() + baseB;
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

        const float* Adata = A.data().data();
        const float* Bdata = B.data().data();
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