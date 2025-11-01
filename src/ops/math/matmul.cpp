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

            // build 2D Tensor views via contiguous copies (req by gemm to be fast) for this batch and call gemm
            //move pointer to the start of this batch's A slice
            const float* A_ptr = A.data().data() + baseA;
            const float* B_ptr = B.data().data() + baseB;

            //allocate temporary flat memory for a M×K 2D matrix
            std::vector<float> A_block(M * K);
            //allocate [K×N] matrix for B's slice
            std::vector<float> B_block(K * N);
            std::copy(A_ptr, A_ptr + (M * K), A_block.begin());
            std::copy(B_ptr, B_ptr + (K * N), B_block.begin());

            Tensor A2D({M, K}, A_block, A.device_type());
            Tensor B2D({K, N}, B_block, B.device_type());
            Tensor C2D = gemm(A2D, B2D);

            // copy result back into the correct batch region of C
            float* C_ptr = C.data().data() + baseC;
            const auto& C2Ddata = C2D.data();
            std::copy(C2Ddata.begin(), C2Ddata.end(), C_ptr);
        }

        return C;
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
        // ===== Use OpenBLAS SGEMM =====
        //
        // SGEMM performs: C = alpha * A * B + beta * C
        // A: MxK
        // B: KxN
        // C: MxN
        //
        // We assume row-major layout for all tensors.
        //
        // CBLAS expects leading dimensions (lda, ldb, ldc) to match the
        // number of columns in each matrix for row-major ordering.

        //openblas_set_num_threads(1);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const float* Adata = A.data().data();
        const float* Bdata = B.data().data();
        float* Cdata = C.data().data();

        cblas_sgemm(
            CblasRowMajor,    // row-major storage
            CblasNoTrans,     // A not transposed
            CblasNoTrans,     // B not transposed
            static_cast<int>(M), // rows of A and C
            static_cast<int>(N), // cols of B and C
            static_cast<int>(K), // shared dimension
            alpha,             // scaling for A * B
            Adata, static_cast<int>(K), // A, leading dimension = K
            Bdata, static_cast<int>(N), // B, leading dimension = N
            beta,              // scaling for existing C
            Cdata, static_cast<int>(N)  // C, leading dimension = N
        );
    #else
        KernelRegistry::instance().getKernel(OpType::Matmul, A.device_type())(A, B, C);
    #endif
        return C;
    }

} // namespace cpptensor