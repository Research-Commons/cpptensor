#include "cpptensor/ops/linearAlgebra/tensordot.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/utils/broadcastUtils.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace cpptensor {

    // Helpers
    static std::vector<int> normalize_axes(const std::vector<int>& axes, size_t rank) {
        std::vector<int> out = axes;
        for (auto& a : out) {
            if (a < 0) a += static_cast<int>(rank);
            if (a < 0 || a >= static_cast<int>(rank))
                throw std::runtime_error("tensordot: axis out of range");
        }
        // check duplicates
        auto tmp = out;
        std::sort(tmp.begin(), tmp.end());
        if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end())
            throw std::runtime_error("tensordot: duplicate axes");
        return out;
    }

    static std::vector<size_t> gather(const std::vector<size_t>& v, const std::vector<int>& idxs) {
        std::vector<size_t> r;
        r.reserve(idxs.size());
        for (int i : idxs) r.push_back(v[static_cast<size_t>(i)]);
        return r;
    }

    static std::vector<int> complement_axes(size_t rank, const std::vector<int>& axes) {
        std::vector<int> all(rank);
        std::iota(all.begin(), all.end(), 0);
        std::vector<int> ax = axes; std::sort(ax.begin(), ax.end());
        std::vector<int> rest; rest.reserve(rank - axes.size());
        size_t j = 0;
        for (size_t i = 0; i < rank; ++i) {
            if (j < ax.size() && static_cast<int>(i) == ax[j]) { ++j; }
            else rest.push_back(static_cast<int>(i));
        }
        return rest;
    }

    static Tensor permute_copy(const Tensor& T, const std::vector<int>& perm) {
        const auto& sh = T.shape();
        size_t rank = sh.size();
        if (perm.size() != rank) throw std::runtime_error("permute_copy: bad perm size");

        // compute output shape
        std::vector<size_t> out_shape(rank);
        for (size_t i = 0; i < rank; ++i) out_shape[i] = sh[static_cast<size_t>(perm[i])];

        // strides
        auto in_stride = T.stride();
        auto out_stride = compute_strides(out_shape);

        // map each output position to input offset
        size_t total = 1; for (auto d : out_shape) total *= d;
        std::vector<float> out_data(total);

        // // Memory usage warning for large tensors (>100MB per copy)
        // constexpr size_t WARN_THRESHOLD = 100 * 1024 * 1024 / sizeof(float); // ~25M elements
        // if (total > WARN_THRESHOLD) {
        //
        // }

        for (size_t pos = 0; pos < total; ++pos) {
            // decode pos into out multi-index
            size_t tmp = pos;
            size_t in_off = 0;
            for (size_t od = 0; od < rank; ++od) {
                size_t coord = tmp / out_stride[od];
                tmp = tmp % out_stride[od];
                size_t id = static_cast<size_t>(perm[od]);
                in_off += coord * in_stride[id];
            }
            out_data[pos] = T.data()[in_off];
        }

        return Tensor(out_shape, out_data, T.device_type());
    }

    // Collapse a list of dims to their product
    static size_t prod(const std::vector<size_t>& v) {
        size_t p = 1; for (auto x : v) p *= x; return p;
    }

    // Reshape as a view (no copy) - just wraps same data with new shape
    static Tensor reshape_view(const Tensor& T, const std::vector<size_t>& new_shape) {
        if (T.numel() != prod(new_shape))
            throw std::runtime_error("reshape_view: numel mismatch");
        // Create tensor with same data pointer, new shape
        return Tensor(new_shape, T.data(), T.device_type());
    }

    Tensor tensordot(const Tensor& A, const Tensor& B, int axes) {
        if (axes < 0) throw std::runtime_error("tensordot: axes must be non-negative");
        size_t ra = A.shape().size();
        size_t rb = B.shape().size();
        size_t k = static_cast<size_t>(axes);
        if (k > ra || k > rb)
            throw std::runtime_error("tensordot: axes exceeds rank");

        // last k of A with first k of B
        std::vector<int> axesA; axesA.reserve(k);
        std::vector<int> axesB; axesB.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            axesA.push_back(static_cast<int>(ra - k + i));
            axesB.push_back(static_cast<int>(i));
        }
        return tensordot(A, B, axesA, axesB);
    }

    Tensor tensordot(const Tensor& A, const Tensor& B, const std::vector<int>& axesA_in, const std::vector<int>& axesB_in) {
        if (A.device_type() != B.device_type())
            throw std::runtime_error("tensordot: device mismatch");

        const auto& Ash = A.shape();
        const auto& Bsh = B.shape();
        size_t ra = Ash.size();
        size_t rb = Bsh.size();

        if (axesA_in.size() != axesB_in.size())
            throw std::runtime_error("tensordot: axes lists must have same length");

        // Handle edge case: empty axes (outer product) - still works with existing logic
        // Result will have shape [*A.shape, *B.shape]

        auto axesA = normalize_axes(axesA_in, ra);
        auto axesB = normalize_axes(axesB_in, rb);

        // Validate contracted dims match
        for (size_t i = 0; i < axesA.size(); ++i) {
            size_t da = Ash[static_cast<size_t>(axesA[i])];
            size_t db = Bsh[static_cast<size_t>(axesB[i])];
            if (da != db) throw std::runtime_error("tensordot: contracted dimensions mismatch");
        }

        // Build permutations
        auto A_rest = complement_axes(ra, axesA);
        auto B_rest = complement_axes(rb, axesB);

        std::vector<int> permA = A_rest; permA.insert(permA.end(), axesA.begin(), axesA.end());
        std::vector<int> permB = axesB;  permB.insert(permB.end(), B_rest.begin(), B_rest.end());

        // Permute to [A_rest..., K...] and [K..., B_rest...]
        Tensor Ap = permute_copy(A, permA);
        Tensor Bp = permute_copy(B, permB);

        // Shapes
        auto Ap_sh = Ap.shape();
        auto Bp_sh = Bp.shape();

        std::vector<size_t> A_rest_sh(A_rest.size());
        for (size_t i = 0; i < A_rest.size(); ++i) A_rest_sh[i] = Ap_sh[i];
        std::vector<size_t> A_k_sh(Ap_sh.begin() + A_rest.size(), Ap_sh.end());

        std::vector<size_t> B_k_sh(axesB.size());
        for (size_t i = 0; i < axesB.size(); ++i) B_k_sh[i] = Bp_sh[i];
        std::vector<size_t> B_rest_sh(Bp_sh.begin() + axesB.size(), Bp_sh.end());

        size_t M = prod(A_rest_sh);
        size_t K = prod(A_k_sh); // == prod(B_k_sh)
        size_t N = prod(B_rest_sh);

        // Reshape to 2D and GEMM
        Tensor A2D = reshape_view(Ap, {M, K});
        Tensor B2D = reshape_view(Bp, {K, N});
        Tensor C2D = matmul(A2D, B2D); // uses existing kernels and batching

        // Reshape back to A_rest + B_rest
        std::vector<size_t> out_shape = A_rest_sh;
        out_shape.insert(out_shape.end(), B_rest_sh.begin(), B_rest_sh.end());
        Tensor Out = reshape_view(C2D, out_shape);
        return Out;
    }

} // namespace cpptensor
