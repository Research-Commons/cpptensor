#include "cpptensor/ops/linearAlgebra/tensordot.hpp"
#include "cpptensor/ops/math/matmul.hpp"
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

    // Collapse a list of dims to their product
    static size_t prod(const std::vector<size_t>& v) {
        size_t p = 1; for (auto x : v) p *= x; return p;
    }

    static bool is_prefix_axes(const std::vector<int>& axes) {
        for (size_t i = 0; i < axes.size(); ++i) {
            if (axes[i] != static_cast<int>(i)) {
                return false;
            }
        }
        return true;
    }

    static bool is_suffix_axes(const std::vector<int>& axes, size_t rank) {
        const size_t suffix_start = rank - axes.size();
        for (size_t i = 0; i < axes.size(); ++i) {
            if (axes[i] != static_cast<int>(suffix_start + i)) {
                return false;
            }
        }
        return true;
    }

    enum class MatrixPrepKind {
        DirectView,
        TransposedView,
        Materialize
    };

    struct MatrixPrepPlan {
        MatrixPrepKind kind;
        std::vector<int> permutation;
        size_t rows;
        size_t cols;
    };

    static MatrixPrepPlan plan_left_matrix(const std::vector<int>& axesA,
                                           const std::vector<int>& A_rest,
                                           size_t rank,
                                           size_t M,
                                           size_t K) {
        if (is_suffix_axes(axesA, rank)) {
            return {MatrixPrepKind::DirectView, {}, M, K};
        }
        if (is_prefix_axes(axesA)) {
            return {MatrixPrepKind::TransposedView, {}, M, K};
        }

        std::vector<int> perm = A_rest;
        perm.insert(perm.end(), axesA.begin(), axesA.end());
        return {MatrixPrepKind::Materialize, std::move(perm), M, K};
    }

    static MatrixPrepPlan plan_right_matrix(const std::vector<int>& axesB,
                                            const std::vector<int>& B_rest,
                                            size_t rank,
                                            size_t K,
                                            size_t N) {
        if (is_prefix_axes(axesB)) {
            return {MatrixPrepKind::DirectView, {}, K, N};
        }
        if (is_suffix_axes(axesB, rank)) {
            return {MatrixPrepKind::TransposedView, {}, K, N};
        }

        std::vector<int> perm = axesB;
        perm.insert(perm.end(), B_rest.begin(), B_rest.end());
        return {MatrixPrepKind::Materialize, std::move(perm), K, N};
    }

    static Tensor prepare_matrix(const Tensor& input, const MatrixPrepPlan& plan) {
        if (!input.is_contiguous() && plan.kind != MatrixPrepKind::Materialize) {
            return prepare_matrix(input.contiguous(), plan);
        }

        switch (plan.kind) {
            case MatrixPrepKind::DirectView:
                return input.view({plan.rows, plan.cols});
            case MatrixPrepKind::TransposedView:
                return input.view({plan.cols, plan.rows}).transpose();
            case MatrixPrepKind::Materialize:
                return input.permute(plan.permutation).contiguous().view({plan.rows, plan.cols});
        }

        throw std::runtime_error("tensordot: unknown matrix preparation plan");
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

        std::vector<size_t> A_rest_sh;
        A_rest_sh.reserve(A_rest.size());
        for (int axis : A_rest) {
            A_rest_sh.push_back(Ash[static_cast<size_t>(axis)]);
        }

        std::vector<size_t> A_k_sh;
        A_k_sh.reserve(axesA.size());
        for (int axis : axesA) {
            A_k_sh.push_back(Ash[static_cast<size_t>(axis)]);
        }

        std::vector<size_t> B_rest_sh;
        B_rest_sh.reserve(B_rest.size());
        for (int axis : B_rest) {
            B_rest_sh.push_back(Bsh[static_cast<size_t>(axis)]);
        }

        std::vector<size_t> B_k_sh;
        B_k_sh.reserve(axesB.size());
        for (int axis : axesB) {
            B_k_sh.push_back(Bsh[static_cast<size_t>(axis)]);
        }

        size_t M = prod(A_rest_sh);
        size_t K = prod(A_k_sh); // == prod(B_k_sh)
        size_t N = prod(B_rest_sh);

        MatrixPrepPlan left_plan = plan_left_matrix(axesA, A_rest, ra, M, K);
        MatrixPrepPlan right_plan = plan_right_matrix(axesB, B_rest, rb, K, N);

        Tensor A2D = prepare_matrix(A, left_plan);
        Tensor B2D = prepare_matrix(B, right_plan);
        Tensor C2D = matmul(A2D, B2D); // uses existing kernels and batching

        // Reshape back to A_rest + B_rest
        std::vector<size_t> out_shape = A_rest_sh;
        out_shape.insert(out_shape.end(), B_rest_sh.begin(), B_rest_sh.end());
        return C2D.view(out_shape);
    }

} // namespace cpptensor
