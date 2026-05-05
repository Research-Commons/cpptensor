#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cpptensor/ops/comparison/eq.hpp"
#include "cpptensor/ops/comparison/ge.hpp"
#include "cpptensor/ops/comparison/gt.hpp"
#include "cpptensor/ops/comparison/le.hpp"
#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/ops/comparison/ne.hpp"
#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/ops/manipulation/stack.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/backend/backend_loader.hpp"
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/utils/broadcastUtils.hpp"

#include <vector>

using Catch::Approx;

namespace {

void require_shape(const cpptensor::Tensor& tensor, std::vector<size_t> expected) {
    REQUIRE(tensor.shape() == expected);
}

void require_data(const cpptensor::Tensor& tensor, const std::vector<float>& expected) {
    REQUIRE(tensor.data().size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(tensor.data()[i] == Approx(expected[i]));
    }
}

} // namespace

TEST_CASE("cat concatenates tensors along existing dimensions", "[manipulation][cat]") {
    cpptensor::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor b({2, 3}, {7, 8, 9, 10, 11, 12});

    auto dim0 = cpptensor::cat({a, b}, 0);
    require_shape(dim0, {4, 3});
    require_data(dim0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    auto dim1 = cpptensor::cat({a, b}, 1);
    require_shape(dim1, {2, 6});
    require_data(dim1, {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12});

    auto neg_dim = cpptensor::cat({a, b}, -1);
    require_shape(neg_dim, {2, 6});
    require_data(neg_dim, dim1.data());
}

TEST_CASE("stack inserts a new dimension", "[manipulation][stack]") {
    cpptensor::Tensor a({2, 2}, {1, 2, 3, 4});
    cpptensor::Tensor b({2, 2}, {5, 6, 7, 8});

    auto dim0 = cpptensor::stack({a, b}, 0);
    require_shape(dim0, {2, 2, 2});
    require_data(dim0, {1, 2, 3, 4, 5, 6, 7, 8});

    auto dim1 = cpptensor::stack({a, b}, 1);
    require_shape(dim1, {2, 2, 2});
    require_data(dim1, {1, 2, 5, 6, 3, 4, 7, 8});

    auto dim2 = cpptensor::stack({a, b}, 2);
    require_shape(dim2, {2, 2, 2});
    require_data(dim2, {1, 5, 2, 6, 3, 7, 4, 8});
}

TEST_CASE("squeeze can reduce singleton tensors to scalars", "[manipulation][squeeze]") {
    cpptensor::Tensor singleton({1}, std::vector<float>{42});
    auto scalar = singleton.squeeze();
    require_shape(scalar, std::vector<size_t>{});
    REQUIRE(scalar.ndim() == 0);
    require_data(scalar, {42});

    cpptensor::Tensor all_singletons({1, 1, 1}, std::vector<float>{7});
    auto squeezed = all_singletons.squeeze();
    require_shape(squeezed, std::vector<size_t>{});
    REQUIRE(squeezed.ndim() == 0);
    require_data(squeezed, {7});

    auto specific_dim = all_singletons.squeeze(1);
    require_shape(specific_dim, {1, 1});
    REQUIRE(specific_dim.ndim() == 2);
    require_data(specific_dim, {7});
}

TEST_CASE("comparison operators support tensor, scalar, and broadcast operands", "[comparison]") {
    cpptensor::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor b({2, 3}, {1, 0, 3, 10, 5, 0});

    require_data(a == b, {1, 0, 1, 0, 1, 0});
    require_data(a != b, {0, 1, 0, 1, 0, 1});
    require_data(a > b, {0, 1, 0, 0, 0, 1});
    require_data(a < b, {0, 0, 0, 1, 0, 0});
    require_data(a >= b, {1, 1, 1, 0, 1, 1});
    require_data(a <= b, {1, 0, 1, 1, 1, 0});

    require_data(a > 3.0f, {0, 0, 0, 1, 1, 1});
    require_data(3.0f < a, {0, 0, 0, 1, 1, 1});

    cpptensor::Tensor row({1, 3}, {1, 5, 10});
    require_shape(a < row, {2, 3});
    require_data(a < row, {0, 1, 1, 0, 0, 1});
}

TEST_CASE("compute_broadcast_shape rejects incompatible dimensions and preserves valid broadcasts",
          "[broadcast]") {
    REQUIRE_THROWS_WITH(cpptensor::compute_broadcast_shape({2}, {3}),
                        Catch::Matchers::ContainsSubstring("incompatible dimensions 2 and 3"));

    REQUIRE(cpptensor::compute_broadcast_shape({2, 1, 4}, {1, 3, 4}) ==
            std::vector<size_t>{2, 3, 4});
}

TEST_CASE("gemv and matmul produce the same matrix-vector result", "[matmul][gemv]") {
    cpptensor::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor x({3}, {1, 0, -1});

    require_shape(cpptensor::gemv(a, x), {2});
    require_data(cpptensor::gemv(a, x), {-2, -2});

    auto y = cpptensor::matmul(a, x);
    require_shape(y, {2});
    require_data(y, {-2, -2});
}

TEST_CASE("reductions handle global and dimension-specific forms", "[reduction]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

    auto global_sum = t.sum();
    require_shape(global_sum, {});
    REQUIRE(global_sum.ndim() == 0);
    require_data(global_sum, {21});

    auto global_mean = t.mean();
    require_shape(global_mean, {});
    REQUIRE(global_mean.ndim() == 0);
    require_data(global_mean, {3.5f});

    auto global_max = t.max();
    require_shape(global_max, {});
    REQUIRE(global_max.ndim() == 0);
    require_data(global_max, {6});

    auto global_min = t.min();
    require_shape(global_min, {});
    REQUIRE(global_min.ndim() == 0);
    require_data(global_min, {1});

    require_shape(t.sum(true), {1, 1});
    require_data(t.sum(true), {21});
    require_shape(t.mean(true), {1, 1});
    require_data(t.mean(true), {3.5f});
    require_shape(t.max(true), {1, 1});
    require_data(t.max(true), {6});
    require_shape(t.min(true), {1, 1});
    require_data(t.min(true), {1});

    cpptensor::Tensor v({3}, {1, 2, 3});
    require_shape(v.sum(0), {});
    require_data(v.sum(0), {6});
    require_shape(v.mean(0), {});
    require_data(v.mean(0), {2});
    require_shape(v.max(0), {});
    require_data(v.max(0), {3});
    require_shape(v.min(0), {});
    require_data(v.min(0), {1});

    require_shape(t.sum(0), {3});
    require_data(t.sum(0), {5, 7, 9});
    require_shape(t.sum(1, true), {2, 1});
    require_data(t.sum(1, true), {6, 15});

    require_shape(t.mean(1), {2});
    require_data(t.mean(1), {2, 5});

    require_shape(t.max(0), {3});
    require_data(t.max(0), {4, 5, 6});
    require_shape(t.min(-1), {2});
    require_data(t.min(-1), {1, 4});
}

TEST_CASE("contiguous materializes view values from the logical view start", "[tensor][contiguous]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});

    auto stepped = base.slice(0, 1, 4, 2);
    auto materialized = stepped.contiguous();

    REQUIRE_FALSE(stepped.is_contiguous());
    require_shape(materialized, {2});
    require_data(materialized, {1, 3});
}

TEST_CASE("contiguous honors raw-pointer-backed view offsets", "[tensor][contiguous][from_ptr]") {
    cpptensor::Tensor owner({6}, {0, 1, 2, 3, 4, 5});
    auto subrange = cpptensor::Tensor::from_ptr(
        {4},
        owner.data().data() + 1,
        owner.impl(),
        owner.device_type());

    auto stepped = subrange.slice(0, 0, 4, 2);
    auto materialized = stepped.contiguous();

    REQUIRE_FALSE(stepped.is_contiguous());
    require_shape(materialized, {2});
    require_data(materialized, {1, 3});
}

TEST_CASE("clone deep-copies sliced views using the logical view contents", "[tensor][clone][view]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});

    auto sliced = base.slice(0, 1, 3);
    auto cloned = sliced.clone();

    require_shape(cloned, {2});
    require_data(cloned, {1, 2});

    base.data()[1] = 99.0f;
    require_data(cloned, {1, 2});
}

TEST_CASE("clone preserves the logical order of transposed views", "[tensor][clone][view]") {
    cpptensor::Tensor matrix({2, 2}, {1, 2, 3, 4});

    auto transposed = matrix.transpose(0, 1);
    auto cloned = transposed.clone();

    require_shape(cloned, {2, 2});
    require_data(cloned, {1, 3, 2, 4});
}

TEST_CASE("contiguous copies non-contiguous view values from the logical offset",
          "[tensor][contiguous][view]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});

    auto stepped = base.slice(0, 1, 4, 2);
    auto materialized = stepped.contiguous();

    REQUIRE_FALSE(stepped.is_contiguous());
    require_shape(materialized, {2});
    require_data(materialized, {1, 3});

    base.data()[1] = 99.0f;
    require_data(materialized, {1, 3});
}
