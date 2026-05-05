#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cpptensor/ops/comparison/eq.hpp"
#include "cpptensor/ops/comparison/ge.hpp"
#include "cpptensor/ops/comparison/gt.hpp"
#include "cpptensor/ops/comparison/le.hpp"
#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/ops/comparison/ne.hpp"
#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/ops/manipulation/stack.hpp"
#include "cpptensor/backend/backend_loader.hpp"
#include "cpptensor/tensor/tensor.hpp"

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

TEST_CASE("reductions handle global and dimension-specific forms", "[reduction]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

    require_data(t.sum(), {21});
    require_data(t.mean(), {3.5f});
    require_data(t.max(), {6});
    require_data(t.min(), {1});

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
