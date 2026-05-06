#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/tensor/tensor.hpp"

#include <cmath>
#include <vector>

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

void require_shape(const cpptensor::Tensor& tensor, const std::vector<size_t>& expected) {
    REQUIRE(tensor.shape() == expected);
}

void require_data(const cpptensor::Tensor& tensor, const std::vector<float>& expected) {
    const auto& data = tensor.data();
    REQUIRE(data.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(data[i] == Approx(expected[i]));
    }
}

void require_scalar(const cpptensor::Tensor& tensor, float expected) {
    require_shape(tensor, {});
    require_data(tensor, {expected});
}

} // namespace

TEST_CASE("tensor construction and factory helpers preserve shape invariants", "[tensor][factories]") {
    cpptensor::Tensor matrix({2, 3}, {1, 2, 3, 4, 5, 6});
    require_shape(matrix, {2, 3});
    REQUIRE(matrix.ndim() == 2);
    REQUIRE(matrix.numel() == 6);

    auto zeros = cpptensor::Tensor::zeros({2, 2});
    auto ones = cpptensor::Tensor::ones({2, 2});
    auto full = cpptensor::Tensor::full({2, 2}, 7.5f);
    auto randn = cpptensor::Tensor::randn({16});

    require_data(zeros, {0, 0, 0, 0});
    require_data(ones, {1, 1, 1, 1});
    require_data(full, {7.5f, 7.5f, 7.5f, 7.5f});
    REQUIRE(randn.shape() == std::vector<size_t>{16});

    bool has_non_zero = false;
    for (float value : randn.data()) {
        REQUIRE(std::isfinite(value));
        if (value != 0.0f) {
            has_non_zero = true;
        }
    }
    REQUIRE(has_non_zero);
}

TEST_CASE("view and reshape follow contiguity and aliasing contracts", "[tensor][view][reshape]") {
    cpptensor::Tensor base({2, 3}, {1, 2, 3, 4, 5, 6});

    auto reshaped_view = base.view({3, 2});
    REQUIRE(reshaped_view.is_contiguous());
    reshaped_view.data()[2] = 99.0f;
    REQUIRE(base.data()[2] == Approx(99.0f));

    auto transposed = base.transpose();
    REQUIRE_FALSE(transposed.is_contiguous());
    REQUIRE_THROWS_WITH(transposed.view({6}), ContainsSubstring("must be contiguous"));

    auto flattened_copy = transposed.reshape({6});
    REQUIRE(flattened_copy.is_contiguous());
    require_data(flattened_copy, {1, 4, 2, 5, 99, 6});

    flattened_copy.data()[0] = -123.0f;
    REQUIRE(base.data()[0] == Approx(1.0f));
}

TEST_CASE("tuple indexing and stepped slicing return stable logical contents", "[tensor][slice][index]") {
    cpptensor::Tensor cube(
        {2, 3, 4},
        {0, 1, 2, 3,
         4, 5, 6, 7,
         8, 9, 10, 11,
         12, 13, 14, 15,
         16, 17, 18, 19,
         20, 21, 22, 23});

    auto indexed = cube.index({
        cpptensor::Tensor::SliceSpec{0, 2},
        1,
        cpptensor::Tensor::SliceSpec{0, 4, 2},
    });
    require_shape(indexed, {2, 2});
    require_data(indexed, {4, 6, 16, 18});

    cpptensor::Tensor vector({6}, {0, 1, 2, 3, 4, 5});
    auto reversed = vector.slice(0, std::nullopt, std::nullopt, -1);
    require_shape(reversed, {6});
    require_data(reversed, {5, 4, 3, 2, 1, 0});
}

TEST_CASE("broadcasted arithmetic composes with reductions", "[tensor][broadcast][reduction]") {
    cpptensor::Tensor lhs({2, 1, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor rhs({1, 4, 1}, {10, 20, 30, 40});

    auto out = lhs + rhs;
    require_shape(out, {2, 4, 3});
    require_data(out, {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        14, 15, 16,
        24, 25, 26,
        34, 35, 36,
        44, 45, 46,
    });

    auto reduced = out.mean(1, true);
    require_shape(reduced, {2, 1, 3});
    require_data(reduced, {26, 27, 28, 29, 30, 31});

    require_scalar(out.sum(), 684.0f);
}

TEST_CASE("tensor APIs throw stable errors for invalid shape and dimension inputs",
          "[tensor][negative]") {
    REQUIRE_THROWS_WITH(cpptensor::Tensor({2, 2}, {1, 2, 3}), ContainsSubstring("data size"));

    cpptensor::Tensor matrix({2, 3}, {1, 2, 3, 4, 5, 6});
    REQUIRE_THROWS_WITH(matrix.view({5}), ContainsSubstring("cannot reshape"));

    cpptensor::Tensor scalar(std::vector<size_t>{}, std::vector<float>{1.0f});
    REQUIRE_THROWS_WITH(scalar.flatten(), ContainsSubstring("cannot flatten scalar"));
    REQUIRE_THROWS_WITH(matrix.flatten(1, 0), ContainsSubstring("invalid dimension range"));

    REQUIRE_THROWS_WITH(matrix.slice(2), ContainsSubstring("out of range"));

    REQUIRE_THROWS_WITH(
        matrix.index({0, 0, 0}),
        ContainsSubstring("received 3 indices for tensor with rank 2"));

    REQUIRE_THROWS_WITH(
        matrix.index({cpptensor::Tensor::SliceSpec{0, 2, 0}}),
        ContainsSubstring("slice step cannot be zero"));

    REQUIRE_THROWS_WITH(matrix.squeeze(1), ContainsSubstring("expected 1"));
    REQUIRE_THROWS_WITH(matrix.unsqueeze(3), ContainsSubstring("dimension out of range"));
}
