#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cpptensor/tensor/tensor.hpp"

#include <utility>
#include <vector>

using Catch::Approx;

namespace {

void require_data(const cpptensor::Tensor& tensor, const std::vector<float>& expected) {
    const auto& data = tensor.data();
    REQUIRE(data.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(data[i] == Approx(expected[i]));
    }
}

} // namespace

TEST_CASE("slice data exposes logical const contents and rejects mutable view storage",
          "[tensor][views][data]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});
    cpptensor::Tensor slice = base.slice(0, 1, 3);

    const auto& logical = std::as_const(slice).data();
    REQUIRE(logical.size() == 2);
    REQUIRE(logical[0] == Approx(1.0f));
    REQUIRE(logical[1] == Approx(2.0f));
    REQUIRE_THROWS(slice.data());

    cpptensor::Tensor compact = slice.contiguous();
    require_data(compact, {1, 2});

    auto& compact_data = compact.data();
    compact_data[0] = 99.0f;
    REQUIRE(base.data()[1] == Approx(1.0f));
}

TEST_CASE("transpose data materializes logical row-major order", "[tensor][views][data]") {
    cpptensor::Tensor base({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor transposed = base.transpose();

    require_data(transposed, {1, 4, 2, 5, 3, 6});
    REQUIRE_THROWS(transposed.data());

    cpptensor::Tensor cloned = transposed.clone();
    require_data(cloned, {1, 4, 2, 5, 3, 6});
    REQUIRE(cloned.shape() == std::vector<size_t>{3, 2});
}

TEST_CASE("full contiguous views still expose shared mutable storage", "[tensor][views][data]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});
    cpptensor::Tensor reshaped = base.view({2, 2});

    auto& reshaped_data = reshaped.data();
    reshaped_data[0] = 42.0f;

    REQUIRE(base.data()[0] == Approx(42.0f));
    require_data(reshaped, {42, 1, 2, 3});
}

TEST_CASE("tuple-style indexing supports multi-axis slice and scalar selection",
          "[tensor][views][index]") {
    cpptensor::Tensor base(
        {2, 3, 4},
        {0, 1, 2, 3,
         4, 5, 6, 7,
         8, 9, 10, 11,
         12, 13, 14, 15,
         16, 17, 18, 19,
         20, 21, 22, 23});

    cpptensor::Tensor indexed = base.index({
        cpptensor::Tensor::SliceSpec{0, 2},
        -2,
        cpptensor::Tensor::SliceSpec{1, 4, 2},
    });

    REQUIRE(indexed.shape() == std::vector<size_t>{2, 2});
    require_data(indexed, {5, 7, 17, 19});

    base.data()[5] = 55.0f;
    require_data(indexed, {55, 7, 17, 19});
    REQUIRE_THROWS(indexed.data());
}

TEST_CASE("negative-step slicing is supported", "[tensor][views][slice]") {
    cpptensor::Tensor base({6}, {0, 1, 2, 3, 4, 5});
    cpptensor::Tensor reversed = base.slice(0, std::nullopt, std::nullopt, -1);

    REQUIRE(reversed.shape() == std::vector<size_t>{6});
    require_data(reversed, {5, 4, 3, 2, 1, 0});

    base.data()[0] = 123.0f;
    require_data(reversed, {5, 4, 3, 2, 1, 0});
}

TEST_CASE("scalar indexing reduces dimensions and supports negative indices",
          "[tensor][views][index]") {
    cpptensor::Tensor base({2, 3}, {0, 1, 2, 3, 4, 5});
    cpptensor::Tensor scalar = base.index({-1, -2});

    REQUIRE(scalar.shape().empty());
    const auto& scalar_data = std::as_const(scalar).data();
    REQUIRE(scalar_data.size() == 1);
    REQUIRE(scalar_data[0] == Approx(4.0f));

    base.data()[4] = 42.0f;
    REQUIRE(std::as_const(scalar).data()[0] == Approx(42.0f));
    REQUIRE_THROWS(scalar.data());
}

TEST_CASE("slice plus transpose remains correct with tuple-style indexing",
          "[tensor][views][index][transpose]") {
    cpptensor::Tensor base(
        {3, 4},
        {0, 1, 2, 3,
         4, 5, 6, 7,
         8, 9, 10, 11});

    cpptensor::Tensor sliced = base.index({
        cpptensor::Tensor::SliceSpec{1, 3},
        cpptensor::Tensor::SliceSpec{0, 4, 2},
    });
    cpptensor::Tensor transposed = sliced.transpose();

    REQUIRE(sliced.shape() == std::vector<size_t>{2, 2});
    require_data(sliced, {4, 6, 8, 10});

    REQUIRE(transposed.shape() == std::vector<size_t>{2, 2});
    require_data(transposed, {4, 8, 6, 10});
}

TEST_CASE("expand and broadcast_to create zero-copy broadcast views", "[tensor][views][broadcast]") {
    cpptensor::Tensor row({1, 3}, {1, 2, 3});

    auto expanded = row.expand({4, 3});
    REQUIRE(expanded.shape() == std::vector<size_t>{4, 3});
    REQUIRE(expanded.stride() == std::vector<size_t>{0, 1});
    REQUIRE(expanded.impl()->data_ptr() == row.impl()->data_ptr());
    require_data(expanded, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});
    REQUIRE_THROWS(expanded.data());

    auto mixed_rank = row.broadcast_to({2, 1, 3});
    REQUIRE(mixed_rank.shape() == std::vector<size_t>{2, 1, 3});
    REQUIRE(mixed_rank.stride() == std::vector<size_t>{0, 3, 1});
    require_data(mixed_rank, {1, 2, 3, 1, 2, 3});
}

TEST_CASE("expand rejects incompatible target shapes", "[tensor][views][broadcast]") {
    cpptensor::Tensor matrix({2, 3}, {1, 2, 3, 4, 5, 6});

    REQUIRE_THROWS(matrix.expand({3, 3}));
    REQUIRE_THROWS(matrix.broadcast_to({3}));
}

TEST_CASE("repeat materializes tiled tensors from contiguous and view inputs", "[tensor][repeat]") {
    cpptensor::Tensor base({2, 1}, {1, 2});
    REQUIRE_THROWS(base.repeat({2}));

    auto repeated = base.repeat({1, 3});
    REQUIRE(repeated.shape() == std::vector<size_t>{2, 3});
    require_data(repeated, {1, 1, 1, 2, 2, 2});

    cpptensor::Tensor vector({3}, {1, 2, 3});
    auto mixed_rank = vector.repeat({2, 2});
    REQUIRE(mixed_rank.shape() == std::vector<size_t>{2, 6});
    require_data(mixed_rank, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});

    cpptensor::Tensor matrix({2, 3}, {1, 2, 3, 4, 5, 6});
    auto transposed = matrix.transpose();
    auto transposed_repeat = transposed.repeat({1, 2});
    REQUIRE(transposed_repeat.shape() == std::vector<size_t>{3, 4});
    require_data(transposed_repeat, {1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6});
}
