#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/utils/broadcastUtils.hpp"

#include <vector>

TEST_CASE("pad_shape_right rejects smaller target ranks", "[broadcast][utils]") {
    REQUIRE_THROWS_WITH(
        cpptensor::pad_shape_right({2, 3}, 1),
        Catch::Matchers::ContainsSubstring("target rank smaller than shape rank"));
}

TEST_CASE("pad_shape_right left-pads shapes up to the requested rank", "[broadcast][utils]") {
    REQUIRE(cpptensor::pad_shape_right({2, 3}, 4) == std::vector<size_t>{1, 1, 2, 3});
    REQUIRE(cpptensor::pad_shape_right({2, 3}, 2) == std::vector<size_t>{2, 3});
}

TEST_CASE("squeeze_padded_to_unpadded rejects buffers whose size does not match padded_shape", "[broadcast][utils]") {
    const std::vector<size_t> padded_shape{2, 2};
    const std::vector<size_t> unpadded_shape{2, 2};

    REQUIRE_THROWS_WITH(
        cpptensor::squeeze_padded_to_unpadded({1.0f, 2.0f, 3.0f}, padded_shape, unpadded_shape),
        Catch::Matchers::ContainsSubstring("padded buffer size"));

    REQUIRE_THROWS_WITH(
        cpptensor::squeeze_padded_to_unpadded({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, padded_shape, unpadded_shape),
        Catch::Matchers::ContainsSubstring("padded buffer size"));
}

TEST_CASE("squeeze_padded_to_unpadded rejects oversized scalar reductions", "[broadcast][utils]") {
    const std::vector<size_t> padded_shape{1, 1};
    const std::vector<size_t> unpadded_shape{};

    REQUIRE_THROWS_WITH(
        cpptensor::squeeze_padded_to_unpadded({1.0f, 99.0f}, padded_shape, unpadded_shape),
        Catch::Matchers::ContainsSubstring("padded buffer size"));
}

TEST_CASE("compute_broadcast_view_strides returns stride-0 for expanded dimensions", "[broadcast][utils]") {
    const std::vector<size_t> source_shape{1, 3};
    const std::vector<size_t> source_stride{3, 1};

    REQUIRE(cpptensor::compute_broadcast_view_strides(source_shape, source_stride, {4, 3}) ==
            std::vector<size_t>{0, 1});
    REQUIRE(cpptensor::compute_broadcast_view_strides(source_shape, source_stride, {2, 1, 3}) ==
            std::vector<size_t>{0, 3, 1});
}

TEST_CASE("compute_broadcast_view_strides rejects incompatible shapes", "[broadcast][utils]") {
    REQUIRE_THROWS_WITH(
        cpptensor::compute_broadcast_view_strides({2, 3}, {3, 1}, {4, 3}),
        Catch::Matchers::ContainsSubstring("not broadcastable"));
}
