#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/utils/broadcastUtils.hpp"

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
