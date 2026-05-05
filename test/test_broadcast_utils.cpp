#include <catch2/catch_test_macros.hpp>

#include "cpptensor/utils/broadcastUtils.hpp"

#include <vector>

TEST_CASE("pad_shape_right rejects smaller target ranks", "[broadcast][utils]") {
    try {
        static_cast<void>(cpptensor::pad_shape_right({2, 3}, 1));
        FAIL("Expected pad_shape_right to throw for a smaller target rank");
    } catch (const std::runtime_error& error) {
        REQUIRE(std::string(error.what()) == "pad_shape_right: target rank smaller than shape rank");
    }
}

TEST_CASE("pad_shape_right left-pads shapes up to the requested rank", "[broadcast][utils]") {
    REQUIRE(cpptensor::pad_shape_right({2, 3}, 4) == std::vector<size_t>{1, 1, 2, 3});
    REQUIRE(cpptensor::pad_shape_right({2, 3}, 2) == std::vector<size_t>{2, 3});
}
