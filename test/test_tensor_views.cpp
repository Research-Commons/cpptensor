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
