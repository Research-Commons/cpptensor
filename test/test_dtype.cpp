#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/tensor/tensor.hpp"

#include <cstdint>
#include <vector>

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

void require_data(const cpptensor::Tensor& tensor, const std::vector<float>& expected) {
    const auto& actual = tensor.data();
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(actual[i] == Approx(expected[i]));
    }
}

} // namespace

TEST_CASE("tensor dtype metadata survives view/clone/contiguous", "[tensor][dtype]") {
    cpptensor::Tensor ints({2, 2}, std::vector<std::int32_t>{1, 2, 3, 4});
    REQUIRE(ints.dtype() == DType::INT32);

    auto viewed = ints.view({4});
    REQUIRE(viewed.dtype() == DType::INT32);

    auto transposed = ints.transpose();
    REQUIRE(transposed.dtype() == DType::INT32);

    auto compact = transposed.contiguous();
    REQUIRE(compact.dtype() == DType::INT32);
    require_data(compact, {1, 3, 2, 4});

    auto cloned = ints.clone();
    REQUIRE(cloned.dtype() == DType::INT32);
    require_data(cloned, {1, 2, 3, 4});
}

TEST_CASE("comparison ops return bool tensors", "[comparison][dtype]") {
    cpptensor::Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    cpptensor::Tensor b({2, 2}, {1.0f, 5.0f, 2.0f, 4.0f});

    auto gt_mask = a > b;
    REQUIRE(gt_mask.dtype() == DType::BOOL);
    require_data(gt_mask, {0.0f, 0.0f, 1.0f, 0.0f});

    cpptensor::Tensor ints({2, 2}, std::vector<std::int32_t>{1, 2, 3, 4});
    cpptensor::Tensor doubles({2, 2}, std::vector<double>{1.0, 1.5, 3.0, 5.0});
    auto eq_mask = ints == doubles;
    REQUIRE(eq_mask.dtype() == DType::BOOL);
    require_data(eq_mask, {1.0f, 0.0f, 1.0f, 0.0f});
}

TEST_CASE("dtype-aware factories and cast round-trip", "[tensor][dtype]") {
    auto zeros_i32 = cpptensor::Tensor::zeros({2, 2}, DeviceType::CPU, DType::INT32);
    REQUIRE(zeros_i32.dtype() == DType::INT32);
    require_data(zeros_i32, {0, 0, 0, 0});

    auto ones_bool = cpptensor::Tensor::ones({3}, DeviceType::CPU, DType::BOOL);
    REQUIRE(ones_bool.dtype() == DType::BOOL);
    require_data(ones_bool, {1, 1, 1});

    auto full_f64 = cpptensor::Tensor::full({2}, 2.5, DeviceType::CPU, DType::FLOAT64);
    REQUIRE(full_f64.dtype() == DType::FLOAT64);
    require_data(full_f64, {2.5f, 2.5f});

    auto casted = full_f64.astype(DType::INT32);
    REQUIRE(casted.dtype() == DType::INT32);
    require_data(casted, {2.0f, 2.0f});

    auto round_trip = casted.astype(DType::FLOAT64);
    REQUIRE(round_trip.dtype() == DType::FLOAT64);
    require_data(round_trip, {2.0f, 2.0f});
}

TEST_CASE("mixed dtype arithmetic has explicit promotion/rejection behavior", "[arithmetic][dtype]") {
    cpptensor::Tensor ints({2}, std::vector<std::int32_t>{1, 2});
    cpptensor::Tensor floats({2}, std::vector<float>{0.5f, 1.5f});

    auto promoted = ints + floats;
    REQUIRE(promoted.dtype() == DType::FLOAT32);
    require_data(promoted, {1.5f, 3.5f});

    cpptensor::Tensor more_ints({2}, std::vector<std::int32_t>{2, 3});
    REQUIRE_THROWS_WITH(
        ints + more_ints,
        ContainsSubstring("float32 compute only"));
}
