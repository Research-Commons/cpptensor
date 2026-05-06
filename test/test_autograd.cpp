#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/ops/manipulation/stack.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/tensor/tensor.hpp"

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

} // namespace

TEST_CASE("device metadata is preserved across tensor creation and view materialization",
          "[tensor][device]") {
    auto cpu = cpptensor::Tensor::ones({2, 3}, DeviceType::CPU);
    auto cuda = cpptensor::Tensor::full({2, 3}, 5.0f, DeviceType::CUDA);

    REQUIRE(cpu.device_type() == DeviceType::CPU);
    REQUIRE(cuda.device_type() == DeviceType::CUDA);

    auto cuda_slice = cuda.slice(0, 0, 1);
    REQUIRE(cuda_slice.device_type() == DeviceType::CUDA);

    auto cuda_clone = cuda_slice.clone();
    auto cuda_compact = cuda_slice.contiguous();
    REQUIRE(cuda_clone.device_type() == DeviceType::CUDA);
    REQUIRE(cuda_compact.device_type() == DeviceType::CUDA);

    require_shape(cuda_clone, {1, 3});
    require_data(cuda_clone, {5, 5, 5});
}

TEST_CASE("mixed-device contracts fail consistently across core tensor ops",
          "[tensor][device-mismatch]") {
    cpptensor::Tensor cpu_vec({2}, {1, 2}, DeviceType::CPU);
    cpptensor::Tensor cuda_vec({2}, {3, 4}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(cpu_vec + cuda_vec,
                        ContainsSubstring("Binary op requires matching devices"));

    cpptensor::Tensor cpu_mat({2, 2}, {1, 2, 3, 4}, DeviceType::CPU);
    cpptensor::Tensor cuda_mat({2, 2}, {5, 6, 7, 8}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(cpptensor::matmul(cpu_mat, cuda_mat),
                        ContainsSubstring("matmul: device mismatch"));
    REQUIRE_THROWS_WITH(cpptensor::cat({cpu_mat, cuda_mat}, 0),
                        ContainsSubstring("same device"));
    REQUIRE_THROWS_WITH(cpptensor::stack({cpu_mat, cuda_mat}, 0),
                        ContainsSubstring("same device"));
}

TEST_CASE("shape and dimension contract regressions stay enforced", "[tensor][contracts]") {
    cpptensor::Tensor a({2}, {1, 2});
    cpptensor::Tensor b({3}, {3, 4, 5});
    REQUIRE_THROWS_WITH(a + b, ContainsSubstring("not broadcastable"));

    cpptensor::Tensor left({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor right({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    REQUIRE_THROWS_WITH(cpptensor::matmul(left, right),
                        ContainsSubstring("dimension mismatch"));

    cpptensor::Tensor x({2, 2}, {1, 2, 3, 4});
    cpptensor::Tensor y({2, 3}, {5, 6, 7, 8, 9, 10});
    REQUIRE_THROWS_WITH(cpptensor::cat({x, y}, 0), ContainsSubstring("same shape except"));
    REQUIRE_THROWS_WITH(cpptensor::stack({x, y}, 0), ContainsSubstring("same shape"));

    REQUIRE_THROWS_WITH(left.sum(2), ContainsSubstring("Sum dimension out of range"));
    REQUIRE_THROWS_WITH(left.mean(-3), ContainsSubstring("Mean dimension out of range"));
    REQUIRE_THROWS_WITH(left.max(2), ContainsSubstring("out of range"));
    REQUIRE_THROWS_WITH(left.min(2), ContainsSubstring("out of range"));
}
