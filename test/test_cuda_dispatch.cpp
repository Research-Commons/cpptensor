#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/backend/backend_loader.hpp"
#include "cpptensor/ops/comparison/eq.hpp"
#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/ops/math/exp.hpp"
#include "cpptensor/tensor/tensor.hpp"

using Catch::Matchers::ContainsSubstring;

TEST_CASE("CUDA-tagged tensors fail clearly when a binary kernel is unavailable",
          "[dispatcher][cuda][fallback]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor a({2}, {1.0f, 2.0f}, DeviceType::CUDA);
    cpptensor::Tensor b({2}, {3.0f, 4.0f}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(a - b,
                        ContainsSubstring("No forward kernel registered for op Sub on device CUDA"));

#ifndef BUILD_CUDA
    REQUIRE_THROWS_WITH(a + b,
                        ContainsSubstring("No forward kernel registered for op Add on device CUDA"));
#endif
}

TEST_CASE("mixed-device binary ops fail with a consistent boundary error",
          "[dispatcher][device-mismatch][binary]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor cpu({2}, {1.0f, 2.0f}, DeviceType::CPU);
    cpptensor::Tensor cuda({2}, {3.0f, 4.0f}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(cpu - cuda,
                        ContainsSubstring("Binary op requires matching devices, got lhs=CPU and rhs=CUDA"));
    REQUIRE_THROWS_WITH(cuda * cpu,
                        ContainsSubstring("Binary op requires matching devices, got lhs=CUDA and rhs=CPU"));
    REQUIRE_THROWS_WITH(cpu / cuda,
                        ContainsSubstring("Binary op requires matching devices, got lhs=CPU and rhs=CUDA"));
    REQUIRE_THROWS_WITH(cpptensor::eq(cpu, cuda),
                        ContainsSubstring("Binary op requires matching devices, got lhs=CPU and rhs=CUDA"));
    REQUIRE_THROWS_WITH((cuda < cpu),
                        ContainsSubstring("Binary op requires matching devices, got lhs=CUDA and rhs=CPU"));
}

TEST_CASE("CUDA-tagged tensors fail clearly when a unary kernel is unavailable",
          "[dispatcher][cuda][fallback]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor a({2}, {1.0f, 2.0f}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(cpptensor::exp(a),
                        ContainsSubstring("No unary kernel registered for op Exp on device CUDA"));
}

TEST_CASE("CUDA-tagged tensors fail clearly when a reduction kernel is unavailable",
          "[dispatcher][cuda][fallback]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor a({2}, {1.0f, 2.0f}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(a.sum(),
                        ContainsSubstring("No reduction kernel registered for op Sum on device CUDA"));
}
