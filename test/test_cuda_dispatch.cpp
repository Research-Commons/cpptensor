#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/backend/backend_loader.hpp"
#include "cpptensor/ops/arithmetic/add.hpp"
#include "cpptensor/ops/arithmetic/div.hpp"
#include "cpptensor/ops/arithmetic/mul.hpp"
#include "cpptensor/ops/arithmetic/pow.hpp"
#include "cpptensor/ops/comparison/eq.hpp"
#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/ops/manipulation/stack.hpp"
#include "cpptensor/ops/math/exp.hpp"
#include "cpptensor/ops/math/log.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/ops/math/sqrt.hpp"
#include "cpptensor/tensor/tensor.hpp"

using Catch::Matchers::ContainsSubstring;

TEST_CASE("CUDA-tagged tensors fail clearly when a binary kernel is unavailable",
          "[dispatcher][cuda][fallback]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor a({2}, {1.0f, 2.0f}, DeviceType::CUDA);
    cpptensor::Tensor b({2}, {3.0f, 4.0f}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(a - b,
                        ContainsSubstring("No forward kernel registered for op Sub on device CUDA"));
    REQUIRE_THROWS_WITH(a / b,
                        ContainsSubstring("No forward kernel registered for op Div on device CUDA"));
    REQUIRE_THROWS_WITH(cpptensor::pow(a, b),
                        ContainsSubstring("No forward kernel registered for op Pow on device CUDA"));

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
    REQUIRE_THROWS_WITH(cpptensor::log(a),
                        ContainsSubstring("No unary kernel registered for op Log on device CUDA"));
    REQUIRE_THROWS_WITH(cpptensor::sqrt(a),
                        ContainsSubstring("No unary kernel registered for op Sqrt on device CUDA"));
}

TEST_CASE("CUDA-tagged tensors fail clearly when a reduction kernel is unavailable",
          "[dispatcher][cuda][fallback]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor a({2}, {1.0f, 2.0f}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(a.sum(),
                        ContainsSubstring("No reduction kernel registered for op Sum on device CUDA"));
}

TEST_CASE("Tensor device transfer APIs are explicit and round-trip logical values",
          "[tensor][device][transfer]") {
    cpptensor::Tensor cpu({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, DeviceType::CPU);
    auto cpu_copy = cpu.copy_to(DeviceType::CPU);
    REQUIRE(cpu_copy.device_type() == DeviceType::CPU);
    REQUIRE(cpu_copy.shape() == cpu.shape());
    REQUIRE(cpu_copy.data() == cpu.data());

#ifdef BUILD_CUDA
    auto cuda = cpu.to(DeviceType::CUDA);
    REQUIRE(cuda.device_type() == DeviceType::CUDA);
    auto back = cuda.to(DeviceType::CPU);
    REQUIRE(back.device_type() == DeviceType::CPU);
    REQUIRE(back.data() == cpu.data());

    auto copied_back = cpu.copy_to(DeviceType::CUDA).copy_to(DeviceType::CPU);
    REQUIRE(copied_back.data() == cpu.data());
#else
    REQUIRE_THROWS_WITH(cpu.to(DeviceType::CUDA),
                        ContainsSubstring("built without BUILD_CUDA"));
    REQUIRE_THROWS_WITH(cpu.copy_to(DeviceType::CUDA),
                        ContainsSubstring("built without BUILD_CUDA"));
#endif
}

#ifdef BUILD_CUDA
TEST_CASE("supported CUDA kernels keep tensors on device across chained ops",
          "[dispatcher][cuda][pipeline]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor a({3}, {1.0f, 2.0f, 3.0f}, DeviceType::CUDA);
    cpptensor::Tensor b({3}, {4.0f, 5.0f, 6.0f}, DeviceType::CUDA);

    auto summed = a + b;
    REQUIRE(summed.device_type() == DeviceType::CUDA);

    auto multiplied = summed * b;
    REQUIRE(multiplied.device_type() == DeviceType::CUDA);

    auto on_cpu = multiplied.to(DeviceType::CPU);
    REQUIRE(on_cpu.data() == std::vector<float>{20.0f, 35.0f, 54.0f});
}
#endif

TEST_CASE("unsupported CUDA ops fail loudly instead of running on CPU implicitly",
          "[dispatcher][cuda][fallback]") {
    cpptensor::Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, DeviceType::CUDA);
    cpptensor::Tensor b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f}, DeviceType::CUDA);

    REQUIRE_THROWS_WITH(cpptensor::matmul(a, b),
                        ContainsSubstring("no CUDA kernel is registered"));
    REQUIRE_THROWS_WITH(cpptensor::dot(cpptensor::Tensor({2}, {1.0f, 2.0f}, DeviceType::CUDA),
                                       cpptensor::Tensor({2}, {3.0f, 4.0f}, DeviceType::CUDA)),
                        ContainsSubstring("no CUDA kernel is registered"));
    REQUIRE_THROWS_WITH(cpptensor::cat({a, b}, 0),
                        ContainsSubstring("no CUDA kernel is registered"));
    REQUIRE_THROWS_WITH(cpptensor::stack({a, b}, 0),
                        ContainsSubstring("no CUDA kernel is registered"));
}
