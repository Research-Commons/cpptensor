#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cpptensor/ops/arithmetic/add.hpp"
#include "cpptensor/ops/arithmetic/div.hpp"
#include "cpptensor/ops/arithmetic/mul.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/tensor/tensor.hpp"

#include <vector>

using Catch::Approx;

namespace {

void require_data(const cpptensor::Tensor& tensor, const std::vector<float>& expected) {
    REQUIRE(tensor.data().size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        INFO("index " << i);
        REQUIRE(tensor.data()[i] == Approx(expected[i]));
    }
}

} // namespace

TEST_CASE("autograd computes scalar chain gradients", "[autograd]") {
    auto a = cpptensor::Tensor::full({}, 2.0f, true);
    auto b = cpptensor::Tensor::full({}, 3.0f, true);
    auto c = cpptensor::Tensor::full({}, 4.0f, true);

    auto z = a * b * c;
    z.backward();

    require_data(a.grad(), {12.0f});
    require_data(b.grad(), {8.0f});
    require_data(c.grad(), {6.0f});
}

TEST_CASE("autograd accumulates gradients when tensor is reused", "[autograd]") {
    auto x = cpptensor::Tensor::full({}, 3.0f, true);
    auto z = x * x + x + x;

    z.backward();
    require_data(x.grad(), {8.0f});

    z.backward();
    require_data(x.grad(), {16.0f});
}

TEST_CASE("autograd supports tensor-output backward and broadcasting", "[autograd]") {
    auto a = cpptensor::Tensor({2, 1}, {1.0f, 2.0f}, true);
    auto b = cpptensor::Tensor({1, 3}, {10.0f, 20.0f, 30.0f}, true);

    auto y = a + b;
    y.backward();

    require_data(a.grad(), {3.0f, 3.0f});
    require_data(b.grad(), {2.0f, 2.0f, 2.0f});
}

TEST_CASE("zero_grad resets gradients deterministically", "[autograd]") {
    auto x = cpptensor::Tensor::full({}, 2.0f, true);
    auto y = x * x;

    y.backward();
    require_data(x.grad(), {4.0f});

    x.zero_grad();
    require_data(x.grad(), {0.0f});

    y.backward();
    require_data(x.grad(), {4.0f});
}

TEST_CASE("autograd computes matmul gradients", "[autograd]") {
    auto a = cpptensor::Tensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    auto b = cpptensor::Tensor({3, 2}, {7, 8, 9, 10, 11, 12}, true);

    auto out = cpptensor::matmul(a, b);
    auto grad_out = cpptensor::Tensor::ones({2, 2});
    out.backward(grad_out);

    require_data(a.grad(), {15, 19, 23, 15, 19, 23});
    require_data(b.grad(), {5, 5, 7, 7, 9, 9});
}

TEST_CASE("unsupported autograd operations fail explicitly", "[autograd]") {
    auto x = cpptensor::Tensor::full({2, 2}, 2.0f, true);
    auto y = cpptensor::Tensor::full({2, 2}, 1.0f, true);

    REQUIRE_THROWS_WITH(x / y, Catch::Matchers::ContainsSubstring("autograd"));
    REQUIRE_THROWS_WITH(x.transpose(), Catch::Matchers::ContainsSubstring("autograd"));
}
