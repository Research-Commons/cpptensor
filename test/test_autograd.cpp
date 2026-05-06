#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "gradient_check_utils.hpp"

#include "cpptensor/ops/arithmetic/add.hpp"
#include "cpptensor/ops/arithmetic/div.hpp"
#include "cpptensor/ops/arithmetic/mul.hpp"
#include "cpptensor/ops/arithmetic/pow.hpp"
#include "cpptensor/ops/math/exp.hpp"
#include "cpptensor/ops/math/log.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/ops/math/sqrt.hpp"
#include "cpptensor/ops/reduction/mean.hpp"
#include "cpptensor/ops/reduction/sum.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cpptensor::Tensor;
using cpptensor::test::AnalyticGradientFn;
using cpptensor::test::GradientCheckReport;
using cpptensor::test::GradientCheckTolerance;
using cpptensor::test::GradientInput;
using cpptensor::test::ScalarObjectiveFn;
using cpptensor::test::broadcast_offset;
using cpptensor::test::broadcast_shape;
using cpptensor::test::check_gradients;
using cpptensor::test::make_upstream;
using cpptensor::test::numel;
using cpptensor::test::row_major_strides;
using cpptensor::test::scalar_value;
using cpptensor::test::unravel_index;

void require_gradient_check_passes(const std::string& test_name,
                                   const std::vector<GradientInput>& inputs,
                                   const ScalarObjectiveFn& objective,
                                   const AnalyticGradientFn& analytic,
                                   GradientCheckTolerance tolerance) {
    const GradientCheckReport report = check_gradients(inputs, objective, analytic, tolerance);
    INFO(test_name << " :: " << report.summary());
    REQUIRE(report.passed());
}

template <typename Dfdx, typename Dfdy>
std::pair<std::vector<float>, std::vector<float>> binary_elementwise_grads(
    const std::vector<size_t>& lhs_shape,
    const std::vector<float>& lhs_values,
    const std::vector<size_t>& rhs_shape,
    const std::vector<float>& rhs_values,
    const std::vector<float>& upstream,
    Dfdx&& dfdx,
    Dfdy&& dfdy) {

    const auto out_shape = broadcast_shape(lhs_shape, rhs_shape);
    const size_t out_total = numel(out_shape);
    if (upstream.size() != out_total) {
        throw std::runtime_error("binary_elementwise_grads: upstream size mismatch");
    }

    const auto lhs_strides = row_major_strides(lhs_shape);
    const auto rhs_strides = row_major_strides(rhs_shape);

    std::vector<float> lhs_grad(numel(lhs_shape), 0.0f);
    std::vector<float> rhs_grad(numel(rhs_shape), 0.0f);

    for (size_t out_flat = 0; out_flat < out_total; ++out_flat) {
        const auto out_index = unravel_index(out_flat, out_shape);
        const size_t lhs_flat = broadcast_offset(out_index, lhs_shape, lhs_strides);
        const size_t rhs_flat = broadcast_offset(out_index, rhs_shape, rhs_strides);

        const float x = lhs_values[lhs_flat];
        const float y = rhs_values[rhs_flat];
        const float go = upstream[out_flat];

        lhs_grad[lhs_flat] += go * dfdx(x, y);
        rhs_grad[rhs_flat] += go * dfdy(x, y);
    }

    return {lhs_grad, rhs_grad};
}

template <typename DerivativeFn>
std::vector<float> unary_elementwise_grad(const std::vector<float>& values,
                                          const std::vector<float>& upstream,
                                          DerivativeFn&& derivative) {
    if (values.size() != upstream.size()) {
        throw std::runtime_error("unary_elementwise_grad: size mismatch");
    }

    std::vector<float> grad(values.size(), 0.0f);
    for (size_t i = 0; i < values.size(); ++i) {
        grad[i] = upstream[i] * derivative(values[i]);
    }
    return grad;
}

} // namespace

TEST_CASE("Gradient check: broadcasted binary arithmetic ops", "[autograd][gradcheck]") {
    SECTION("add") {
        const std::vector<GradientInput> inputs{
            {{2, 3}, {0.80f, -1.10f, 2.20f, 1.50f, -0.30f, 0.70f}},
            {{1, 3}, {1.20f, -0.40f, 0.90f}},
        };

        const std::vector<float> upstream = make_upstream(6, 0.20f, 0.05f, true);

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = (vars[0] + vars[1]) * Tensor({2, 3}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            return [&]() {
                auto [dx, dy] = binary_elementwise_grads(
                    vars[0].shape(), vars[0].data(),
                    vars[1].shape(), vars[1].data(),
                    upstream,
                    [](float, float) { return 1.0f; },
                    [](float, float) { return 1.0f; });
                return std::vector<std::vector<float>>{dx, dy};
            }();
        };

        require_gradient_check_passes("add broadcast grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 2.0e-3f, 2.0e-2f});
    }

    SECTION("mul") {
        const std::vector<GradientInput> inputs{
            {{2, 3}, {0.60f, -1.30f, 2.00f, -0.80f, 0.45f, 1.10f}},
            {{1, 3}, {1.25f, -0.55f, 0.90f}},
        };

        const std::vector<float> upstream = make_upstream(6, 0.18f, 0.06f, true);

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = (vars[0] * vars[1]) * Tensor({2, 3}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            auto [dx, dy] = binary_elementwise_grads(
                vars[0].shape(), vars[0].data(),
                vars[1].shape(), vars[1].data(),
                upstream,
                [](float, float y) { return y; },
                [](float x, float) { return x; });
            return std::vector<std::vector<float>>{dx, dy};
        };

        require_gradient_check_passes("mul broadcast grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 2.5e-3f, 3.0e-2f});
    }

    SECTION("div") {
        const std::vector<GradientInput> inputs{
            {{2, 3}, {0.75f, -1.40f, 1.80f, -0.35f, 0.95f, 0.50f}},
            {{1, 3}, {1.10f, 0.70f, 1.60f}},
        };

        const std::vector<float> upstream = make_upstream(6, 0.16f, 0.04f, true);

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = (vars[0] / vars[1]) * Tensor({2, 3}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            auto [dx, dy] = binary_elementwise_grads(
                vars[0].shape(), vars[0].data(),
                vars[1].shape(), vars[1].data(),
                upstream,
                [](float, float y) { return 1.0f / y; },
                [](float x, float y) { return -x / (y * y); });
            return std::vector<std::vector<float>>{dx, dy};
        };

        // Slightly looser tolerance: division finite differences are sensitive near small denominators.
        require_gradient_check_passes("div broadcast grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 3.0e-3f, 3.5e-2f});
    }

    SECTION("pow") {
        const std::vector<GradientInput> inputs{
            {{2, 2}, {1.20f, 1.80f, 2.40f, 3.10f}},
            {{2, 2}, {0.50f, 1.40f, 1.90f, 2.20f}},
        };

        const std::vector<float> upstream = make_upstream(4, 0.25f, 0.07f, false);

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = cpptensor::pow(vars[0], vars[1]) * Tensor({2, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            const auto& base = vars[0].data();
            const auto& exponent = vars[1].data();

            std::vector<float> dbase(base.size(), 0.0f);
            std::vector<float> dexponent(exponent.size(), 0.0f);

            for (size_t i = 0; i < base.size(); ++i) {
                const float b = base[i];
                const float e = exponent[i];
                const float p = std::pow(b, e);
                dbase[i] = upstream[i] * e * std::pow(b, e - 1.0f);
                dexponent[i] = upstream[i] * p * std::log(b);
            }

            return std::vector<std::vector<float>>{dbase, dexponent};
        };

        // pow(base, exponent) combines exp/log style curvature; keep a slightly wider margin.
        require_gradient_check_passes("pow grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 4.5e-3f, 4.0e-2f});
    }
}

TEST_CASE("Gradient check: unary math ops", "[autograd][gradcheck]") {
    SECTION("exp") {
        const std::vector<GradientInput> inputs{
            {{2, 2}, {-0.70f, -0.10f, 0.40f, 1.00f}},
        };

        const std::vector<float> upstream = make_upstream(4, 0.14f, 0.09f, true);

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = cpptensor::exp(vars[0]) * Tensor({2, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            const auto& x = vars[0].data();
            return std::vector<std::vector<float>>{
                unary_elementwise_grad(x, upstream, [](float v) { return std::exp(v); }),
            };
        };

        require_gradient_check_passes("exp grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 2.0e-3f, 2.5e-2f});
    }

    SECTION("log") {
        const std::vector<GradientInput> inputs{
            {{2, 2}, {0.80f, 1.30f, 2.00f, 3.40f}},
        };

        const std::vector<float> upstream = make_upstream(4, 0.21f, 0.08f, true);

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = cpptensor::log(vars[0]) * Tensor({2, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            const auto& x = vars[0].data();
            return std::vector<std::vector<float>>{
                unary_elementwise_grad(x, upstream, [](float v) { return 1.0f / v; }),
            };
        };

        require_gradient_check_passes("log grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 2.5e-3f, 3.0e-2f});
    }

    SECTION("sqrt") {
        const std::vector<GradientInput> inputs{
            {{2, 2}, {0.50f, 1.10f, 2.50f, 4.00f}},
        };

        const std::vector<float> upstream = make_upstream(4, 0.13f, 0.06f, false);

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = cpptensor::sqrt(vars[0]) * Tensor({2, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            const auto& x = vars[0].data();
            return std::vector<std::vector<float>>{
                unary_elementwise_grad(x, upstream, [](float v) { return 0.5f / std::sqrt(v); }),
            };
        };

        require_gradient_check_passes("sqrt grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 2.5e-3f, 3.0e-2f});
    }
}

TEST_CASE("Gradient check: reductions", "[autograd][gradcheck]") {
    SECTION("global sum") {
        const std::vector<GradientInput> inputs{
            {{2, 3}, {0.20f, -0.50f, 1.40f, 2.00f, -1.10f, 0.70f}},
        };

        constexpr float upstream = 1.35f;

        const ScalarObjectiveFn objective = [](const std::vector<Tensor>& vars) {
            return scalar_value(vars[0].sum());
        };

        const AnalyticGradientFn analytic = [](const std::vector<Tensor>& vars) {
            return std::vector<std::vector<float>>{std::vector<float>(vars[0].numel(), 1.0f)};
        };

        // Objective is already scalar; this is the cleanest reduction baseline.
        require_gradient_check_passes("global sum grad", inputs,
                                      [upstream, objective](const std::vector<Tensor>& vars) {
                                          return upstream * objective(vars);
                                      },
                                      [upstream, analytic](const std::vector<Tensor>& vars) {
                                          auto grad = analytic(vars);
                                          for (float& v : grad[0]) {
                                              v *= upstream;
                                          }
                                          return grad;
                                      },
                                      GradientCheckTolerance{1.0e-3f, 2.0e-3f, 2.0e-2f});
    }

    SECTION("sum over dim with keepdim=false") {
        const std::vector<GradientInput> inputs{
            {{2, 3, 2}, {0.4f, -0.2f, 1.1f, 0.7f, -1.3f, 0.9f, 0.5f, 1.2f, -0.6f, 0.3f, 0.8f, -0.4f}},
        };

        const std::vector<float> upstream = make_upstream(4, 0.17f, 0.05f, true); // output shape {2,2}

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor reduced = cpptensor::sum(vars[0], 1, false); // {2,2}
            Tensor weighted = reduced * Tensor({2, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            std::vector<float> dx(vars[0].numel(), 0.0f);
            const auto in_shape = vars[0].shape();
            const std::vector<size_t> out_shape{in_shape[0], in_shape[2]};
            const auto out_strides = row_major_strides(out_shape);

            for (size_t i = 0; i < dx.size(); ++i) {
                const auto idx = unravel_index(i, in_shape);
                const size_t out_flat = idx[0] * out_strides[0] + idx[2] * out_strides[1];
                dx[i] = upstream[out_flat];
            }

            return std::vector<std::vector<float>>{dx};
        };

        require_gradient_check_passes("sum dim=1 grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 2.0e-3f, 2.5e-2f});
    }

    SECTION("mean over dim with keepdim=true") {
        const std::vector<GradientInput> inputs{
            {{2, 3, 2}, {0.9f, -0.1f, 0.4f, 1.5f, -0.8f, 0.6f, 1.1f, -0.7f, 0.2f, 0.3f, -1.0f, 1.4f}},
        };

        const std::vector<float> upstream = make_upstream(4, 0.19f, 0.04f, false); // output shape {2,1,2}

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor reduced = cpptensor::mean(vars[0], 1, true); // {2,1,2}
            Tensor weighted = reduced * Tensor({2, 1, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            std::vector<float> dx(vars[0].numel(), 0.0f);
            const auto in_shape = vars[0].shape();
            const auto out_shape = std::vector<size_t>{in_shape[0], 1, in_shape[2]};
            const auto out_strides = row_major_strides(out_shape);
            constexpr float reduction_size = 3.0f;

            for (size_t i = 0; i < dx.size(); ++i) {
                const auto idx = unravel_index(i, in_shape);
                const std::vector<size_t> out_idx{idx[0], 0, idx[2]};
                const size_t out_flat = out_idx[0] * out_strides[0] + out_idx[1] * out_strides[1] + out_idx[2] * out_strides[2];
                dx[i] = upstream[out_flat] / reduction_size;
            }

            return std::vector<std::vector<float>>{dx};
        };

        // keepdim=true + mean introduces an extra scale factor (1/reduction_size).
        require_gradient_check_passes("mean dim=1 keepdim grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 2.5e-3f, 3.0e-2f});
    }
}

TEST_CASE("Gradient check: matmul", "[autograd][gradcheck]") {
    SECTION("2D matmul") {
        const std::vector<GradientInput> inputs{
            {{2, 3}, {0.7f, -1.2f, 0.5f, 1.3f, 0.2f, -0.8f}},
            {{3, 2}, {0.9f, -0.3f, -1.1f, 0.4f, 0.6f, 1.2f}},
        };

        const std::vector<float> upstream = make_upstream(4, 0.11f, 0.06f, true); // output {2,2}

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = cpptensor::matmul(vars[0], vars[1]) * Tensor({2, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            const Tensor G({2, 2}, upstream);
            Tensor dA = cpptensor::matmul(G, vars[1].transpose());
            Tensor dB = cpptensor::matmul(vars[0].transpose(), G);
            return std::vector<std::vector<float>>{dA.data(), dB.data()};
        };

        require_gradient_check_passes("matmul 2D grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.0e-3f, 3.0e-3f, 3.5e-2f});
    }

    SECTION("batched matmul with broadcasted lhs batch") {
        const std::vector<GradientInput> inputs{
            {{1, 2, 3}, {0.6f, -0.9f, 1.1f, -0.5f, 0.4f, 0.8f}},
            {{2, 3, 2}, {1.0f, -0.7f, 0.3f, 0.9f, -1.2f, 0.5f, -0.4f, 1.3f, 0.8f, -0.6f, 0.2f, 0.7f}},
        };

        const std::vector<float> upstream = make_upstream(8, 0.10f, 0.03f, true); // output {2,2,2}

        const ScalarObjectiveFn objective = [upstream](const std::vector<Tensor>& vars) {
            Tensor weighted = cpptensor::matmul(vars[0], vars[1]) * Tensor({2, 2, 2}, upstream);
            return scalar_value(weighted.sum());
        };

        const AnalyticGradientFn analytic = [upstream](const std::vector<Tensor>& vars) {
            const Tensor G({2, 2, 2}, upstream);

            Tensor dA_broadcasted = cpptensor::matmul(G, vars[1].transpose(-1, -2)); // shape {2,2,3}
            Tensor dA = cpptensor::sum(dA_broadcasted, 0, true);                      // reduce broadcast batch -> {1,2,3}

            Tensor dB = cpptensor::matmul(vars[0].transpose(-1, -2), G);              // shape {2,3,2}

            return std::vector<std::vector<float>>{dA.data(), dB.data()};
        };

        // Broadcasting over batch dims introduces a reduction in dA (sum over expanded axis).
        require_gradient_check_passes("matmul batched broadcast grad", inputs, objective, analytic,
                                      GradientCheckTolerance{1.5e-3f, 4.0e-3f, 4.5e-2f});
    }
}

TEST_CASE("Gradient-check utility guards", "[autograd][gradcheck]") {
    SECTION("scalar_value enforces singleton tensors") {
        const Tensor vector({2}, {1.0f, 2.0f});
        REQUIRE_THROWS_WITH(scalar_value(vector), Catch::Matchers::ContainsSubstring("exactly one element"));
    }

    SECTION("non-finite numeric gradients fail checks") {
        const std::vector<GradientInput> inputs{
            {{1}, {-1.0f}},
        };

        const ScalarObjectiveFn objective = [](const std::vector<Tensor>& vars) {
            return scalar_value(cpptensor::sqrt(vars[0]));
        };

        const AnalyticGradientFn analytic = [](const std::vector<Tensor>& vars) {
            const auto& x = vars[0].data();
            return std::vector<std::vector<float>>{
                std::vector<float>{0.5f / std::sqrt(x[0])},
            };
        };

        const GradientCheckReport report = check_gradients(inputs, objective, analytic);
        REQUIRE_FALSE(report.passed());
        REQUIRE(report.failed > 0);
    }
}
