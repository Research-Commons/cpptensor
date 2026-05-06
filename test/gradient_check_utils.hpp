#pragma once

#include "cpptensor/tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cpptensor::test {

struct GradientInput {
    std::vector<size_t> shape;
    std::vector<float> values;
};

struct GradientCheckTolerance {
    float epsilon = 1.0e-3f;
    float atol = 2.0e-3f;
    float rtol = 2.0e-2f;
};

struct GradientFailure {
    size_t input_index = 0;
    size_t element_index = 0;
    float analytic = 0.0f;
    float numeric = 0.0f;
    float abs_error = 0.0f;
    float tolerance = 0.0f;
};

struct GradientCheckReport {
    size_t compared = 0;
    size_t failed = 0;
    float max_abs_error = 0.0f;
    float max_tolerance = 0.0f;
    std::vector<GradientFailure> failures;

    [[nodiscard]] bool passed() const {
        return failed == 0;
    }

    [[nodiscard]] std::string summary() const {
        std::ostringstream out;
        out << "compared=" << compared
            << ", failed=" << failed
            << ", max_abs_error=" << max_abs_error
            << ", max_tolerance=" << max_tolerance;

        if (!failures.empty()) {
            const auto& first = failures.front();
            out << ", first_failure=(input=" << first.input_index
                << ", element=" << first.element_index
                << ", analytic=" << first.analytic
                << ", numeric=" << first.numeric
                << ", abs_error=" << first.abs_error
                << ", tol=" << first.tolerance
                << ')';
        }

        return out.str();
    }
};

using ScalarObjectiveFn = std::function<float(const std::vector<Tensor>&)>;
using AnalyticGradientFn = std::function<std::vector<std::vector<float>>(const std::vector<Tensor>&)>;

inline size_t numel(const std::vector<size_t>& shape) {
    size_t total = 1;
    for (size_t d : shape) {
        total *= d;
    }
    return total;
}

inline std::vector<size_t> row_major_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 0);
    if (shape.empty()) {
        return strides;
    }

    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
    }

    return strides;
}

inline std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& shape) {
    std::vector<size_t> index(shape.size(), 0);
    if (shape.empty()) {
        return index;
    }

    const auto strides = row_major_strides(shape);
    for (size_t dim = 0; dim < shape.size(); ++dim) {
        if (shape[dim] == 0) {
            index[dim] = 0;
            continue;
        }
        index[dim] = flat / strides[dim];
        flat %= strides[dim];
    }
    return index;
}

inline size_t ravel_index(const std::vector<size_t>& index, const std::vector<size_t>& strides) {
    size_t flat = 0;
    for (size_t dim = 0; dim < index.size(); ++dim) {
        flat += index[dim] * strides[dim];
    }
    return flat;
}

inline std::vector<size_t> broadcast_shape(const std::vector<size_t>& lhs, const std::vector<size_t>& rhs) {
    const size_t rank = std::max(lhs.size(), rhs.size());
    std::vector<size_t> out(rank, 1);

    for (size_t i = 0; i < rank; ++i) {
        const size_t ldim = (i < rank - lhs.size()) ? 1 : lhs[i - (rank - lhs.size())];
        const size_t rdim = (i < rank - rhs.size()) ? 1 : rhs[i - (rank - rhs.size())];

        if (ldim != rdim && ldim != 1 && rdim != 1) {
            throw std::runtime_error("broadcast_shape: incompatible inputs");
        }

        out[i] = std::max(ldim, rdim);
    }

    return out;
}

inline size_t broadcast_offset(const std::vector<size_t>& out_index,
                               const std::vector<size_t>& in_shape,
                               const std::vector<size_t>& in_strides) {
    if (in_shape.empty()) {
        return 0;
    }

    const size_t out_rank = out_index.size();
    const size_t in_rank = in_shape.size();
    const size_t shift = out_rank - in_rank;

    std::vector<size_t> index(in_rank, 0);
    for (size_t d = 0; d < in_rank; ++d) {
        const size_t aligned_dim = d + shift;
        index[d] = (in_shape[d] == 1) ? 0 : out_index[aligned_dim];
    }

    return ravel_index(index, in_strides);
}

inline std::vector<float> reduce_sum_to_shape(const std::vector<float>& out_grad,
                                              const std::vector<size_t>& out_shape,
                                              const std::vector<size_t>& in_shape) {
    std::vector<float> reduced(numel(in_shape), 0.0f);
    const auto in_strides = row_major_strides(in_shape);

    const size_t total = numel(out_shape);
    for (size_t out_flat = 0; out_flat < total; ++out_flat) {
        const auto out_index = unravel_index(out_flat, out_shape);
        const size_t in_flat = broadcast_offset(out_index, in_shape, in_strides);
        reduced[in_flat] += out_grad[out_flat];
    }

    return reduced;
}

inline std::vector<Tensor> make_tensors(const std::vector<GradientInput>& inputs,
                                        DeviceType device = DeviceType::CPU) {
    std::vector<Tensor> tensors;
    tensors.reserve(inputs.size());

    for (const auto& input : inputs) {
        const size_t expected = numel(input.shape);
        if (input.values.size() != expected) {
            throw std::runtime_error("make_tensors: values.size() != numel(shape)");
        }
        tensors.emplace_back(input.shape, input.values, device);
    }

    return tensors;
}

inline GradientCheckReport check_gradients(const std::vector<GradientInput>& inputs,
                                           const ScalarObjectiveFn& objective,
                                           const AnalyticGradientFn& analytic,
                                           GradientCheckTolerance tolerance = {},
                                           size_t max_failures_to_store = 8) {
    if (tolerance.epsilon <= 0.0f) {
        throw std::runtime_error("check_gradients: epsilon must be positive");
    }

    const std::vector<Tensor> base_tensors = make_tensors(inputs);
    const std::vector<std::vector<float>> analytic_grads = analytic(base_tensors);

    if (analytic_grads.size() != inputs.size()) {
        throw std::runtime_error("check_gradients: analytic gradient count mismatch");
    }

    GradientCheckReport report;

    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        const size_t n = numel(inputs[input_idx].shape);
        if (analytic_grads[input_idx].size() != n) {
            throw std::runtime_error("check_gradients: analytic gradient size mismatch for input");
        }

        for (size_t elem_idx = 0; elem_idx < n; ++elem_idx) {
            auto plus_inputs = inputs;
            auto minus_inputs = inputs;

            plus_inputs[input_idx].values[elem_idx] += tolerance.epsilon;
            minus_inputs[input_idx].values[elem_idx] -= tolerance.epsilon;

            const float f_plus = objective(make_tensors(plus_inputs));
            const float f_minus = objective(make_tensors(minus_inputs));
            const float numeric = (f_plus - f_minus) / (2.0f * tolerance.epsilon);
            const float analytic_value = analytic_grads[input_idx][elem_idx];
            const bool finite_inputs = std::isfinite(f_plus) && std::isfinite(f_minus);
            const bool finite_grads = std::isfinite(numeric) && std::isfinite(analytic_value);

            const float abs_error = std::fabs(numeric - analytic_value);
            const float scaled = std::max(std::fabs(analytic_value), std::fabs(numeric));
            const float threshold = tolerance.atol + tolerance.rtol * scaled;
            const bool finite_error_terms = std::isfinite(abs_error) && std::isfinite(threshold);

            report.compared += 1;
            report.max_abs_error = std::max(report.max_abs_error, abs_error);
            report.max_tolerance = std::max(report.max_tolerance, threshold);

            const bool failed = !finite_inputs || !finite_grads || !finite_error_terms || (abs_error > threshold);
            if (failed) {
                report.failed += 1;
                if (report.failures.size() < max_failures_to_store) {
                    report.failures.push_back(GradientFailure{
                        input_idx,
                        elem_idx,
                        analytic_value,
                        numeric,
                        abs_error,
                        threshold,
                    });
                }
            }
        }
    }

    return report;
}

inline float scalar_value(const Tensor& tensor) {
    const auto& data = tensor.data();
    if (data.size() != 1) {
        throw std::runtime_error("scalar_value: expected tensor with exactly one element");
    }
    return data[0];
}

inline std::vector<float> make_upstream(size_t n,
                                        float base = 0.15f,
                                        float step = 0.07f,
                                        bool alternating_sign = true) {
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float v = base + step * static_cast<float>(i % 13);
        if (alternating_sign && (i % 2 == 1)) {
            v = -v;
        }
        out[i] = v;
    }
    return out;
}

} // namespace cpptensor::test
