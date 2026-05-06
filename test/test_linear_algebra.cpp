#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cpptensor/ops/linearAlgebra/eig.hpp"
#include "cpptensor/ops/linearAlgebra/svd.hpp"
#include "cpptensor/tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace {

[[maybe_unused]] std::vector<float> materialize_matrix(const cpptensor::Tensor& tensor) {
    const auto& shape = tensor.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("materialize_matrix: expected rank-2 tensor");
    }

    const auto& stride = tensor.stride();
    const float* data_ptr = tensor.impl()->data_ptr();

    std::vector<float> matrix(shape[0] * shape[1]);
    for (size_t row = 0; row < shape[0]; ++row) {
        for (size_t col = 0; col < shape[1]; ++col) {
            matrix[row * shape[1] + col] = data_ptr[row * stride[0] + col * stride[1]];
        }
    }
    return matrix;
}

[[maybe_unused]] double frobenius_norm(const std::vector<float>& values) {
    double sum = 0.0;
    for (float value : values) {
        sum += static_cast<double>(value) * static_cast<double>(value);
    }
    return std::sqrt(sum);
}

[[maybe_unused]] double svd_relative_reconstruction_error(const cpptensor::Tensor& input,
                                                          const cpptensor::SVDResult& result) {
    const auto input_shape = input.shape();
    const size_t rows = input_shape[0];
    const size_t cols = input_shape[1];
    const size_t rank = result.S.shape()[0];

    const auto original = materialize_matrix(input);
    const auto u = materialize_matrix(result.U);
    const auto vt = materialize_matrix(result.Vt);

    std::vector<float> reconstructed(rows * cols, 0.0f);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            double value = 0.0;
            for (size_t k = 0; k < rank; ++k) {
                value += static_cast<double>(u[row * result.U.shape()[1] + k]) *
                         static_cast<double>(result.S.data()[k]) *
                         static_cast<double>(vt[k * cols + col]);
            }
            reconstructed[row * cols + col] = static_cast<float>(value);
        }
    }

    std::vector<float> diff(rows * cols, 0.0f);
    for (size_t i = 0; i < diff.size(); ++i) {
        diff[i] = reconstructed[i] - original[i];
    }

    return frobenius_norm(diff) / std::max(1.0, frobenius_norm(original));
}

[[maybe_unused]] double column_orthogonality_error(const cpptensor::Tensor& matrix) {
    const auto shape = matrix.shape();
    const auto data = materialize_matrix(matrix);

    double max_error = 0.0;
    for (size_t left = 0; left < shape[1]; ++left) {
        for (size_t right = 0; right < shape[1]; ++right) {
            double dot = 0.0;
            for (size_t row = 0; row < shape[0]; ++row) {
                dot += static_cast<double>(data[row * shape[1] + left]) *
                       static_cast<double>(data[row * shape[1] + right]);
            }
            const double target = (left == right) ? 1.0 : 0.0;
            max_error = std::max(max_error, std::abs(dot - target));
        }
    }
    return max_error;
}

[[maybe_unused]] double row_orthogonality_error(const cpptensor::Tensor& matrix) {
    const auto shape = matrix.shape();
    const auto data = materialize_matrix(matrix);

    double max_error = 0.0;
    for (size_t top = 0; top < shape[0]; ++top) {
        for (size_t bottom = 0; bottom < shape[0]; ++bottom) {
            double dot = 0.0;
            for (size_t col = 0; col < shape[1]; ++col) {
                dot += static_cast<double>(data[top * shape[1] + col]) *
                       static_cast<double>(data[bottom * shape[1] + col]);
            }
            const double target = (top == bottom) ? 1.0 : 0.0;
            max_error = std::max(max_error, std::abs(dot - target));
        }
    }
    return max_error;
}

[[maybe_unused]] double symmetric_eig_max_relative_residual(const cpptensor::Tensor& input,
                                                            const cpptensor::EigResult& result) {
    const auto shape = input.shape();
    const size_t n = shape[0];

    const auto a = materialize_matrix(input);
    const auto vectors = materialize_matrix(result.eigenvectors);

    double max_error = 0.0;
    for (size_t col = 0; col < n; ++col) {
        double residual_sq = 0.0;
        double vector_sq = 0.0;
        for (size_t row = 0; row < n; ++row) {
            double av = 0.0;
            for (size_t inner = 0; inner < n; ++inner) {
                av += static_cast<double>(a[row * n + inner]) *
                      static_cast<double>(vectors[inner * n + col]);
            }

            const double scaled = static_cast<double>(result.eigenvalues.data()[col]) *
                                  static_cast<double>(vectors[row * n + col]);
            const double residual = av - scaled;
            residual_sq += residual * residual;
            vector_sq += static_cast<double>(vectors[row * n + col]) *
                         static_cast<double>(vectors[row * n + col]);
        }

        max_error = std::max(
            max_error,
            std::sqrt(residual_sq) / std::max(1.0, std::sqrt(vector_sq))
        );
    }

    return max_error;
}

[[maybe_unused]] double general_eig_max_relative_residual(const cpptensor::Tensor& input,
                                                          const cpptensor::EigResult& result) {
    const auto shape = input.shape();
    const size_t n = shape[0];

    const auto a = materialize_matrix(input);
    const auto vr = materialize_matrix(result.eigenvectors);

    auto check_pair = [&](const std::complex<double>& lambda,
                          const std::vector<std::complex<double>>& vector) {
        double residual_sq = 0.0;
        double vector_sq = 0.0;

        for (size_t row = 0; row < n; ++row) {
            std::complex<double> av(0.0, 0.0);
            for (size_t inner = 0; inner < n; ++inner) {
                av += static_cast<double>(a[row * n + inner]) * vector[inner];
            }

            const std::complex<double> residual = av - lambda * vector[row];
            residual_sq += std::norm(residual);
            vector_sq += std::norm(vector[row]);
        }

        return std::sqrt(residual_sq) / std::max(1.0, std::sqrt(vector_sq));
    };

    double max_error = 0.0;
    for (size_t col = 0; col < n; ++col) {
        const double imag = result.eigenvalues_imag.data()[col];
        if (imag < 0.0) {
            continue;
        }

        if (std::abs(imag) < 1e-6) {
            std::vector<std::complex<double>> vector(n);
            for (size_t row = 0; row < n; ++row) {
                vector[row] = {static_cast<double>(vr[row * n + col]), 0.0};
            }
            max_error = std::max(
                max_error,
                check_pair({static_cast<double>(result.eigenvalues.data()[col]), 0.0}, vector)
            );
            continue;
        }

        if (col + 1 >= n ||
            std::abs(result.eigenvalues.data()[col] - result.eigenvalues.data()[col + 1]) > 1e-5f ||
            std::abs(result.eigenvalues_imag.data()[col + 1] + result.eigenvalues_imag.data()[col]) > 1e-5f) {
            throw std::runtime_error("general_eig_max_relative_residual: invalid LAPACK conjugate-pair packing");
        }

        std::vector<std::complex<double>> vector(n);
        std::vector<std::complex<double>> conjugate_vector(n);
        for (size_t row = 0; row < n; ++row) {
            const std::complex<double> real_part(vr[row * n + col], 0.0);
            const std::complex<double> imag_part(vr[row * n + col + 1], 0.0);
            vector[row] = real_part + std::complex<double>(0.0, 1.0) * imag_part;
            conjugate_vector[row] = real_part - std::complex<double>(0.0, 1.0) * imag_part;
        }

        max_error = std::max(
            max_error,
            check_pair(
                {static_cast<double>(result.eigenvalues.data()[col]),
                 static_cast<double>(result.eigenvalues_imag.data()[col])},
                vector
            )
        );
        max_error = std::max(
            max_error,
            check_pair(
                {static_cast<double>(result.eigenvalues.data()[col + 1]),
                 static_cast<double>(result.eigenvalues_imag.data()[col + 1])},
                conjugate_vector
            )
        );
    }

    return max_error;
}

} // namespace

#ifdef USE_OPENBLAS
TEST_CASE("svd reconstructs representative matrices and accepts non-contiguous views",
          "[linear-algebra][svd]") {
    cpptensor::Tensor source({4, 4},
                             {1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f,
                              2.0f, 1.0f, 0.0f, 3.0f,
                              4.0f, 1.0f, 5.0f, 2.0f});

    auto transpose_view = source.transpose();
    REQUIRE_FALSE(transpose_view.is_contiguous());

    const auto full = cpptensor::svd(transpose_view, true, true);
    REQUIRE(full.U.shape() == std::vector<size_t>{4, 4});
    REQUIRE(full.S.shape() == std::vector<size_t>{4});
    REQUIRE(full.Vt.shape() == std::vector<size_t>{4, 4});
    REQUIRE(svd_relative_reconstruction_error(transpose_view, full) < 5e-4);
    REQUIRE(column_orthogonality_error(full.U) < 5e-4);
    REQUIRE(row_orthogonality_error(full.Vt) < 5e-4);

    for (size_t i = 1; i < full.S.shape()[0]; ++i) {
        REQUIRE(full.S.data()[i - 1] >= full.S.data()[i]);
        REQUIRE(full.S.data()[i] >= -1e-6f);
    }

    auto sliced_view = source.slice(0, 1, 4).slice(1, 0, 3);
    REQUIRE_FALSE(sliced_view.is_contiguous());

    const auto reduced = cpptensor::svd(sliced_view, false, true);
    REQUIRE(reduced.U.shape() == std::vector<size_t>{3, 3});
    REQUIRE(reduced.S.shape() == std::vector<size_t>{3});
    REQUIRE(reduced.Vt.shape() == std::vector<size_t>{3, 3});
    REQUIRE(svd_relative_reconstruction_error(sliced_view, reduced) < 5e-4);
    REQUIRE(column_orthogonality_error(reduced.U) < 5e-4);
    REQUIRE(row_orthogonality_error(reduced.Vt) < 5e-4);

    const auto values_only = cpptensor::svd(sliced_view, false, false);
    REQUIRE(values_only.S.shape() == std::vector<size_t>{3});
    REQUIRE(values_only.U.shape() == std::vector<size_t>{0, 0});
    REQUIRE(values_only.Vt.shape() == std::vector<size_t>{0, 0});
    for (size_t i = 0; i < values_only.S.shape()[0]; ++i) {
        REQUIRE(values_only.S.data()[i] == Catch::Approx(reduced.S.data()[i]).margin(1e-5));
    }
}

TEST_CASE("eig_symmetric returns ascending eigenpairs with small residuals",
          "[linear-algebra][eig]") {
    cpptensor::Tensor source({5, 5},
                             {9.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                              1.0f, 8.0f, 2.0f, 0.0f, 0.0f,
                              0.0f, 2.0f, 7.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 6.0f, 2.0f,
                              0.0f, 0.0f, 0.0f, 2.0f, 5.0f});

    auto symmetric_view = source.slice(0, 1, 4).slice(1, 1, 4);
    REQUIRE_FALSE(symmetric_view.is_contiguous());

    const auto result = cpptensor::eig_symmetric(symmetric_view, true);
    REQUIRE(result.eigenvalues.shape() == std::vector<size_t>{3});
    REQUIRE(result.eigenvalues_imag.shape() == std::vector<size_t>{0});
    REQUIRE(result.eigenvectors.shape() == std::vector<size_t>{3, 3});

    for (size_t i = 1; i < result.eigenvalues.shape()[0]; ++i) {
        REQUIRE(result.eigenvalues.data()[i - 1] <= result.eigenvalues.data()[i]);
    }

    REQUIRE(column_orthogonality_error(result.eigenvectors) < 5e-4);
    REQUIRE(symmetric_eig_max_relative_residual(symmetric_view, result) < 5e-4);

    const auto values_only = cpptensor::eig_symmetric(symmetric_view, false);
    REQUIRE(values_only.eigenvalues.shape() == std::vector<size_t>{3});
    REQUIRE(values_only.eigenvectors.shape() == std::vector<size_t>{0, 0});
}

TEST_CASE("general eig exposes LAPACK ordering semantics and packed eigenvectors",
          "[linear-algebra][eig]") {
    cpptensor::Tensor matrix({3, 3},
                             {0.0f, -1.0f, 0.0f,
                              1.0f,  0.0f, 0.0f,
                              0.0f,  0.0f, 2.0f});

    const auto result = cpptensor::eig(matrix, true);
    REQUIRE(result.eigenvalues.shape() == std::vector<size_t>{3});
    REQUIRE(result.eigenvalues_imag.shape() == std::vector<size_t>{3});
    REQUIRE(result.eigenvectors.shape() == std::vector<size_t>{3, 3});
    REQUIRE(general_eig_max_relative_residual(matrix, result) < 5e-4);

    bool saw_complex_pair = false;
    for (size_t i = 0; i + 1 < result.eigenvalues_imag.shape()[0]; ++i) {
        if (result.eigenvalues_imag.data()[i] > 0.0f &&
            result.eigenvalues_imag.data()[i + 1] == -result.eigenvalues_imag.data()[i]) {
            saw_complex_pair = true;
            break;
        }
    }
    REQUIRE(saw_complex_pair);

    const auto values_only = cpptensor::eig(matrix, false);
    REQUIRE(values_only.eigenvectors.shape() == std::vector<size_t>{0, 0});
}

#endif

TEST_CASE("svd and eig reject unsupported input shapes", "[linear-algebra][errors]") {
    cpptensor::Tensor rank3({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    cpptensor::Tensor non_square({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    REQUIRE_THROWS_WITH(cpptensor::svd(rank3), Catch::Matchers::ContainsSubstring("input must be 2D"));
    REQUIRE_THROWS_WITH(cpptensor::eig(rank3), Catch::Matchers::ContainsSubstring("input must be 2D"));
    REQUIRE_THROWS_WITH(cpptensor::eig_symmetric(non_square),
                        Catch::Matchers::ContainsSubstring("matrix must be square"));
}
