#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/tensor/tensor.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;

namespace {

std::filesystem::path make_temp_checkpoint_path() {
    static std::atomic<uint64_t> counter{0};
    const uint64_t suffix = counter.fetch_add(1, std::memory_order_relaxed);
    const auto ticks = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return std::filesystem::temp_directory_path() /
           ("cpptensor_tensor_checkpoint_" + std::to_string(ticks) + "_" +
            std::to_string(suffix) + ".cpt");
}

void require_tensor_data(const cpptensor::Tensor& tensor,
                         const std::vector<float>& expected) {
    const auto& values = tensor.data();
    REQUIRE(values.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(values[i] == Approx(expected[i]));
    }
}

} // namespace

TEST_CASE("tensor save/load round-trips contiguous tensors with metadata",
          "[tensor][serialization]") {
    const std::filesystem::path checkpoint = make_temp_checkpoint_path();

    cpptensor::Tensor original({2, 3}, {1, 2, 3, 4, 5, 6}, DeviceType::CPU);
    original.save(checkpoint.string());
    cpptensor::Tensor loaded = cpptensor::Tensor::load(checkpoint.string());

    REQUIRE(loaded.shape() == std::vector<size_t>{2, 3});
    REQUIRE(loaded.device_type() == DeviceType::CPU);
    REQUIRE(loaded.is_contiguous());
    require_tensor_data(loaded, {1, 2, 3, 4, 5, 6});

    std::filesystem::remove(checkpoint);
}

TEST_CASE("tensor serialization materializes view tensors on save",
          "[tensor][serialization][views]") {
    const std::filesystem::path checkpoint = make_temp_checkpoint_path();

    cpptensor::Tensor base({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor transposed = base.transpose();
    REQUIRE_FALSE(transposed.is_contiguous());

    transposed.save(checkpoint.string());
    cpptensor::Tensor loaded = cpptensor::Tensor::load(checkpoint.string());

    REQUIRE(loaded.shape() == std::vector<size_t>{3, 2});
    REQUIRE(loaded.is_contiguous());
    require_tensor_data(loaded, {1, 4, 2, 5, 3, 6});

    std::filesystem::remove(checkpoint);
}

TEST_CASE("tensor save/load round-trips scalar and large tensors",
          "[tensor][serialization]") {
    const std::filesystem::path scalar_checkpoint = make_temp_checkpoint_path();
    cpptensor::Tensor scalar({}, std::vector<float>{42.5f});
    scalar.save(scalar_checkpoint.string());
    cpptensor::Tensor scalar_loaded = cpptensor::Tensor::load(scalar_checkpoint.string());
    REQUIRE(scalar_loaded.shape().empty());
    REQUIRE(scalar_loaded.numel() == 1);
    REQUIRE(scalar_loaded.data()[0] == Approx(42.5f));
    std::filesystem::remove(scalar_checkpoint);

    const std::filesystem::path large_checkpoint = make_temp_checkpoint_path();
    std::vector<float> values(200000);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((i % 4096) * 0.125f);
    }

    cpptensor::Tensor large({400, 500}, values);
    large.save(large_checkpoint.string());
    cpptensor::Tensor large_loaded = cpptensor::Tensor::load(large_checkpoint.string());

    REQUIRE(large_loaded.shape() == std::vector<size_t>{400, 500});
    REQUIRE(large_loaded.numel() == values.size());
    const auto& loaded_data = large_loaded.data();
    REQUIRE(loaded_data.front() == Approx(values.front()));
    REQUIRE(loaded_data[12345] == Approx(values[12345]));
    REQUIRE(loaded_data.back() == Approx(values.back()));

    std::filesystem::remove(large_checkpoint);
}

TEST_CASE("tensor load rejects corrupted and incompatible-version checkpoints",
          "[tensor][serialization]") {
    const std::filesystem::path corrupted_checkpoint = make_temp_checkpoint_path();
    {
        std::ofstream out(corrupted_checkpoint, std::ios::binary | std::ios::trunc);
        out.write("not-a-checkpoint", 16);
    }

    REQUIRE_THROWS_WITH(cpptensor::Tensor::load(corrupted_checkpoint.string()),
                        ContainsSubstring("not a cpptensor checkpoint"));
    std::filesystem::remove(corrupted_checkpoint);

    const std::filesystem::path wrong_version_checkpoint = make_temp_checkpoint_path();
    cpptensor::Tensor t({2}, {3.0f, 4.0f});
    t.save(wrong_version_checkpoint.string());

    {
        std::fstream io(wrong_version_checkpoint, std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(io.is_open());
        io.seekp(8, std::ios::beg); // version field starts after 8-byte magic
        const char incompatible_version[2] = {static_cast<char>(0xFF), static_cast<char>(0x7F)};
        io.write(incompatible_version, 2);
    }

    REQUIRE_THROWS_WITH(cpptensor::Tensor::load(wrong_version_checkpoint.string()),
                        ContainsSubstring("unsupported checkpoint version"));
    std::filesystem::remove(wrong_version_checkpoint);
}
