#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "cpptensor/tensor/tensor.hpp"

#include <functional>

using Catch::Matchers::ContainsSubstring;

namespace {

void require_uninitialized_tensor_error(const char* method_name,
                                        const std::function<void()>& action) {
    INFO(std::string("Expecting Tensor::") + method_name + " to reject an uninitialized tensor");
    REQUIRE_THROWS_WITH(action(),
                        ContainsSubstring(std::string("Tensor::") + method_name +
                                          ": tensor is uninitialized"));
}

} // namespace

TEST_CASE("default-constructed tensor methods fail with deterministic exceptions", "[tensor][guard]") {
    cpptensor::Tensor tensor;

    require_uninitialized_tensor_error("shape", [&] { (void)tensor.shape(); });
    require_uninitialized_tensor_error("numel", [&] { (void)tensor.numel(); });
    require_uninitialized_tensor_error("clone", [&] { (void)tensor.clone(); });
    require_uninitialized_tensor_error("slice", [&] { (void)tensor.slice(0); });
}
