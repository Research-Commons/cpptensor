#include "cpptensor/backend/backend_loader.hpp"

#include "backend/cpu_backend.h"
#include "backend/cuda_backend.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/backend/isa/avx2.hpp"
#include "cpptensor/backend/isa/avx512.hpp"

namespace cpptensor {

    void initialize_kernels() {
        auto& R = KernelRegistry::instance();

        R.registerKernel(OpType::Add, DeviceType::CPU, CPU::addKernel);
        R.registerKernel(OpType::Mul, DeviceType::CPU, CPU::mulKernel);
        R.registerKernel(OpType::Sub, DeviceType::CPU, CPU::subKernel);
        R.registerKernel(OpType::Div, DeviceType::CPU, CPU::divKernel);
        R.registerKernel(OpType::Pow, DeviceType::CPU, CPU::powKernel);

        R.registerUnaryKernel(OpType::Exp, DeviceType::CPU, CPU::expKernel);
        R.registerUnaryKernel(OpType::Log, DeviceType::CPU, CPU::logKernel);
        R.registerUnaryKernel(OpType::Abs, DeviceType::CPU, CPU::absKernel);
        R.registerUnaryKernel(OpType::Sqrt, DeviceType::CPU,CPU::sqrtKernel);
        R.registerUnaryKernel(OpType::Sin, DeviceType::CPU, CPU::sinKernel);
        R.registerUnaryKernel(OpType::Cos, DeviceType::CPU, CPU::cosKernel);
        R.registerUnaryKernel(OpType::Tan, DeviceType::CPU, CPU::tanKernel);
        R.registerUnaryKernel(OpType::Sigmoid, DeviceType::CPU, CPU::sigmoidKernel);
        R.registerUnaryKernel(OpType::Relu, DeviceType::CPU, CPU::reluKernel);

        R.registerBackwardKernel(OpType::Add, DeviceType::CPU, CPU::addBackwardKernel);
        R.registerBackwardKernel(OpType::Mul, DeviceType::CPU, CPU::mulBackwardKernel);
        R.registerBackwardKernel(OpType::Sub, DeviceType::CPU, CPU::subBackwardKernel);
        R.registerBackwardKernel(OpType::Div, DeviceType::CPU, CPU::divBackwardKernel);

#ifdef BUILD_AVX2
        R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, add_f32_avx2);
        R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, mul_f32_avx2);
        R.registerKernel(OpType::Sub, DeviceType::CPU, CpuIsa::AVX2, sub_f32_avx2);
        R.registerKernel(OpType::Div, DeviceType::CPU, CpuIsa::AVX2, div_f32_avx2);
        R.registerKernel(OpType::Pow, DeviceType::CPU, CpuIsa::AVX2, pow_f32_avx2);

        R.registerUnaryKernel(OpType::Exp, DeviceType::CPU, CpuIsa::AVX2,exp_f32_avx2);
        R.registerUnaryKernel(OpType::Log, DeviceType::CPU, CpuIsa::AVX2,log_f32_avx2);
        R.registerUnaryKernel(OpType::Abs, DeviceType::CPU, CpuIsa::AVX2,abs_f32_avx2);
        R.registerUnaryKernel(OpType::Sqrt,DeviceType::CPU, CpuIsa::AVX2,sqrt_f32_avx2);
        R.registerUnaryKernel(OpType::Sin, DeviceType::CPU, CpuIsa::AVX2,sin_f32_avx2);
        R.registerUnaryKernel(OpType::Cos, DeviceType::CPU, CpuIsa::AVX2,cos_f32_avx2);
        R.registerUnaryKernel(OpType::Tan, DeviceType::CPU, CpuIsa::AVX2,tan_f32_avx2);
        R.registerUnaryKernel(OpType::Sigmoid, DeviceType::CPU, CpuIsa::AVX2,sigmoid_f32_avx2);
        R.registerUnaryKernel(OpType::Relu, DeviceType::CPU, CpuIsa::AVX2,relu_f32_avx2);
#endif

#ifdef BUILD_AVX512
        R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX512, cpptensor::add_f32_avx512);
        R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX512, cpptensor::mul_f32_avx512);
#endif

#ifdef BUILD_CUDA
        R.registerKernel(OpType::Add, DeviceType::CUDA, CUDA::addKernel);
        R.registerKernel(OpType::Mul, DeviceType::CUDA, CUDA::mulKernel);
#endif

        std::cout << "[cppgrad] Kernel registry initialized.\n";
    }

} // namespace cppgrad
