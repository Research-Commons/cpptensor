#include "cppgrad/backend/backend_loader.hpp"

#include "backend/cpu_backend.h"
#include "backend/cuda_backend.hpp"
#include "cppgrad/dispatcher/kernelRegistry.h"
#include "cppgrad/enums/dispatcherEnum.h"
#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/backend/isa/avx2.hpp"
#include "cppgrad/backend/isa/avx512.hpp"

namespace cppgrad {

    void initialize_kernels() {
        auto& R = KernelRegistry::instance();

        R.registerKernel(OpType::Add, DeviceType::CPU, CPU::addKernel);
        R.registerKernel(OpType::Mul, DeviceType::CPU, CPU::mulKernel);
        R.registerKernel(OpType::Sub, DeviceType::CPU, CPU::subKernel);
        R.registerKernel(OpType::Div, DeviceType::CPU, CPU::divKernel);

        R.registerBackwardKernel(OpType::Add, DeviceType::CPU, CPU::addBackwardKernel);
        R.registerBackwardKernel(OpType::Mul, DeviceType::CPU, CPU::mulBackwardKernel);
        R.registerBackwardKernel(OpType::Sub, DeviceType::CPU, CPU::subBackwardKernel);
        R.registerBackwardKernel(OpType::Div, DeviceType::CPU, CPU::divBackwardKernel);

#ifdef BUILD_AVX2
        R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cppgrad::add_f32_avx2);
        R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cppgrad::mul_f32_avx2);
#endif

#ifdef BUILD_AVX512
        R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX512, cppgrad::add_f32_avx512);
        R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX512, cppgrad::mul_f32_avx512);
#endif

#ifdef BUILD_CUDA
        R.registerKernel(OpType::Add, DeviceType::CUDA, CUDA::addKernel);
        R.registerKernel(OpType::Mul, DeviceType::CUDA, CUDA::mulKernel);
#endif

        std::cout << "[cppgrad] Kernel registry initialized.\n";
    }

} // namespace cppgrad
