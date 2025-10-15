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

        R.registerBackwardKernel(OpType::Add, DeviceType::CPU, CPU::addBackwardKernel);
        R.registerBackwardKernel(OpType::Mul, DeviceType::CPU, CPU::mulBackwardKernel);
        R.registerBackwardKernel(OpType::Sub, DeviceType::CPU, CPU::subBackwardKernel);
        R.registerBackwardKernel(OpType::Div, DeviceType::CPU, CPU::divBackwardKernel);

#ifdef BUILD_AVX2
        R.registerKernel(OpType::Add, DeviceType::CPU, CpuIsa::AVX2, cpptensor::add_f32_avx2);
        R.registerKernel(OpType::Mul, DeviceType::CPU, CpuIsa::AVX2, cpptensor::mul_f32_avx2);
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
