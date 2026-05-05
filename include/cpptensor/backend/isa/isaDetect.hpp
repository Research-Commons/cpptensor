#pragma once

#include "cpptensor/enums/dispatcherEnum.h"

#include <cstdlib>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define CPPTENSOR_HAS_X86_CPUID 1
#else
#define CPPTENSOR_HAS_X86_CPUID 0
#endif

#if CPPTENSOR_HAS_X86_CPUID
    #if defined(_MSC_VER)
    #include <immintrin.h>
    #include <intrin.h>
    #else
    #include <cpuid.h>
    #endif
#endif

namespace cpptensor {
    inline bool has_avx2() {
    #if !CPPTENSOR_HAS_X86_CPUID
            return false;
    #elif defined(_MSC_VER)
            int info[4];
            __cpuid(info, 0);
            if (info[0] < 7) return false;
            __cpuidex(info, 7, 0);
            return (info[1] & (1 << 5)) != 0; // EBX bit5 AVX2
    #else
            unsigned int eax, ebx, ecx, edx;
            if (!__get_cpuid_max(0, 0) || __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
            return (ebx & (1u << 5)) != 0;
    #endif
        }

    inline bool has_avx512f() {
    #if !CPPTENSOR_HAS_X86_CPUID
            return false;
    #elif defined(_MSC_VER)
            int info[4];
            __cpuid(info, 0);
            if (info[0] < 7) return false;
            __cpuidex(info, 7, 0);
            return (info[1] & (1 << 16)) != 0; // EBX bit16 AVX512F
    #else
            unsigned int eax, ebx, ecx, edx;
            if (!__get_cpuid_max(0, 0) || __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
            return (ebx & (1u << 16)) != 0; // AVX-512F
    #endif
        }

    inline CpuIsa detect_best_cpu_isa() {
        // Optional: allow override via env
        if (const char* env = std::getenv("CPPGRAD_CPU_ISA")) {
            if (std::string(env) == "avx512") return CpuIsa::AVX512;
            if (std::string(env) == "avx2")   return CpuIsa::AVX2;
            return CpuIsa::GENERIC;
        }
        if (has_avx512f()) return CpuIsa::AVX512;
        if (has_avx2())    return CpuIsa::AVX2;
        return CpuIsa::GENERIC;
    }
}
#undef CPPTENSOR_HAS_X86_CPUID
