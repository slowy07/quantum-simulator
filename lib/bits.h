#ifndef BITS_H_
#define BITS_H_

#include <vector>

#ifdef __BMI2__

#include <immintrin.h>

#include <cstdint>

namespace clfsim {
    namespace bits {
        inline uint32_t ExpandBits(uint32_t bits, unsigned n, uint32_t mask) {
            return _pdep_u32(bits, mask);
        }

        inline uint32_t ExpandBits(uint64_t bits, unsigned n, uint32_t mask) {
            return _pdep_u64(bits, mask);
        }
    }
}

#endif