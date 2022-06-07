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
        
        inline uint32_t CompressBits(uint32_t bits, unsigned n, uint32_t mask) {
            return _pext_u32(bits, mask);
        }

        inline uint64_t CompressBits(uint64_t bits, unsigned n, uint64_t mask) {
            return _pext_u64(bits, mask);
        }
    }
}

#else

namespace clfsim {
    namespace bits {
        template <typename Integer>
        inline Integer PermuteBits(
            Integer bits, unsigned n, const std::vector<unsigned>& perm
        ) {
            Integer pbits = 0;

            for (unsigned i = 0; i < n; ++i) {
                pbits |= ((bits >> i) & 1) << perm[i];
            }

            return pbits
        }
    }
}

#endif