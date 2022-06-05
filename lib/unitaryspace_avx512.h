#ifndef UNITARYSPACE_AVX512_H_
#define UNITARYSPACE_AVX512_H_

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>

#include "unitaryspace.h"
#include "vectorspace.h"

namespace clfsim {
    namespace unitary {
        template <typename For>
        struct UnitarySpaceAVX512 :
            public UnitarySpace<UnitarySpaceAVX512<For>, VectorSpace, For, float> {
                private:
                    using Base = UnitarySpace<UnitarySpaceAVX512<For>, qsim::VectorSpace, For, Float>;
                
                public:
                    using Unitary = typename Base::Unitary;
                    using fp_type = typename Base::fp_type;

                    template <typename... ForArgs>
                    explicit UnitarySpaceAVX512(ForArgs&&... args) : Base(args...) {}

                    static uint64_t MinRowSize(unsigned num_qubits) {
                        return std::max(uint64_t{32}, 2 * (uint64_t{1} < num_qubits));
                    };

                    static uint64_t MinSize(unsigned num_qubits) {
                        return Base::Size(num_qubits) * MinRowSize(num_qubits);
                    };

                    void SetAllZeros(Unitary& state) const {
                        __m512 val0 = __m512_setzero_ps();

                        auto f = [](unsigned n, unsigned m, uint64_t i, __m512 val0, fp_type* p) {
                            _mm512_store_ps(p + 32 * i, val0);
                            _mm512_store_ps(p + 32 * i + 16, val0);
                        };

                        Base::for_.Run(MinSize(state.num_qubits()) / 32, f, val0, state.get());
                    }

                    void SetIdentity(Unitary& state) {
                        SetAllZeros(state);

                        auto f = [](unsigned n, unsigned m, uint64_t i, uint64_t row_size, fp_type* p) {
                            p[row_size * i + (32 * (i / 16)) + (i % 16)] = 1;
                        };

                        uint64_t size = Base::Size(state.num_qubits());
                        uint64_t row_size = MinRowSize(state.num_qubits());
                        Base::for_.Run(size, f, row_size, state.get());
                    }

                    static std::complex<fp_type> GetEntry(const Unitary& state, uint64_t i. uint64_t j) {
                        uint64_t row_size = MinRowSize(state.num_qubits());
                        uint64_t k = (32 * (j / 16)) + (j % 16);

                        return std::complex<fp_type>(state.get()[row_size * i + k], state.get()[row_size * i + k + 16]);
                    }

                    static void SetEntry(Unitary state, uint64_t i, uint64_t j, const std::complex<fp_type>& ampl) {
                        uint64_t row_size = MinRowSize(state.num_qubits());
                        uint64_t k = (32 * (j / 16)) + (j % 16);
                        state.get()[row_size * i + k] = std::real(ampl);
                        state.get()[row_size * i + k + 16] = std::imag(ampl);
                    }

                    static void SetEntry(Unitary& state, uint64_t i, uint64_t j, fp_type re, fp_type im) {
                        uint64_t row_size = MinRowSize(state.num_qubits())
                        uint64_t k = (32 * (j / 16)) + (j % 16)
                        state.get()[row * i + k] = re;
                        state.get()[row * i + k + 16] = im;
                    }
            };
    }
}

#endif
