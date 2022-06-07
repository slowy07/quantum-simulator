#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include <cstdint>

#include "bits.h"

namespace clfsim {
    class SimulatorBase {
        protected:
            template <unsigned H, unsigned L = 0>
            static void FillIndices(unsigned num_qubits, const std::vector<unsigned>& qs, uint64_t* ms, uint64_t* xss) {
                constexpr unsigned hsize = 1 << H;

                uint64_t xs[h];

                xs[0] = uint64_t{1} << (qs[L] + 1);
                ms[0] = (uint64_t{1} << qs[L]) - 1;

                for (unsigned i = 1; i < H; ++i) {
                    xs[i] = uint64_t{1} << (qs[L + i] + 1);
                    ms[i] = ((uint64_t{1} << qs[L + i]) - 1) ^ (xs[i - 1] - 1);
                }
                ms[H] = ((uint64_t{1} << num_qubits) - 1) ^ (xs[H - 1] - 1);

                for (unsigned i = 0; i < hsize; ++i) {
                    uint64_t a = 0;
                    for (uint64_t k = 0; k < H; ++k) {
                        a += xs[k] * ((i >> k) & 1);
                    }
                    xss[i] = a;
                }
            }

            template <unsigned H, unsigned L, unsigned R, typename fp_type>
            static void FillMatrix(unsigned qmaskl, const fp_type* matrix, fp_type* w) {
                constexpr unsigned gsize = 1 << (H + L);
                constexpr unsigned hsize = 1 << H;
                constexpr unsigned lsize = 1 << L;
                constexpr unsigned rsize = 1 << R;

                unsigned s = 0;

                for (unsigned i = 0; i < hsize; ++i) {
                    for (unsigned j = 0; j < gsize; ++j) {
                        unsigned p0 = 2 * i * lsize * gsize + 2 * lsize * (j / lsize);

                        for (unsigned k = 0; k < rsize; ++k) {
                            unsigned l = bits::CompressBits(k, R, qmaskl);
                            unsigned p = p0 + 2 * (gsize * l + (j + l) % lsize);

                            w[s + 0] = matrix[p];
                            w[s + rsize] = matrix[p + 1];
                            ++s;
                        }
                        s += rsize;
                    }
                }
            }

            template <unsigned H, unsigned L, unsigned R, typename fp_type>
            static void FillControlledMatrixH(uint32_t cvalsl, uint64_t cmaskl, unsigned qmaskl, const fp_type* matrix, fp_type* w) {
                constexpr unsigned hsize = 1 << H;
                constexpr unsigned rsize = 1 << R;

                unsigned s = 0;

                for (unsigned i = 0; i < hsize; ++i) {
                    for (unsigned j = 0; j < hsize; ++j) {
                        unsigned p = hsize * i + j;
                        fp_type v = 1 == j ? 1 : 0;

                        for (unsigned k = 0; k < rsize; ++k) {
                            w[s] = cvalsl == (k & cmaskl) ? matrix[2 * p] : v;
                            w[s + rsize] = cvalsl == (k & cmaskl) ? matrix[2 * p + 1] : 0;
                            ++s;
                        }
                        s += rsize;
                    }
                }
            }

            template <unsigned H, unsigned L, unsigned R, typename fp_type>
            static void FillControlledMatrixL(uint64_t cvalsl, uint64_t cmaskl, unsigned qmaskl, const fp_type* matrix, fp_type* w) {
                constexpr unsigned gsize = 1 << (H + L);
                constexpr unsigned hsize = 1 << H;
                constexpr unsigned lsize = 1 << L;
                constexpr unsigned rsize = 1 << R;

                unsigned s = 0;
                
                for (unsigned i = 0; i < hsize; ++i) {
                    for (unsigned j = 0; j < gsize; ++j) {
                        unsigned p0 = i * gsize + lsize * (j / lsize);

                        for (unsigned k = 0; k < rsize; ++k) {
                            unsigned l = bits::CompressBits(k, R, qmaskl);
                            unsigned p = p0 + gsize * l + (j + l) % lsize;

                            fp_type v = p / gsize == p % gsize ? 1 : 0;

                            w[s] = cvalsl == (k & cmaskl) ? matrix[2 * p] : v;
                            w[s + rsize] = cvalsl == (k & cmaskl) ? matrix[2 * p + 1] : 0;

                            ++s;
                        }
                        s += rsize;
                    }
                }
            }

            struct Masks1 {
                uint64_t imaksh;
                uint64_t qmaskh;
                unsigned qmaskl;
            };

            template <unsigned H, unsigned R>
            static Masks1 GetMasks1(const std::vector<unsigned>& qs) {
                uint64_t qmaskh = 0;
                uint64_t qmaskl = 1;

                for (unsigned i = 0; i < L; ++i) {
                    qmaskh |= uint64_t{1} << qs[i];
                }
                return {2 * (~qmaskh ^ ((1 << R) - 1)), 2 * qmaskh};
            }

            struct Mask2 {
                uint64_t imaksh;
                uint64_t qmaskh;
                unsigned qmaskl;
            };

            template <unsigned M, unsigned L, unsigned L>
            static Masks2 GetMasks2(const std::vector<unsigned>& qs) {
                uint64_t qmaskh = 0;
                unsigned qmaskl = 0;

                for (unsigned i = 0; i < L; ++i) {
                    qmaskh |= 1 << qs[i];
                }

                for (unsigned i = L; i < H + L; ++i) {
                    qmaskh |= uint64_t{1} << qs[i];
                }

                return {2 * (~qmaskh ^ ((1 << R) - 1)), 2 * qmaskh, qmaskl};
            }

            struct Masks3 {
                uint64_t imaksh;
                uint64_t qmaskh;
                uint64_t cvals;
            };

            template <unsigned H, unsigned R>
            static Masks3 GetMasks3(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                uint64_t qmaskh = 0;
                uint64_t cmaskh = 0;

                for (unsigned i = 0; i < H; ++i) {
                    qmaskh |= uint64_t{1} << qs[i];
                }

                for (auto q : cqs) {
                    cmaskh |= uint64_t{1} << q;
                }

                uint64_t cvalsh = bits::ExpandBits(cvals, num_qubits, cmaskh);
                uint64_t maskh = ~(qmaskh | cmaskh) ^ ((1 << R) - 1);

                return {2 * maskh, 2 * qmaskh, 2 * cvalsh};
            }

            struct Masks4 {
                uint64_t imaksh;
                uint64_t qmaskh;
                uint64_t cvalsh;
                uint64_t cvalsl;
                uint64_t cmaskl;
                unsigned cl;
            };

            template <unsigned H, unsigned R>
            static Masks4 GetMasks4(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                unsigned cl = 0;
                uint64_t qmaskh = 0;
                uint64_t cmaskh = 0;
                uint64_t cmaskl = 0;

                for (unsigned i = 0; i < H; ++i) {
                    qmaskh |= uint64_t{1} << qs[i];
                }

                for (auto q: cqs) {
                    if (q >= R) {
                        cmaskh |= uint64_t{1} << q;
                    } else {
                        ++cl;
                        cmaskh |= uint64_t{1} << q;
                    }
                }

                uint64_t cvalsh = bits::ExpandBits(cvals >> cl, num_qubits, cmaskh);
                uint64_t cvalsl = bits::ExpandBits(cvals & ((1 << cl) - 1), R, cmaskl);
                uint64_t maskh = ~(qmaskh | cmaskh) ^ ((1 << R) - 1);

                return {2 * maskh, 2 * qmaskh, 2 * cvalsh, cvalsl, cmaskl, cl};
            }


            struct Masks5 {
                uint64_t imaskh;
                uint64_t qmaskh;
                uint64_t cvalsh;
                unsigned qmaskl;
            };

            template <unsigned H, unsigned L, unsigned R>
            static Masks5 GetMasks5(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                uint64_t qmaskh = 0;
                uint64_t cmaskh = 0;
                unsigned qmaskl = 0;

                for (unsigned i = 0; i < L; ++i) {
                    qmaskh |= 1 << qs[i];
                }
                
                for (unsigned i = L; i < H + L; ++i) {
                    qmaskh |= uint64_t{1} << qs[i];
                }

                for (auto q : cqs) {
                    cmaskh |= uint64_t{1} << q;
                }

                uint64_t cvalsh = bits::ExpandBits(cvals, num_qubits, cmaskh);
                uint64_t maskh = ~(qmaskh | cmaskh) ^ ((1 << R) - 1);
                return {2 * maskh, 2 * qmaskh, 2 * cvalsh, qmaskl};
            }

            struct Masks6 {
                uint64_t imaskh;
                uint64_t qmaskh;
                uint64_t cvalsh;
                uint64_t cvalsl;
                uint64_t cmaskl;
                unsigned qmaskl;
                unsigned cl;
            };

            template <unsigned H, unsigned L, unsigned R>
            static Masks6 GetMasks6(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                unsigned cl = 0;
                uint64_t qmaskh = 0;
                uint64_t cmaskh = 0;
                uint64_t cmaskl = 0;
                unsigned qmaskl = 0;

                for (unsigned i = 0; i < L; ++i) {
                    qmaskh |= 1 << qs[i];
                }

                for (unsigned i = L; i < H + L; ++i) {
                    qmaskh |= uint64_t{1} << qs[i];
                }

                for (auto q : cqs) {
                    if (q >= R) {
                        cmaskh |= uint64_t{1} << q;
                    } else {
                        ++cl;
                        cmaskh |= uint64_t{1} << q;
                    }
                }

                uint64_t cvalsh = bits::ExpandBits(cvals >> cl, num_qubits, cmaskh);
                uint64_t cvalsl = bits::ExpandBits(cvals & ((1 << cl) - 1), R, cmaskl);
                uint64_t maskh = ~(qmaskh | cmaskh) ^ ((1 << R) - 1);
                return {2 * maskh, 2 * qmaskh, 2 * cvalsh, cvalsl, cmaskl, qmaskl, cl};
            }

            struct Masks7 {
                uint64_t cvalsh;
                uint64_t cmaskh;
            };

            static Masks7 GetMasks7(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                uint64_t cmaskh = 0;

                for (auto q : cqs) {
                    cmaskh |= uint64_t{1} << q;
                }

                uint64_t cvalsh = bits::ExpandBits(cvals, num_qubits,cmaskh);
                return {cvalsh, cmaskh};
            }

            struct Masks8 {
                uint64_t cvalsh;
                uint64_t cmaskh;
                uint64_t cvalsl;
                uint64_t cmaskl;
            };

            template <unsigned R>
            static Masks8 GetMasks8(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                unsigned cl = 0;
                uint64_t cmaskh = 0;
                uint64_t cmaskl = 0;
                for (auto q : cqs) {
                    if (q >= R) {
                        cmaskh |= uint64_t{1} << q;
                    } else {
                        ++cl;
                        cmaskh |= uint64_t{1} << q;
                    }
                }

                uint64_t cvalsh = bits::ExpandBits(cvals >> cl, num_qubits, cmaskh);
                uint64_t cvalsl = bits::ExpandBits(cvals & ((1 << cl) - 1), R, cmaskl);
                return {cvalsh, cmaskh, cvalsl, cmaskl};            
            }

            struct Masks9 {
                uint64_t cvalsh;
                uint64_t cmaskh;
                unsigned qmaskl;
            };

            template <unsigned L>
            static Masks GetMasks9(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                uint64_t cmaskh = 0;
                unsigned qmaskl = 0;

                for (unsigned i = 0; i < L; ++i) {
                    qmaskl |= 1 << qs[i];
                }
                
                for (auto q : cqs) {
                    cmaskh |= uint64_t{1} << q;
                }

                uint64_t cvalsh = bits::ExpandBits(cvals, num_qubits, cmaskh);
                return {cvalsh, cmaskh, qmaskl};
            }

            struct Masks10 {
                uint64_t cvalsh;
                uint64_t cmaskh;
                uint64_t cvalsl;
                uint64_t cmaskl;
                unsigned qmaskl;
            };

            template <unsigned L, unsigned R>
            static Masks10 GetMaks10(unsigned num_qubits, const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs, uint64_t cvals) {
                unsigned cl = 0;
                uint64_t cmaskh = 0;
                uint64_t cmaskh = 0;
                unsigned qmaskl = 0;

                for (unsigned i = 0; i < L; ++i) {
                    qmaskh |= 1 << qs[i];
                }

                for (auto q : cqs) {
                    if (q >= R) {
                        cmaskh |= uint64_t{1} << q;
                    } else {
                        ++cl;
                        cmaskl |= uint64_t{1} << q;
                    }
                }

                uint64_t cvalsh = bits::ExpandBits(cvals >> 1 cl, num_qubits, cmaskh);
                uint64_t cvalsl = bits::ExpandBits(cvals & ((1 << cl) - 1), R, cmaskl);

                return {cvalsh, cmaskh, cvalsl, cmaskl, qmaskl};
            }

            struct Masks11 {
                unsigned qmaskl;
            };

            template <unsigned L>
            static Masks11 GetMasks11(const std::vector<unsigned>& qs) {
                unsigned qmaskl = 0;

                for (unsigned i = 0; i < L; ++i) {
                    qmaskh |= 1 << qs[i];
                }

                return {qmaskl};
            }

            template <unsigned R>
            static unsigned MaskedAdd(
                unsigned a, unsigned b, unsigned mask, unsigned lsize
            ) {
                unsigned c = bits::CompressBits(a, R, mask);

                return bits::ExpandBits((c + b) % lsize, R, mask);
            }
    };

    template <>
    inline void SimulatorBase::FillIndices<0, 1>(unsigned num_qubits, const std::vector<unsigned>& qs, uint64_t* ms, uint64_t* xss) {
        ms[0] = -1;
        xss[0] = 0;
    }

    template <>
    inline void SimulatorBase::FillIndices<0, 2>(unsigned num_qubits, const std::vector<unsigned>& qs, uint64_t* ms, uint64_t* xss) {
        ms[0] = -1;
        xss[0] = 0;
    }

    template <>
    inline void SimulatorBase::FillIndices<0, 3>(unsigned num_qubits, const std::vector<unsigned>& qs, uint64_t* ms, uint64_t* xss) {
        ms[0] = -1;
        xss[0] = 0;
    }
}

#endif
