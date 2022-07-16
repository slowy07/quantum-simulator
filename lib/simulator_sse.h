#ifndef SIMULATOR_SSE_H_
#define SIMULATOR_SSE_H_

#include <smmintrin.h>
#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

#include "simulator.h"
#include "statespace_sse.h"

namespace clfsim {
  template <typename For>
  class SimulatorSSE final : public SimulatorBase {
    public:
      using StateSpace = StateSpaceSSE<For>;
      using State = typename StateSpace::State;
      using fp_type = typename StateSpace::fp_type;

      template <typename... ForArgs>
      explicit SimulatorSSE(ForArgs&&... args) : for_(args...) {}

      void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix, State& state) const {
        switch (qs.size()) {
          case 1:
            if (qs[0] > 1) {
              ApplyGateH<1>(qs, matrix, state);
            } else {
              ApplyGateL<0, 1>(qs, matrix, state);
            }
            break;
          case 2:
            if (qs[0] > 1) {
              ApplyGateH<2>(qs, matrix, state);
            } else if (qs[1] > 1) {
              ApplyGateL<1, 1>(qs, matrix, state);
            } else {
              ApplyGateL<0, 2>(qs, matrix, state);
            }
            break;
          case 3:
            if (qs[0] > 1) {
              ApplyGateH<3>(qs, matrix, state);
            } else if (qs[1] > 1) {
              ApplyGateL<2, 1>(qs, matrix, state);
            } else {
              ApplyGateL<1, 2>(qs, matrix, state);
            }
            break;
          case 4:
            if (qs[0] > 1) {
              ApplyGateH<4>(qs, matrix, state);
            } else if (qs[1] > 1) {
              ApplyGateL<3, 1>(qs, matrix, state);
            } else {
              ApplyGateL<2, 2>(qs, matrix, state);
            }
            break;
          case 5:
            if (qs[0] > 1) {
              ApplyGateH<5>(qs, matrix, state);
            } else if (qs[1] > 1) {
              ApplyGateL<4, 1>(qs, matrix, state);
            } else {
              ApplyGateL<3, 2>(qs, matrix, state);
            }
            break;
          case 6:
            if (qs[0]) > 1) {
              ApplyGateH<6>(qs, matrix, state);
            } else if (qs[1] > 1) {
              ApplyGateL<5, 1>(qs, matrix, state);
            } else {
              ApplyGateL<4, 2>(qs, matrix, state);
            }
            break;
          default:
            // not implemented
            break;
        }
      }
      /**
       * applies a controlled gate using sse instruction
       * @param qs indices of the qubit affected by this gate
       * @param cqs indices of control qubits
       * @param cvals bit mask of control qubit values
       * @param matrix representation of the gate to be applied
       * @param state state of the system
       */
      void ApplyControlledGate(const std::vector<unsigned>& qs,
          const std::vector<unsigned>& cqs, uint64_t cvals,
          const fp_type* matrix, State& state) const {
        if (cqs.size() == 0) {
          ApplyGate(qs, matrix, state);
          return;
        }

        switch(qs.size()) {
          case 1:
            if (qs[0] > 1) {
              if (cqs[0] > 1) {
                ApplyControlledGateHH<1>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateHL<1>(qs, cqs, cvals, matrix, state);
              }
            } else {
              if (cqs[0] > 1) {
                ApplyControlledGateL<0, 1, 1>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateL<0, 1, 0>(qs, cqs, cvals, matrix, state);
              }
            }
            break;
          case 2:
            if (qs[0] > 1) {
              if (cqs[0] > 1) {
                ApplyControlledGateHH<2>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateHL<2>(qs, cqs, cvals, matrix, state);
              }
            } else {
              if (cqs[0] > 1) {
                ApplyControlledGateL<0, 2, 1>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateL<0, 2, 0>(qs, cqs, cvals, matrix, state);
              }
            }
            break;
          case 3:
            if (qs[0] > 1) {
              if (cqs[0] > 1) {
                ApplyControlledGateHH<3>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateHL<3>(qs, cqs, cvals, matrix, state);
              }
            } else if (qs[1] > 1) {
              if (cqs[0] > 1) {
                ApplyControlledGateL<2, 1, 1>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateL<2, 1, 0>(qs, cqs, cvals, matrix, state);
              }
            } else {
              if (cqs[0] > 1) {
                ApplyControlledGateL<1, 2, 1>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateL<1, 2, 0>(qs, cqs, cvals, matrix, state);
              }
            }
            break;
          case 4:
            if (qs[0] > 1) {
              if (cqs[0] > 1) {
                ApplyControlledGateHH<4>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateHL<4>(qs, cqs, cvals, matrix, state);
              }
            } else if (qs[1] > 1) {
              if (cqs[0] > 1) {
                ApplyControlledGateL<3, 1, 1>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateL<3, 1, 0>(qs, cqs, cvals, matrix, state);
              }
            } else {
              if (cqs[0] > 1) {
                ApplyControlledGateL<2, 2, 1>(qs, cqs, cvals, matrix, state);
              } else {
                ApplyControlledGateL<2, 2, 0>(qs, cqs, cvals, matrix, state);
              }
            }
            break;
          default:
            // not implemented
            break;
        }
      }
      
      /**
       * Computes the expectation value of an operator using SSE
       * @param qs indices of the qubits the operator acts on 
       * @param matrix the operator matrix
       * @param state state of the system
       * @return the computed expectation value
       */
      std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
          const fp_type* matrix,
          const State& state) const {
        switch (qs.size()) {
          case 1:
            if (qs[0] > 1) {
              return ExpectationValueH<1>(qs, matrix, state);
            } else {
              return ExpectationValueL<0, 1>(qs, matrix, state);
            }
            break; 
          case 2:
            if (qs[0] > 1) {
              return ExpectationValueH<2>(qs, matrix, state);
            } else if (qs[1] > 1) {
              return ExpectationValueL<1, 1>(qs, matrix, state);
            } else {
              return ExpectationValue<0, 2>(qs, matrix, state);
            }
            break;
          case 3:
            if (qs[0] > 1) {
              return ExpectationValueH<3>(qs, matrix, state);
            } else if (qs[1] > 1) {
              return ExpectationValueL<2,1>(qs, matrix, state);
            } else {
              return ExpectationValueL<1, 2>(qs, matrix, state);
            }
            break;
          case 4:
            if (qs[0] > 1) {
              return ExpectationValueH<4>(qs, matrix, state);
            } else if (qs[1] > 1) {
              return ExpectationValueL<3, 1>(qs, matrix, state);
            } else {
              return ExpectationValueL<2, 2>(qs, matrix, state);
            }
            break;
          case 5:
            if (qs[0] > 1) {
              return ExpectationValueH<5>(qs, matrix, state);
            } else if (qs[1] > 1) {
              return ExpectationValueL<4, 1>(qs, matrix, state);
            } else {
              return ExpectationValueL<3, 2>(qs, matrix, state);
            }
            break;
          case 6:
            if (qs[0] > 1) {
              return ExpectationValueH<6>(qs, matrix, state);
            } else if (qs[1] > 1) {
              return ExpectationValueL<5, 1>(qs, matrix, state);
            } else {
              return ExpectationValueL<4, 2>(qs, matrix, state);
            }
            break;
          default:
            // not implemented
            break;
        }
        return 0;
      }

      /**
       * @return the size of SIMD
       */

      static unsigned SIMDRegisteredSize() {
        return 4;
      }

    private:
      template<unsigned H>
      void ApplyGateH(const std::vector<unsigned>& qs,
          const fp_type* matrix, State& state) const {
        auto f = [](unsigned n, uint64_t i , const fp_type *v,
            const uint64_t* ms, const uint64_t* xss, fp_type* rstate) {
          constexpr unsigned hsize = 1 << H;

          __m128 ru, iu, rn, in;
          __m128 rs[hsize], is[hsize];

          i *= 4;
          uint64_t ii = i & ms[0];
          for (unsigned j = 1; j <= H; ++j) {
            i *= 2;
            ii |= i & ms[j];
          }
          auto p0 = rstate + 2 * ii;
          
          for (unsigned k = 0; k < hsize; ++k) {
            rs[k] = _mm_load_ps(p0 + xss[k]);
            is[k] = _mm_load_ps(p0 + xsss[k] + 4);
          }

          uint64_t j = 0;

          for (unsigned k = 0; k < hsize; ++k) {
            ru = _mm_set1_ps(v[j]);
            iu = _mm_set1_ps(v[j + 1]);
            rn = _mm_mul_ps(rs[0], ru);
            in = _mm_sub_ps(rs[0], iu);
            rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], iu));
            in = _mm_add_ps(in, _mm_mul_ps(is[0], ru));

            j += 2;
          }

          _mm_store_ps(p0 + xss[k], rn);
          _mm_store_ps(p0 + xss[k] + 4, in);
        }
      };

      uint64_t ms[H + 1];
      uint64_t xss[1 << H];
      
      FillIndices<H>(state.num_qubits(), qs, ms, xss);
      
      unsigned k = 2 + H;
      unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
      uint64_t size = uint64_t{1} << n;

      for_.Run(size, f, matrix, ms, xss, state.get());
  }
  
  template <unsigned H, unsigned L>
  void ApplyGateL(const std::vector<unsigned>& qs,
      const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const __m128* w,
        const uint64_t * ms, const uint64_t* xss, unsigned q0,
        fp_type* rstate) {
      constexpr unsigned gsize = 1 << (H + L);
      constexpr unsigned hsize = 1 << H;
      constexpr unsigned lsize = 1 << L;

      __m128 run, in;
      __m128 rs[gsize], is[gsize];
      i *= 4;

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        unsigned k2 = lsize * k;

        rs[k2] = _mm_load_ps(p0 + xss[k]);
        is[k2] = _mm_load_ps(p0 + xss[k] + 4);

        if (L == 1) {
          rs[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(rs[k2], rs[k2], 177)
            : _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(is[k2], is[k2], 177)
            : _mm_shuffle_ps(is[k2], is[k2], 78);
        } else if (L == 2) {
          rs[k2 + 1] = _mm_shuffle_ps(rs[k2], rs[k2], 57);
          is[k2 + 1] = _mm_shuffle_ps(is[k2], is[k2], 57);
          rs[k2 + 2] = _mm_shuffle_ps(rs[k2], rs[k2], 78);
          is[k2 + 2] = _mm_shuffle_ps(rs[k2], rs[k2], 78);
          rs[k2 + 3] = _mm_shuffle_ps(is[k2], is[k2], 147);
          rs[k2 + 3] = _mm_shuffle_ps(rs[k2], rs[k2], 147);
        }
      }
      
      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = _mm_mul_ps(rs[0], w[j]);
        in = _mm_mul_ps(rs[0], w[j + 1]);
        rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], w[j + 1]));
        in = _mm_add_ps(in, _mm_mul_ps(is[0], w[j]));

        j += 2;
        
        for (unsigned l = 1; l < gsize; ++l) {
          rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], w[j]));
          in = _mm_add_ps(in, _mm_mul_ps(rs[l], w[j + 1]));
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], w[j + 1]));
          in = _mm_add_ps(in, _mm_mul_ps(is[l], w[j]));
          
          j += 2;
        }

        _mm_store_ps(p0 + xss[k], rn);
        _mm_store_ps(p0 + xss[k] + 4, in);
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];
    __m128 w[1 << (1 + 2 * H + L)];

    auto m = GetMask11<L>(qs);

    FillIndices<H, L>(state.num_qubits(), qs, ms, xss);
    FillMatrix<H, L, 2>(q.qmask1, matrix, (fp_type*) w);
    
    unsigned k = 2 + H;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, w, ms, xss, qs[0], state.get());

    template <unsigned H>
    void ApplyControlledGateHH(const std::vector<unsigned>& qs,
        const std::vector<unsigned>& cqs, uint64_t cvals,
        const fp_type* matrix, State& state) const {
      auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
          const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
          uint64_t cmaksh, fp_type* rstate) {
        constexpr unsigned hsize = 1 << H;

        __m128 ru, iu, rn, in;
        __m128 rs[hsize], is[hsize];

        i *= 4;
        
        uint64_t ii = i & ms[0];
        for (unsigned j = 1; j <= H; ++j) {
          i *= 2;
          ii |= i & ms[j];
        }
        if ((ii & cmaksh) != cvals) return;

        auto p0 = rstate + 2 * ii;
        
        for (unsigned k = 0; k < hsize; ++k) {
          rs[k] = _mm_load_ps(p0 + xss[k]);
          is[k] = _mm_load_ps(p0 + xss[k] + 4);
        }

        uint64_t j = 0;

        for (unsigned k = 0; k < hsize; ++k) {
          ru = _mm_set1_ps(v[j]);
          iu = _mm_set1_ps(v[j + 1]);
          rn = _mm_sub_ps(rn, _mm_mul_ps(is[0], w[j + 1]));
          in = _mm_add_ps(rn, _mm_mul_ps(is[0], w[j]));
          j += 2;

          for (unsigned l = 1; l < hsize; ++l) {
            rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], w[j]));
            in = _mm_add_ps(in, _mm_mul_ps(rs[l], w[j + 1]));
            rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], w[j + 1]));
            in = _mm_add_ps(in, _mm_mul_ps(is[l], w[j]));

            j += 2;
          }
          _mm_store_ps(p0 + xss[k], rn);
          _mm_store_ps(p0 + xss[k] + 4, in);
        }
      };

      uint64_t ms[H + 1];
      uint64_t xss[1 << H];
      __m128 w[1 << (1 + 2 * H)];

      auto m = GetMasks8<2>(state.num_qubits(), qs, cqs, cvals);
      FillIndices<H>(state.num_qubits(), qs, ms, xss);
      FillControlledMatrix<H, 2>(m.cvalsl, m.cmask1, matrix, (fp_type*) w);

      unsigned r = 2 + H;
      unsigned n = state.num_qubits() > r ? state.num_qubits() - r : 0;
      uint64_t size = uint64_t{1} << n;

      for_.Run(size, f, w, ms, xss, m.cvalsl, m.cmaksh, state.get());

      template <unsigned H, unsigned L, bool CH>
      void ApplyControlledGateL(const std::vector<unsigned>& qs,
          const std::vector<unsigned>& cqs, uint64_t cvals,
          const fp_type* matrix, State& state) const {
        auto f = [](unsigned n, unsigned m, uint64_t i, const __m128* w,
            const uint64_t* ms, const uint64_t* xss, uint64_t cvalsh,
            const cmaksh, unsigned q0, fp_type* rstate) {
          constexpr unsigned gsize = 1 << (H + L);
          constexpr unsigned hsize = 1 << H;
          constexpr unsigned lsize = 1 << L;

          __m128 rn, in;
          __m128 rs[gsize], is[gsize];

          i *= 4;
          uint64_t ii = i & ms[0];
          for (unsigned j = 1; j <= H; ++j) {
            i *= 2;
            ii |= i & ms[j];
          }

          if ((ii & cmaksh) != cvalsh) return;
          
          auto p0 = rstate + 2 * ii;

          for (unsigned k = 0; k < hsize; ++k) {
            unsigned k2 = lsize * k;
            rs[k2] = _mm_load_ps(p0 + xss[k]);
            rs[k2] = _mm_load_ps(p0 + xss[k] + 4);

            if (L == 1) {
              rs[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(rs[k2], rs[k2], 177)
                : _mm_shuffle_ps(rs[k2], rs[k2], 78);
              is[k2 + 1] = q0 == 0 ? _mm_shuffle_ps(is[k2], is[k2], 177)
                : _mm_shuffle_ps(is[k2], is[k2], 78);
            } else if (L == 2) {
              rs[k2 + 1] = _mm_shuffle_ps(rs[k2], rs[k2], 57);
              is[k2 + 1] = _mm_shuffle_ps(is[k2], is[k2], 57);
              rs[k2 + 2] = _mm_shuffle_ps(rs[k2], rs[k2], 78);
              is[k2 + 2] = _mm_shuffle_ps(is[k2], is[k2], 78);
              rs[k2 + 3] = _mm_shuffle_ps(rs[k2], rs[k2], 147);
              is[k2 + 3] = _mm_shuffle_ps(is[k2], is[k2], 147);
            }
          }
          uint64_t j = 0;

          for (unsigned k = 0; k < hsize; ++k) {
            rn = _mm_add_ps(rn, _mm_mul_ps(rs[l], w[j]));
            in = _mm_add_ps(in, _mm_mul_ps(rs[l], w[j + 1]));
            rn = _mm_sub_ps(rn, _mm_mul_ps(is[l], w[j + 1]));
            in = _mm_add_ps(in, _mm_mul_ps(is[l], w[j]));

            j += 2;
          }
          _mm_store_ps(p0 + xss[k], rn);
          _mm_store_ps(p0 + xss[k] + 4, in);
        }
      };

      uint64_t ms[H + 1];
      uint64_t xss[1 << H];
      __m128 w[1 << (1 + 2 * H + L)];
      FillIndices<H, L>(state.num_qubits(), qs, ms, xss);

      unsigned r = 2 + H;
      unsigned n = state.num_qubits() > r ? state.num_qubits() - r : 0;
      uint64_t size = uint64_t{1} << n;

      if (CH) {
        auto m = GateMasks9<L>(state.num_qubits(), qs, cqs, cvalsh);
        FillControlledMatrix<H, L, 2>(
            m.cvalsl, c.cmask1, m.qmask1, matrix, (fp_type*) w);
        for_.Run(size, f, w, ms, xss, m.cvalsh, m.cmaksh, qs[0], state.get());
      }
    }

    template <unsigned H>
    std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
        const fp_type* matrix, const State& state) const {
      // TODO 683
    }
  }
}
