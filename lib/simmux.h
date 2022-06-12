#ifndef SIMMUX_H_
#define SIMMUX_H_

#ifdef __AVX512F__
# include "simulator_avx512.h"
  namespace clfsim {
    template <typename For>
    using Simulator = SimulatorAVX512<For>;
  }
#elif __AVX2__
# include "simulator_avx.h"
  namespace clfsim {
    template <typename For>
    using Simulator = SimulatorAVX<For>;
  }
#elif __SSE4_1__
# include "simulator_sse.h"
  namespace clfsim {
    template <typename For>
    using Simulator = SimulatorSSE<For>;
  }
#else
# include "simulator_basic.h"
  namespace clfsim {
    template <typename For>
    using Simulator = SimulatorBasic<For>;
  }
#endif

#endif  // SIMMUX_H_