#ifndef UMUX_H_
#define UMUX_H_

#ifdef __AVX512F__

#include "unitary_calculator_avx512.h"

namespace clfsim {
    namespace unitary {
        template <typename For>
        using UnitaryCalculator = UnitaryCalculatorAVX512<For>;
    }
}
#elif __AVX2__
#include "unitary_calculator_avx.h"

namespace clfsim {
    namespace unitary {
        template <typename For>
        using UnitaryCalculator = UnitaryCalculatorAVX<For>;
    }
}
#elif __SSE4_1_
#include "unitary_calculator_sse.h"

namespace clfsim {
    namespace unitary {
        template <typename For>
        using UnitaryCalculator = UnitaryCalculatorSSE<For>;
    }
}
#else
#include "unitary_color_basic.h"

namespace clfsim {
    namespace unitary {
        template <typename For>
        using UnitaryCalculator = UnitaryCalculatorBasic<For>;
    }
}
#endif

#endif
