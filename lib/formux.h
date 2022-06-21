#ifndef FORMUX_H_
#define FORMUX_H_

#ifdef _OPENMP
# include "parfor.h"
  namespace clfsim {
    using For = ParallelFor;
  }
#else
# include "seqfor.h"
  namespace clfsim {
    using For = SequentialFor;
  }
#endif

