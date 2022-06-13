#ifndef IO_H_
#define IO_H_

#include <cstdarg>
#include <cstdio>

namespace clfsim {

/**
 * Controller for output logs.
 */
struct IO {
  static void errorf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
  }

  static void messagef(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
  }
};

}  // namespace clfsim

#endif  // IO_H_