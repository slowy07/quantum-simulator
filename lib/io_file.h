ifndef IO_FILE_H_
#define IO_FILE_H_

#include <cstdint>
#include <fstream>
#include <string>

#include "io.h"

namespace clfsim {

/**
 * Controller for output logs with methods for writing to file.
 */
struct IOFile : public IO {
  static std::ifstream StreamFromFile(const std::string& file) {
    std::ifstream fs;
    fs.open(file);
    if (!fs) {
      errorf("cannot open %s for reading.\n", file.c_str());
    }
    return fs;
  }

  static void CloseStream(std::ifstream& fs) {
    fs.close();
  }

  static bool WriteToFile(
      const std::string& file, const std::string& content) {
    return WriteToFile(file, content.data(), content.size());
  }

  static bool WriteToFile(
      const std::string& file, const void* data, uint64_t size) {
    auto fs = std::fstream(file, std::ios::out | std::ios::binary);

    if (!fs) {
      errorf("cannot open %s for writing.\n", file.c_str());
      return false;
    } else {
      fs.write((const char*) data, size);
      if (!fs) {
        errorf("cannot write to %s.\n", file.c_str());
        return false;
      }

      fs.close();
    }

    return true;
  }
};

}  // namespace clfsim

#endif  // IO_FILE_H_