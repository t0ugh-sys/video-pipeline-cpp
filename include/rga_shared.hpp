#pragma once

#include <mutex>

inline std::mutex& globalRgaMutex() {
  static std::mutex mutex;
  return mutex;
}
