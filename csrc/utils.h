#pragma once

namespace mmq {

template <typename T>

T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

} // namespace mmq
