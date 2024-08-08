#pragma once
#include "torch/extension.h"

namespace mmq {

// add out = x + y
void add(torch::Tensor out, torch::Tensor x, torch::Tensor y);

} // namespace mmq
