#pragma once
#include "stdint.h"

namespace mmq {

// get_npu_num_vec_cores
// 获得硬件 vec cores 的数量
// vec cores的数量和 AICores的数量不一定一致.
int64_t get_npu_num_vec_cores();

} // namespace mmq
