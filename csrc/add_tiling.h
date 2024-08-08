#pragma once
#include <stdint.h>

// MMQAddTiling
// Add相关的tiling数据结构
// 因为 Ascend只能暴露C接口，故
// 1. 该文件没有namespace，使用MMQ前缀来标记namespace
struct MMQAddTiling {
  // tiling_size_ 最大的计算块元素数量。
  uint32_t tiling_size_;

  // block_size_ 每个block需要处理的元素数量
  // block和cuda的block概念类似。
  // 但与cuda block概念不同的是，ascend没有warp scheduler，
  // 即如果计算stalling，不能调度不同的计算到计算核心中。
  // 故 通常ascend的block数量和硬件计算核心数量一致。
  uint32_t block_size_;

  // last_block_size_ 最后一个block需要处理的元素数量
  // 通常因为数据不是完全对齐的，最后一个block需要处理的元素
  // 数量与其他block不一致。
  uint32_t last_block_size_;
};
