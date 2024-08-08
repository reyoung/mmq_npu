#include "add_tiling.h"
#include "kernel_operator.h"
#include <variant>

namespace mmq {

static constexpr uint32_t kBufferSize = 2;
template <typename T> class Add {
  // GMPtr 带类型的global memory pointer
  using GMPtr = __gm__ T *;

  // TBuffer 如果不是bfloat16，其实不需要Buffer来做类型转换。
  // 所以当不是bfloat16的时候，TBuffer为一个空struct。
  // 避免在模板实例化的时候，分配多余的寄存器
  using TBuffer = std::conditional_t<
      std::is_same_v<T, bfloat16_t>,
      // 其实Position在Ascend里的设计很不科学。
      //  他同时包含两个含义
      //    * Queue的输入输出内存位置
      //    * Buffer的内存位置
      //  这里VECCALC，即local shared memory，等价于cuda的shared memory
      AscendC::TBuf<AscendC::TPosition::VECCALC>,
      // std::monosate就是一个空的struct。这里可以替换成任何空struct，
      // 没有任何额外语意。
      std::monostate>;

public:
  __aicore__ inline Add(GMPtr x, GMPtr y, GMPtr z,
                        const MMQAddTiling &tiling_data)
      : tiling_data_(tiling_data) {
    int64_t blockIdx = AscendC::GetBlockIdx();
    bool is_last_block = (blockIdx + 1) == AscendC::GetBlockNum();
    block_size_ = tiling_data_.block_size_;
    if (is_last_block) {
      block_size_ = tiling_data_.last_block_size_;
    }

    auto offset = blockIdx * tiling_data_.block_size_;
    global_x_.SetGlobalBuffer(x + offset, block_size_);
    global_y_.SetGlobalBuffer(y + offset, block_size_);
    global_z_.SetGlobalBuffer(z + offset, block_size_);

    pipe_.InitBuffer(x_q_, kBufferSize, tiling_data_.tiling_size_ * sizeof(T));
    pipe_.InitBuffer(y_q_, kBufferSize, tiling_data_.tiling_size_ * sizeof(T));
    pipe_.InitBuffer(z_q_, kBufferSize, tiling_data_.tiling_size_ * sizeof(T));

    if constexpr (std::is_same_v<T, bfloat16_t>) { // vec core does not support
                                                   // bfloat16, cast to fp32
      pipe_.InitBuffer(x_buffer_, tiling_data_.tiling_size_ * sizeof(float));
      pipe_.InitBuffer(y_buffer_, tiling_data_.tiling_size_ * sizeof(float));
    }
  }

  __aicore__ inline void operator()() {
    uint32_t num_steps = (block_size_ + tiling_data_.tiling_size_ - 1) /
                         tiling_data_.tiling_size_;

    for (uint32_t i = 0; i < num_steps; ++i) {
      step(i, tiling_data_.tiling_size_);
    }

    uint32_t last_tiling_size =
        block_size_ - (num_steps - 1) * tiling_data_.tiling_size_;

    step(num_steps - 1, last_tiling_size);
  }

private:
  __aicore__ inline void copy_in(uint32_t i, uint32_t n) {
    // 注意这里，理论上不应该先处理x，再处理y
    // 即 不应该写成
    //   auto x = x_q_.alloc()
    //   copy(x)
    //   x_q_.enqueue(x)
    //   auto y = y_q_.alloc()
    //   copy(y)
    //   y_q_.enqueue(y)
    //
    // 因为，其实Queue在Ascend中的实现是向对应的处理器发送SetEvent/WaitEvent操作
    // WaitEvent的真实含义是等待之前所有的内存操作完毕。
    // 所以，如下写法，实际上硬件只同步了一次(也就是x_q_.EnQue(x)的时候，
    // 同时等待了y的DataCopy)
    // 这样会快很多

    auto x = x_q_.AllocTensor<T>();
    auto y = y_q_.AllocTensor<T>();

    AscendC::DataCopy(x, global_x_[i * tiling_data_.tiling_size_], n);
    AscendC::DataCopy(y, global_y_[i * tiling_data_.tiling_size_], n);

    x_q_.EnQue(x);
    y_q_.EnQue(y);
  }

  __aicore__ inline void copy_out(uint32_t i, uint32_t n) {
    AscendC::LocalTensor<T> z = z_q_.DeQue<T>();
    AscendC::DataCopy(global_z_[i * tiling_data_.tiling_size_], z, n);
    z_q_.FreeTensor(z);
  }

  __aicore__ inline void process(uint32_t i, uint32_t n) {
    auto x = x_q_.DeQue<T>();
    auto y = y_q_.DeQue<T>();
    auto z = z_q_.AllocTensor<T>();

    if constexpr (!std::is_same_v<T, bfloat16_t>) {
      AscendC::Add(z, x, y, n);
    } else {
      // Vec Core没有bfloat16计算能力，需要先转型成float，计算，再转型回去。
      // 因为整个操作都是发生在VecCore之上的，所以不需要额外的同步操作
      auto x_tmp = x_buffer_.template Get<float>();
      auto y_tmp = y_buffer_.template Get<float>();

      AscendC::Cast(x_tmp, x, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Cast(y_tmp, y, AscendC::RoundMode::CAST_NONE, n);

      AscendC::Add(x_tmp, x_tmp, y_tmp, n);

      AscendC::Cast(z, x_tmp, AscendC::RoundMode::CAST_RINT, n);
    }

    x_q_.FreeTensor(x);
    y_q_.FreeTensor(y);
    z_q_.EnQue(z);
  }

  __aicore__ inline void step(uint32_t i, uint32_t n) {
    copy_in(i, n);
    process(i, n);
    copy_out(i, n);
  }

  const MMQAddTiling &tiling_data_;
  uint32_t block_size_{};
  AscendC::TPipe pipe_;
  AscendC::GlobalTensor<T> global_x_;
  AscendC::GlobalTensor<T> global_y_;
  AscendC::GlobalTensor<T> global_z_;

  AscendC::TQue<AscendC::TPosition::VECIN, kBufferSize> x_q_;
  AscendC::TQue<AscendC::TPosition::VECIN, kBufferSize> y_q_;
  AscendC::TQue<AscendC::TPosition::VECOUT, kBufferSize> z_q_;

  TBuffer x_buffer_;
  TBuffer y_buffer_;
};

template <typename T>
static __aicore__ inline void add_npu(__gm__ T *x, __gm__ T *y, __gm__ T *z,
                                      const MMQAddTiling &tiling_data) {
  Add<T> add(x, y, z, tiling_data);
  add();
}

} // namespace mmq

// 因为ascend的kernel必须是一个C函数，所以
// 1. C没有namespace，所以加上前缀 `mmq_npu_` 避免名字冲突
// 2. C没有overwrite，所以加上后缀 `_type`区分不同类型的函数
//
// 该接口只是从 C -> C++
extern "C" __global__ __aicore__ void
mmq_npu_add_float(GM_ADDR x, GM_ADDR y, GM_ADDR z, MMQAddTiling tiling_data) {
  mmq::add_npu<float>(reinterpret_cast<__gm__ float *>(x),
                      reinterpret_cast<__gm__ float *>(y),
                      reinterpret_cast<__gm__ float *>(z), tiling_data);
}
extern "C" __global__ __aicore__ void
mmq_npu_add_half(GM_ADDR x, GM_ADDR y, GM_ADDR z, MMQAddTiling tiling_data) {
  mmq::add_npu<half>(reinterpret_cast<__gm__ half *>(x),
                     reinterpret_cast<__gm__ half *>(y),
                     reinterpret_cast<__gm__ half *>(z), tiling_data);
}
extern "C" __global__ __aicore__ void
mmq_npu_add_bf16(GM_ADDR x, GM_ADDR y, GM_ADDR z, MMQAddTiling tiling_data) {
  mmq::add_npu<bfloat16_t>(reinterpret_cast<__gm__ bfloat16_t *>(x),
                           reinterpret_cast<__gm__ bfloat16_t *>(y),
                           reinterpret_cast<__gm__ bfloat16_t *>(z),
                           tiling_data);
}
