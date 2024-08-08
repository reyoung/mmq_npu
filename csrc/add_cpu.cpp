#include "add_cpu.h"
#include "acl_system_info.h"
#include "add_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "utils.h"

// 这三个头文件是bisheng编译器生成的头文件。用于在C程序中调用npu kernel
#include "aclrtlaunch_mmq_npu_add_bf16.h"
#include "aclrtlaunch_mmq_npu_add_float.h"
#include "aclrtlaunch_mmq_npu_add_half.h"

namespace mmq {

// add out = x + y
void add(torch::Tensor out, torch::Tensor x, torch::Tensor y) {
  TORCH_CHECK(x.dtype() == y.dtype(), "x, y dtype must be same");
  TORCH_CHECK(out.dtype() == x.dtype(), "x, out dtype must be same");
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "x must be npu tensor");
  TORCH_CHECK(y.device().type() == c10::DeviceType::PrivateUse1,
              "y must be npu tensor");
  TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1,
              "out must be npu tensor");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  TORCH_CHECK(x.numel() == y.numel(), "x, y numel must be same");
  TORCH_CHECK(x.numel() == out.numel(), "x, out numel must be same");

  auto stream = c10_npu::getCurrentNPUStream();

  uint32_t numel = x.numel();

  // tiling size
  // 为一次数据搬运/计算的最大元素数量。最好是2的整数次方，方便下文对齐
  constexpr uint32_t tiling_size = 512;

  uint32_t num_vec_cores = mmq::get_npu_num_vec_cores();
  uint32_t num_tilings = mmq::CeilDiv(numel, tiling_size);

  // num_blocks 是 num_tilings 和 num_vec_cores的最小值。
  // 这是因为
  //  1.
  //  一个block至少要做一个tiling，不然这个block就是完全空闲的，没必要启动这个block
  //  2. 因为ascend 没有 warp scheduler，所以block的数量如果超过计算核心的数量，
  //     也不会有任务切换的能力。所以block的数量不应该超过计算核心的数量
  uint32_t num_blocks = std::min(num_tilings, num_vec_cores);

  // 32 byte align
  constexpr uint32_t align_byte = 32;

  uint32_t align_n_elems = align_byte / x.dtype().itemsize();

  // 每个block的大小，基本上是按照 numel / num_blocks均分。不过最后尽量对齐到
  // align_n_elems 避免 1..n 地址不连续
  uint32_t block_size =
      mmq::CeilDiv(numel / num_blocks, align_n_elems) * align_n_elems;

  // 最后一个block的size处理数据参差的问题
  TORCH_CHECK_GT(numel, (num_blocks - 1) * block_size)
      << "last block size is negative, internal error";

  uint32_t last_block_size = numel - (num_blocks - 1) * block_size;

  MMQAddTiling tiling_data{};
  tiling_data.block_size_ = block_size;
  tiling_data.last_block_size_ = last_block_size;
  tiling_data.tiling_size_ = tiling_size;

  // TODO:
  // 将switch操作做成一个macro或者类型，这样其他kernel就不用写好几遍switch了
  uint32_t err = 0;
  switch (x.scalar_type()) {
  case caffe2::ScalarType::Float:
    err = aclrtlaunch_mmq_npu_add_float(num_blocks, stream, x.data_ptr(),
                                        y.data_ptr(), out.data_ptr(),
                                        &tiling_data);
    break;
  case caffe2::ScalarType::Half:
    err = aclrtlaunch_mmq_npu_add_half(num_blocks, stream, x.data_ptr(),
                                       y.data_ptr(), out.data_ptr(),
                                       &tiling_data);
    break;
  case caffe2::ScalarType::BFloat16:
    err = aclrtlaunch_mmq_npu_add_bf16(num_blocks, stream, x.data_ptr(),
                                       y.data_ptr(), out.data_ptr(),
                                       &tiling_data);
    break;
  default:
    TORCH_CHECK(false, "unsupported data type");
  }

  TORCH_CHECK(err == 0, "aclrtlaunch_mmq_npu_add failed", err);
}

} // namespace mmq
