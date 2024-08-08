#include "acl_system_info.h"
#include "acl/acl_base.h"
#include "torch/extension.h"
#include <mutex>

namespace mmq {

int64_t get_npu_num_vec_cores() {
  static int64_t num_vec_cores = -1;
  static std::once_flag flag;
  std::call_once(flag, [&] {
    auto err = aclGetDeviceCapability(
        0, aclDeviceInfo::ACL_DEVICE_INFO_VECTOR_CORE_NUM, &num_vec_cores);
    TORCH_CHECK(err == 0, "get vector core num error", err);
  });
  return num_vec_cores;
}

} // namespace mmq
