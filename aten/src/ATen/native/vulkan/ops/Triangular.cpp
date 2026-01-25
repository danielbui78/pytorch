#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/tril.h>
#include <ATen/ops/where.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor tril(const Tensor& self, int64_t diagonal) {
  const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
  Tensor out_cpu = at::tril(self_cpu, diagonal);
  return out_cpu.to(at::kVulkan);
}

Tensor& tril_out(const Tensor& self, int64_t diagonal, Tensor& out) {
  const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
  Tensor out_cpu = at::tril(self_cpu, diagonal);
  if (out.is_vulkan()) {
    out.copy_(out_cpu.to(at::kVulkan));
  } else {
    out.copy_(out_cpu);
  }
  return out;
}

Tensor where_self(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  const Tensor condition_cpu = condition.is_vulkan() ? condition.cpu() : condition;
  const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
  const Tensor other_cpu = other.is_vulkan() ? other.cpu() : other;
  Tensor out_cpu = at::where(condition_cpu, self_cpu, other_cpu);
  return out_cpu.to(at::kVulkan);
}


#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::tril"), TORCH_FN(tril));
  m.impl(TORCH_SELECTIVE_NAME("aten::tril.out"), TORCH_FN(tril_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::where.self"), TORCH_FN(where_self));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
