#include <ATen/InferSize.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

bool linearPathTraceEnabled() {
  const char* const value = std::getenv("PYTORCH_VULKAN_LINEAR_PATH_TRACE");
  if (nullptr == value) {
    return false;
  }

  return
      0 == std::strcmp(value, "1") ||
      0 == std::strcmp(value, "true") ||
      0 == std::strcmp(value, "TRUE") ||
      0 == std::strcmp(value, "on") ||
      0 == std::strcmp(value, "ON");
}

std::string formatSizes(const IntArrayRef sizes) {
  std::string formatted = "[";
  for (const auto i : c10::irange(sizes.size())) {
    if (0 != i) {
      formatted += ",";
    }
    formatted += std::to_string(sizes[i]);
  }
  formatted += "]";
  return formatted;
}

const char* storageTypeName(const api::StorageType storage_type) {
  switch (storage_type) {
    case api::StorageType::BUFFER:
      return "buffer";
    case api::StorageType::TEXTURE_3D:
      return "texture_3d";
    case api::StorageType::UNKNOWN:
      return "unknown";
  }
  return "unrecognized";
}

void traceViewEvent(
    const char* const event,
    const Tensor& self,
    const IntArrayRef output_sizes) {
  if (!linearPathTraceEnabled() || !self.is_vulkan()) {
    return;
  }

  const vTensor& v_self = convert(self);
  const std::string input_sizes = formatSizes(v_self.sizes());
  const std::string target_sizes = formatSizes(output_sizes);

  std::fprintf(
      stderr,
      "[vk_linear_path] file=Shape event=%s input_sizes=%s output_sizes=%s storage=%s\n",
      event,
      input_sizes.c_str(),
      target_sizes.c_str(),
      storageTypeName(v_self.storage_type()));
  std::fflush(stderr);
}

api::AllocationTag classify_view_output_tag(
    const IntArrayRef input_sizes,
    const IntArrayRef output_sizes) {
  if (
      input_sizes.size() == 3u && output_sizes.size() == 4u &&
      output_sizes[0] == input_sizes[0] &&
      output_sizes[1] == input_sizes[1] &&
      output_sizes[2] > 0 && output_sizes[3] > 0 &&
      output_sizes[2] * output_sizes[3] == input_sizes[2]) {
    return api::AllocationTag::ViewSplitHeadsOutput;
  }

  if (
      input_sizes.size() == 4u && output_sizes.size() == 3u &&
      output_sizes[0] == input_sizes[0] &&
      output_sizes[1] == input_sizes[1] &&
      input_sizes[2] > 0 && input_sizes[3] > 0 &&
      input_sizes[2] * input_sizes[3] == output_sizes[2]) {
    return api::AllocationTag::ViewMergeHeadsOutput;
  }

  if (
      output_sizes.size() == 2u && !input_sizes.empty() &&
      output_sizes[1] == input_sizes.back() &&
      api::utils::multiply_integers(output_sizes) ==
          api::utils::multiply_integers(input_sizes)) {
    return api::AllocationTag::ViewFlattenOutput;
  }

  return api::AllocationTag::ViewOutput;
}

} // namespace

static Tensor view_internal(const Tensor& self_arg, const IntArrayRef shape) {
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);

  at::DimVector inferred_size = at::infer_size_dv(shape, self.numel());
  IntArrayRef output_size(inferred_size);
  traceViewEvent("view_internal_enter", self, output_size);

  api::AllocationTagScope tag_scope(
      classify_view_output_tag(v_self.sizes(), inferred_size));
  vTensor v_output{
      context,
      output_size.vec(),
      v_self.dtype(),
  };
  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  api::StorageBuffer buffer(context, api::kFloat, v_self.gpu_numel(), true);

  utils::pack_vtensor_to_staging(v_self, buffer.buffer());

  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      buffer.buffer(),
      // Previous access
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::WRITE,
      // Next access
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::READ);

  utils::pack_buffer_to_vtensor(buffer.buffer(), v_output, pipeline_barrier);
  traceViewEvent("view_internal_materialized", self, output_size);

  return convert(v_output);
}

inline Tensor view(const Tensor& self_arg, IntArrayRef shape) {
  traceViewEvent("view_dispatch", self_arg, shape);
  return view_internal(self_arg, shape);
}

static Tensor _reshape_alias(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides) {
  traceViewEvent("reshape_alias_dispatch", self_arg, shape);
  return view_internal(self_arg, shape);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
