#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/unsqueeze.h>
#endif

#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

struct Int64Extent3D {
  int64_t width;
  int64_t height;
  int64_t depth;
};

Int64Extent3D estimate_texture_3d_extents(const std::vector<int64_t>& sizes) {
  const int64_t width = val_at(-1, sizes);
  const int64_t height = val_at(-2, sizes);
  const int64_t channels = val_at(-3, sizes);
  const int64_t batch = val_at(-4, sizes);

  const int64_t aligned_channels = align_up(channels, INT64_C(4));
  const int64_t packed_channels = aligned_channels / 4;

  int64_t depth = 0;
  if (packed_channels == 0 || batch == 0) {
    depth = 0;
  } else if (batch > std::numeric_limits<int64_t>::max() / packed_channels) {
    depth = std::numeric_limits<int64_t>::max();
  } else {
    depth = batch * packed_channels;
  }

  return {width, height, depth};
}

bool exceeds_u32_numel(const IntArrayRef sizes) {
  const uint64_t u32_max =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  uint64_t acc = 1;

  for (const int64_t value : sizes) {
    if (value < 0) {
      return true;
    }
    if (value == 0) {
      return false;
    }
    const uint64_t v = static_cast<uint64_t>(value);
    if (v > u32_max || acc > u32_max / v) {
      return true;
    }
    acc *= v;
  }

  return false;
}

bool needs_buffer_storage(const IntArrayRef sizes) {
  if (sizes.size() > 4) {
    return true;
  }

  if (exceeds_u32_numel(sizes)) {
    return true;
  }

  const Int64Extent3D extents = estimate_texture_3d_extents(sizes.vec());
  const VkPhysicalDeviceLimits& limits =
      api::context()->adapter_ptr()->limits();
  const int64_t u32_max =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());

  return extents.width > static_cast<int64_t>(limits.maxImageDimension3D) ||
      extents.height > static_cast<int64_t>(limits.maxImageDimension3D) ||
      extents.depth > static_cast<int64_t>(limits.maxImageDimension3D) ||
      extents.width > u32_max || extents.height > u32_max ||
      extents.depth > u32_max;
}

int64_t size_product(const IntArrayRef sizes, int64_t start, int64_t end) {
  if (start >= end) {
    return 1;
  }
  int64_t acc = 1;
  for (int64_t i = start; i < end; ++i) {
    acc *= sizes[i];
  }
  return acc;
}

vTensor to_buffer_tensor(const Tensor& tensor) {
  api::Context* const context = api::context();

  Tensor vulkan_tensor = tensor.is_vulkan() ? tensor : tensor.vulkan();
  if (!vulkan_tensor.is_contiguous()) {
    vulkan_tensor = vulkan_tensor.contiguous();
  }

  vTensor v_src = convert(vulkan_tensor);
  if (v_src.storage_type() == api::StorageType::BUFFER) {
    return v_src;
  }

  if (v_src.dtype() == api::kHalf) {
    Tensor src_cpu = vulkan_tensor.cpu();
    return ops::to_vulkan(src_cpu, api::StorageType::BUFFER);
  }

  vTensor v_buffer{
      context,
      vulkan_tensor.sizes().vec(),
      v_src.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::StorageBuffer staging(context, v_src.dtype(), v_src.numel(), true);
  utils::pack_vtensor_to_staging(v_src, staging.buffer());

  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::READ);
  utils::pack_buffer_to_vtensor(staging.buffer(), v_buffer, pipeline_barrier);

  return v_buffer;
}

Tensor stack_buffer(const at::TensorList tensors, int64_t dim) {
  api::Context* const context = api::context();

  const Tensor first = tensors[0];
  std::vector<int64_t> input_sizes = first.sizes().vec();
  const int64_t input_ndim = static_cast<int64_t>(input_sizes.size());

  std::vector<int64_t> output_sizes = input_sizes;
  output_sizes.insert(
      output_sizes.begin() + dim, static_cast<int64_t>(tensors.size()));

  vTensor v_output{
      context,
      output_sizes,
      convert_dtype(first.scalar_type()),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  if (v_output.numel() == 0) {
    return convert(v_output);
  }

  std::vector<vTensor> v_inputs;
  v_inputs.reserve(tensors.size());
  for (const auto& t : tensors) {
    v_inputs.emplace_back(to_buffer_tensor(t));
  }

  const int64_t outer = size_product(input_sizes, 0, dim);
  const int64_t inner = size_product(input_sizes, dim, input_ndim);
  if (outer == 0 || inner == 0) {
    return convert(v_output);
  }

  const VkDeviceSize element_bytes = static_cast<VkDeviceSize>(
      v_output.nbytes() / v_output.numel());
  const VkDeviceSize copy_bytes =
      element_bytes * static_cast<VkDeviceSize>(inner);

  const int64_t num_inputs = static_cast<int64_t>(v_inputs.size());
  for (int64_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
    const vTensor& v_input = v_inputs[input_idx];
    TORCH_CHECK(
        v_input.dtype() == v_output.dtype(),
        "Vulkan buffer stack requires matching dtypes.");

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
      const int64_t src_elem_offset = outer_idx * inner;
      const int64_t dst_elem_offset =
          (outer_idx * num_inputs + input_idx) * inner;

      const VkDeviceSize src_offset_bytes =
          element_bytes * static_cast<VkDeviceSize>(src_elem_offset);
      const VkDeviceSize dst_offset_bytes =
          element_bytes * static_cast<VkDeviceSize>(dst_elem_offset);

      api::PipelineBarrier pipeline_barrier{};
      context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
          pipeline_barrier,
          v_input.buffer(pipeline_barrier, api::PipelineStage::TRANSFER),
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          {safe_downcast<uint32_t>(copy_bytes), 0u, 0u},
          {safe_downcast<uint32_t>(src_offset_bytes), 0u, 0u},
          {safe_downcast<uint32_t>(dst_offset_bytes), 0u, 0u},
          VK_NULL_HANDLE);
    }
  }

  return convert(v_output);
}

Tensor stack(const at::TensorList tensors, const int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "Vulkan stack expects at least one tensor");
  const at::Tensor& tensor = tensors[0];
  TORCH_CHECK(
      tensor.dim() <= 3,
      "Vulkan stack only supports up to 3d tensors as input!");

  TORCH_CHECK(
      dim >= -tensor.dim() - 1 && dim <= tensor.dim(),
      "Vulkan stack dimension out of range expected to be in range of [",
      -tensor.dim() - 1,
      ",",
      tensor.dim(),
      "], but got ",
      dim);

  for (const auto& t : tensors) {
    for (const auto d : c10::irange(t.dim())) {
      TORCH_CHECK(
          t.size(d) == tensor.size(d),
          "Vulkan stack inputs must have matching sizes, received ",
          t.size(d),
          tensor.size(d));
    }
  }

  const int64_t normalized_dim = dim < 0 ? dim + tensor.dim() + 1 : dim;

  std::vector<int64_t> output_sizes = tensor.sizes().vec();
  output_sizes.insert(
      output_sizes.begin() + normalized_dim,
      static_cast<int64_t>(tensors.size()));

  const bool output_needs_buffer = needs_buffer_storage(output_sizes);
  bool any_buffer = false;
  if (!output_needs_buffer) {
    for (const auto& t : tensors) {
      if (t.is_vulkan() &&
          convert(t).storage_type() == api::StorageType::BUFFER) {
        any_buffer = true;
        break;
      }
    }
  }

  if (output_needs_buffer || any_buffer) {
    return stack_buffer(tensors, normalized_dim);
  }

  // Unsqueeze each tensor in the list
  std::vector<Tensor> unsqueezed_outputs;
  for (const auto& t : tensors) {
    unsqueezed_outputs.push_back(at::unsqueeze(t, dim));
  }
  // Cat the tensors
  const at::TensorList tensorList = unsqueezed_outputs;
  return at::cat(tensorList, dim);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::stack"), TORCH_FN(stack));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
