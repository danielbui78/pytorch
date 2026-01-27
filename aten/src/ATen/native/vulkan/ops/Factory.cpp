#include <ATen/native/vulkan/ops/Factory.h>
#include <torch/library.h>

#include <limits>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

struct Int64Extent3D {
  int64_t width;
  int64_t height;
  int64_t depth;
};

Int64Extent3D estimate_texture_3d_extents(const std::vector<int64_t>& sizes) {
  using api::utils::align_up;
  using api::utils::val_at;

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

} // namespace

Tensor _empty_affine_quantized(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const double scale,
    const int64_t zero_point,
    const std::optional<MemoryFormat> memory_format) {
    const api::StorageType storage_type = needs_buffer_storage(sizes)
      ? api::StorageType::BUFFER
      : api::StorageType::TEXTURE_3D;
      const c10::MemoryFormat resolved_format =
        memory_format.value_or(c10::MemoryFormat::Contiguous);
  return convert_quantized(vTensor{
      api::context(),
      sizes.vec(),
      scale,
      zero_point,
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
        get_gpu_memory_layout(storage_type, resolved_format),
  });
}

static Tensor empty_memory_format(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<MemoryFormat> memory_format) {
    const api::StorageType storage_type = needs_buffer_storage(sizes)
      ? api::StorageType::BUFFER
      : api::StorageType::TEXTURE_3D;
      const c10::MemoryFormat resolved_format =
        memory_format.value_or(c10::MemoryFormat::Contiguous);
  return convert(vTensor{
      api::context(),
      sizes.vec(),
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
        get_gpu_memory_layout(storage_type, resolved_format),
  });
}

static Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef /* strides */,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
  return empty_memory_format(
      sizes, dtype, layout, device, pin_memory, c10::MemoryFormat::Contiguous);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty.memory_format"),
      at::native::vulkan::ops::empty_memory_format);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_empty_affine_quantized"),
      at::native::vulkan::ops::_empty_affine_quantized);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty_strided"),
      TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
