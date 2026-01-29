#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <limits>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

namespace {
inline int64_t normalize_dim(int64_t d, int64_t n) {
  return (d % n + n) % n;
}
} // namespace

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

Tensor cat_buffer(
    const MaterializedITensorListRef& tensors,
    const int64_t dim,
    const std::vector<int64_t>& result_size,
    const c10::ScalarType scalar_type) {
  api::Context* const context = api::context();

  vTensor v_output{
      context,
      result_size,
      convert_dtype(scalar_type),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  if (v_output.numel() == 0) {
    return convert(v_output);
  }

  std::vector<vTensor> v_inputs;
  v_inputs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    v_inputs.emplace_back(to_buffer_tensor(tensor));
  }

  const IntArrayRef out_sizes(result_size);
  const int64_t ndim = out_sizes.size();
  const int64_t inner = size_product(out_sizes, dim + 1, ndim);
  const int64_t outer = size_product(out_sizes, 0, dim);
  const int64_t total_dim = out_sizes[dim];

  const VkDeviceSize element_bytes = static_cast<VkDeviceSize>(
      v_output.nbytes() / v_output.numel());

  int64_t running_dim = 0;
  for (const vTensor& v_input : v_inputs) {
    TORCH_CHECK(
        v_input.dtype() == v_output.dtype(),
        "Vulkan buffer cat requires matching dtypes.");

    const int64_t dim_size = v_input.sizes().at(dim);
    const int64_t slice_elems = dim_size * inner;
    if (slice_elems == 0) {
      running_dim += dim_size;
      continue;
    }

    const VkDeviceSize copy_bytes =
        element_bytes * static_cast<VkDeviceSize>(slice_elems);

    for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
      const int64_t src_elem_offset = outer_idx * slice_elems;
      const int64_t dst_elem_offset =
          outer_idx * total_dim * inner + running_dim * inner;

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

    running_dim += dim_size;
  }

  return convert(v_output);
}

Tensor cat_batch(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    api::PipelineBarrier pipeline_barrier{};

    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // pipeline barrier
        pipeline_barrier,
        // images
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // copy details
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);

    // Increment by the number of texels in the depth dimension
    dst_offset.data[2u] += v_self.extents().data[2u];
  }

  return convert(v_output);
}

Tensor cat_feature(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  // Determine the channels of the output tensor
  uint32_t ch_total = 0;
  for (const at::Tensor& tensor : tensors) {
    ch_total += get_dim<Dim4D::Channel>(tensor);
  }

  // Running counter of the number of channels already appended.
  uint32_t ch_current = 0;
  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    // Determine the number of channel texels that will be modified by
    // appending this input tensor
    uint32_t start_ch4 = ch_current / 4;

    uint32_t end_ch4 =
        api::utils::div_up(ch_current + get_dim<Dim4D::Channel>(v_self), 4u);

    uint32_t ch4_range = end_ch4 - start_ch4;
    uint32_t nc4_range = ch4_range * get_dim<Dim4D::Batch>(v_self);

    const struct Block final {
      ivec3 outExtents;
      int32_t fill0;
      ivec3 inExtents;
      int32_t fill1;
      uvec2 outChInfo;
      uvec2 inChInfo;
      uvec4 appendedChInfo;
    } block{
        api::utils::make_ivec3(v_output.extents()),
        0,
        api::utils::make_ivec3(v_self.extents()),
        0,
        {
            ch_total,
            api::utils::div_up(ch_total, 4u),
        },
        {
            get_dim<Dim4D::Channel>(v_self),
            api::utils::align_up(get_dim<Dim4D::Channel>(v_self), 4u),
        },
        {
            ch_current,
            start_ch4,
            ch4_range,
            0u,
        },
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(cat_feature),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            get_dim<Dim4D::Width>(v_output),
            get_dim<Dim4D::Height>(v_output),
            nc4_range,
        },
        // local work group size
        adaptive_work_group_size(v_self.extents()),
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());

    ch_current += get_dim<Dim4D::Channel>(v_self);
  }

  return convert(v_output);
}

Tensor cat_feature_mult4ch(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  int64_t depth_size_allprior = 0;
  int64_t ch_interval = 0;
  for (const at::Tensor& tensor : tensors) {
    ch_interval += get_dim<Dim4D::Channel>(tensor);
  }
  const int64_t depth_interval = ch_interval / 4;

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor_arg : tensors) {
    const Tensor tensor =
        tensor_arg.is_vulkan() ? tensor_arg : tensor_arg.vulkan();
    const vTensor& v_self = convert(tensor);

    const uint32_t depth_slice =
        safe_downcast<uint32_t>(get_dim<Dim4D::Channel>(tensor) / 4);

    uvec3 copy_extents{
        v_self.extents().data[0u], v_self.extents().data[1u], depth_slice};

    for (const auto b : c10::irange(get_dim<Dim4D::Batch>(tensor))) {
      src_offset.data[2u] = safe_downcast<uint32_t>(depth_slice * b);
      dst_offset.data[2u] =
          depth_size_allprior + safe_downcast<uint32_t>(depth_interval * b);

      api::PipelineBarrier pipeline_barrier{};

      context->submit_copy<api::VulkanImage, api::VulkanImage>(
          // pipeline barrier
          pipeline_barrier,
          // images
          v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          // copy details
          copy_extents,
          src_offset,
          dst_offset,
          // fence handle
          VK_NULL_HANDLE);
    }

    depth_size_allprior += depth_slice;
  }

  return convert(v_output);
}

Tensor cat_width(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  // TORCH_CHECK(false, "Vulkan cat not implemented for width dimension!");
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    api::PipelineBarrier pipeline_barrier{};

    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // pipeline barrier
        pipeline_barrier,
        // images
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // copy details
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);

    // Increment by width
    dst_offset.data[0u] += v_self.extents().data[0u];
  }

  return convert(v_output);
}

Tensor cat_height(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    api::PipelineBarrier pipeline_barrier{};

    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // pipeline barrier
        pipeline_barrier,
        // images
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // copy details
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);

    // Increment by height
    dst_offset.data[1u] += v_self.extents().data[1u];
  }

  return convert(v_output);
}

Tensor cat(const at::ITensorListRef& tensors, const int64_t in_dim) {
  TORCH_CHECK(!tensors.empty(), "Vulkan cat expects at least one tensor");
  auto materialized = tensors.materialize();
  TORCH_INTERNAL_ASSERT(!materialized.empty(), "Accessing empty array");
  const at::Tensor& tensor = materialized[0];
  auto ndim = safe_downcast<uint32_t>(tensor.dim());
  const int64_t dim = normalize_dim(in_dim, ndim);
  int64_t cat_dim_size = 0;
  bool is_mult4ch = true;

  for (const at::Tensor& t : materialized) {
    TORCH_INTERNAL_ASSERT(
        t.dim() <= 4,
        "Vulkan cat expects inputs to have at most 4 dimensions, but got ",
        t.dim(),
        "d");

    if (ndim < 3 || get_dim<Dim4D::Channel>(t) % 4 != 0) {
      is_mult4ch = false;
    }

    for (const auto d : c10::irange(ndim)) {
      if (d == dim) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          t.size(d) == tensor.size(d),
          "Vulkan cat inputs must have matching sizes except concatenated dimension");
    }
    cat_dim_size += t.size(dim);
  }

  auto result_size = tensor.sizes().vec();
  TORCH_INTERNAL_ASSERT(!result_size.empty(), "Accessing empty array");
  result_size[dim] = cat_dim_size;

  const bool output_needs_buffer = needs_buffer_storage(result_size);
  bool any_buffer = false;
  if (!output_needs_buffer) {
    for (const at::Tensor& t : materialized) {
      if (t.is_vulkan() &&
          convert(t).storage_type() == api::StorageType::BUFFER) {
        any_buffer = true;
        break;
      }
    }
  }

  if (output_needs_buffer || any_buffer) {
    return cat_buffer(materialized, dim, result_size, tensor.scalar_type());
  }

  vTensor v_output{
      api::context(), result_size, convert_dtype(tensor.scalar_type())};

  if (dim == ndim - 1) {
    return cat_width(materialized, v_output);
  }
  if (dim == ndim - 2) {
    return cat_height(materialized, v_output);
  } else if (dim == ndim - 3) {
    if (is_mult4ch) {
      return cat_feature_mult4ch(materialized, v_output);
    }
    return cat_feature(materialized, v_output);
  }
  return cat_batch(materialized, v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::cat"), TORCH_FN(cat));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
