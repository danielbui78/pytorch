#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

inline bool buffer_reduction_dtype_supported(const api::ScalarType dtype) {
  return dtype == api::kFloat ||
      (dtype == api::kHalf && api::context()->fp16_buffer_storage_enabled());
}

enum class BufferReductionRoute {
  Auto = 0,
  ForceSinglePass = 1,
  ForceGuardedFp32 = 2,
  ForceTwoPass = 3,
};

constexpr uint32_t kTwoPassAutoMinDim = 200000u;

inline BufferReductionRoute parse_buffer_reduction_route(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return BufferReductionRoute::Auto;
  }
  if (
      std::strcmp(value, "single_pass") == 0 ||
      std::strcmp(value, "force_single_pass") == 0) {
    return BufferReductionRoute::ForceSinglePass;
  }
  if (
      std::strcmp(value, "guarded_fp32") == 0 ||
      std::strcmp(value, "force_guarded_fp32") == 0) {
    return BufferReductionRoute::ForceGuardedFp32;
  }
  if (
      std::strcmp(value, "two_pass") == 0 ||
      std::strcmp(value, "force_two_pass") == 0) {
    return BufferReductionRoute::ForceTwoPass;
  }
  return BufferReductionRoute::Auto;
}

inline BufferReductionRoute buffer_reduction_route() {
  static const BufferReductionRoute route = parse_buffer_reduction_route(
      std::getenv("PYTORCH_VULKAN_BUFFER_REDUCTION_ROUTE"));
  return route;
}

inline const char* buffer_reduction_route_name(const BufferReductionRoute route) {
  switch (route) {
    case BufferReductionRoute::Auto:
      return "auto";
    case BufferReductionRoute::ForceSinglePass:
      return "single_pass";
    case BufferReductionRoute::ForceGuardedFp32:
      return "guarded_fp32";
    case BufferReductionRoute::ForceTwoPass:
      return "two_pass";
    default:
      return "unknown";
  }
}

inline bool buffer_reduction_debug_enabled() {
  static const bool enabled = []() {
    const char* value =
        std::getenv("PYTORCH_VULKAN_BUFFER_REDUCTION_DEBUG_ROUTE");
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "on") == 0;
  }();
  return enabled;
}

inline void log_buffer_reduction_route_decision(
    const char* op_name,
    const BufferReductionRoute force_mode,
    const bool use_buffer_path,
    const api::ScalarType output_dtype,
    const uint32_t dim_size,
    const bool use_fp16_guard,
    const bool use_two_pass,
    const uint32_t threshold) {
  if (!buffer_reduction_debug_enabled()) {
    return;
  }
  const char* selected_route = "image";
  if (use_fp16_guard) {
    selected_route = "guarded_fp32";
  } else if (use_buffer_path) {
    selected_route = use_two_pass ? "two_pass" : "single_pass";
  }
  std::fprintf(
      stderr,
      "[vulkan_buffer_reduction][%s] force=%s use_buffer=%d dtype=%d dim_size=%u threshold=%u selected=%s\n",
      op_name,
      buffer_reduction_route_name(force_mode),
      use_buffer_path ? 1 : 0,
      static_cast<int>(output_dtype),
      dim_size,
      threshold,
      selected_route);
  std::fflush(stderr);
}

inline bool should_use_fp16_guard(
    const bool use_buffer_path,
    const api::ScalarType output_dtype) {
  if (!use_buffer_path || output_dtype != api::kHalf) {
    return false;
  }
  switch (buffer_reduction_route()) {
    case BufferReductionRoute::ForceSinglePass:
      return false;
    case BufferReductionRoute::Auto:
    case BufferReductionRoute::ForceGuardedFp32:
      return true;
    default:
      return true;
  }
}

inline bool should_use_two_pass(
    const bool use_buffer_path,
    const api::ScalarType output_dtype,
    const uint32_t dim_size) {
  if (!use_buffer_path || output_dtype != api::kFloat || dim_size <= 1u) {
    return false;
  }
  switch (buffer_reduction_route()) {
    case BufferReductionRoute::ForceSinglePass:
      return false;
    case BufferReductionRoute::ForceTwoPass:
      return true;
    case BufferReductionRoute::Auto:
      return dim_size >= kTwoPassAutoMinDim;
    default:
      return false;
  }
}

inline void check_storage_buffer_limit(
    const vTensor& v_tensor,
    const char* name) {
  const VkDeviceSize limit =
      api::context()->adapter_ptr()->limits().maxStorageBufferRange;
  TORCH_CHECK(
      v_tensor.gpu_nbytes() <= limit,
      "Vulkan buffer ",
      name,
      " exceeds maxStorageBufferRange (",
      v_tensor.gpu_nbytes(),
      " > ",
      limit,
      ").");
}

inline uint32_t reduction_dim_size(const vTensor& v_tensor, uint32_t dim) {
  switch (dim) {
    case 0u:
      return get_dim<Dim4D::Batch>(v_tensor);
    case 1u:
      return get_dim<Dim4D::Channel>(v_tensor);
    case 2u:
      return get_dim<Dim4D::Height>(v_tensor);
    case 3u:
      return get_dim<Dim4D::Width>(v_tensor);
    default:
      return 1u;
  }
}

inline const api::ShaderInfo& select_mean_buffer_shader(
    const api::ScalarType dtype) {
  return (dtype == api::kHalf) ? VK_KERNEL(mean_dim_buffer_f16)
                               : VK_KERNEL(mean_dim_buffer);
}

constexpr uint32_t kTwoPassBlockSize = 1024u;

Tensor run_mean_buffer_reduction_two_pass(
    api::Context* const context,
    vTensor& v_input,
    const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& full_output_size,
    bool keepdim,
    uint32_t axis,
    uint32_t buffer_dim,
    uint32_t dim_size) {
  const uint32_t block_count = div_up(dim_size, kTwoPassBlockSize);

  std::vector<int64_t> pass1_output_size = v_input.sizes();
  pass1_output_size.at(buffer_dim) = static_cast<int64_t>(block_count);

  vTensor v_pass1 = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::ReductionOutput);
    return vTensor{
        context,
        pass1_output_size,
        api::kFloat,
        api::StorageType::BUFFER,
        v_input.gpu_memory_layout(),
    };
  }();
  vTensor v_output_buffer = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::ReductionOutput);
    return vTensor{
        context,
        full_output_size,
        api::kFloat,
        api::StorageType::BUFFER,
        v_input.gpu_memory_layout(),
    };
  }();

  check_storage_buffer_limit(v_input, "mean input");
  check_storage_buffer_limit(v_pass1, "mean pass1 output");
  check_storage_buffer_limit(v_output_buffer, "mean output");

  const struct BlockPass1 final {
    uvec2 dim_info;
    uint32_t block_size;
  } block_pass1{
      {axis, dim_size},
      kTwoPassBlockSize,
  };
  api::UniformParamsBuffer params_pass1(context, block_pass1);

  const struct BlockPass2 final {
    uint32_t axis;
    uint32_t block_count;
    float factor;
  } block_pass2{
      axis,
      block_count,
      dim_size == 0u ? 0.0f : 1.0f / static_cast<float>(dim_size),
  };
  api::UniformParamsBuffer params_pass2(context, block_pass2);

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(mean_dim_buffer_pass1),
      pipeline_barrier,
      {safe_downcast<uint32_t>(v_pass1.gpu_numel()), 1u, 1u},
      {32u, 1u, 1u},
      VK_NULL_HANDLE,
      v_pass1.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_pass1.buffer_metadata(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.buffer_metadata(),
      params_pass1.buffer());

  context->submit_compute_job(
      VK_KERNEL(mean_dim_buffer_pass2),
      pipeline_barrier,
      {safe_downcast<uint32_t>(v_output_buffer.gpu_numel()), 1u, 1u},
      {32u, 1u, 1u},
      VK_NULL_HANDLE,
      v_output_buffer.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_output_buffer.buffer_metadata(),
      v_pass1.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_pass1.buffer_metadata(),
      params_pass2.buffer());

  Tensor result = convert(v_output_buffer);
  if (!keepdim) {
    result = result.reshape(output_size);
  }
  return result;
}

Tensor run_mean_buffer_reduction(
    api::Context* const context,
    vTensor& v_input,
    const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& full_output_size,
    bool keepdim,
    uint32_t axis,
    uint32_t dim_size,
    const api::ScalarType output_dtype) {
  vTensor v_output_buffer = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::ReductionOutput);
    return vTensor{
        context,
        full_output_size,
        output_dtype,
        api::StorageType::BUFFER,
        v_input.gpu_memory_layout(),
    };
  }();

  check_storage_buffer_limit(v_input, "mean input");
  check_storage_buffer_limit(v_output_buffer, "mean output");

  const struct Block final {
    uvec2 dim_info;
    float factor;
  } block{
      {axis, dim_size},
      dim_size == 0u ? 0.0f : 1.0f / static_cast<float>(dim_size),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      select_mean_buffer_shader(output_dtype),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      {safe_downcast<uint32_t>(v_output_buffer.gpu_numel()), 1u, 1u},
      // local work group size
      {32u, 1u, 1u},
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output_buffer.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_output_buffer.buffer_metadata(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.buffer_metadata(),
      // params buffer
      params.buffer());

  Tensor result = convert(v_output_buffer);
  if (!keepdim) {
    result = result.reshape(output_size);
  }
  return result;
}

Tensor mean_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  TORCH_CHECK(
      self.dim() >= 2 && self.dim() <= 4,
      "Vulkan mean_dim supports 2d, 3d, 4d tensors as input!");
  TORCH_CHECK(
      dim >= -self.dim() && dim < self.dim(),
      "Vulkan mean.dim dimension out of range expected to be in range of [",
      -self.dim(),
      ",",
      self.dim() - 1,
      "], but got ",
      dim);

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  vTensor& v_input = convert(input);

  // Normalize dim into range [0, self.dim()]
  dim = utils::normalize(dim, self.dim());

  // Create the output texture
  std::vector<int64_t> output_size = v_input.sizes();
  std::vector<int64_t> full_output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }
  full_output_size[dim] = 1;

  ScalarType type = self.scalar_type();
  if (dtype.has_value()) {
    type = dtype.value();
  }

  const api::ScalarType output_dtype = convert_dtype(type);

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  const int64_t buffer_dim = dim;

  // Shift dim into 4d range
  if (self.dim() < 4) {
    dim += (4 - self.dim());
  }

  const uint32_t normalized_dim = static_cast<uint32_t>(dim);
  const bool use_buffer_path =
      v_input.storage_type() == api::StorageType::BUFFER &&
      buffer_reduction_dtype_supported(output_dtype);
    const BufferReductionRoute force_mode = buffer_reduction_route();
    const bool use_fp16_guard = should_use_fp16_guard(use_buffer_path, output_dtype);
    const bool use_two_pass =
      should_use_two_pass(use_buffer_path, output_dtype, dim_size);
    log_buffer_reduction_route_decision(
      "mean",
      force_mode,
      use_buffer_path,
      output_dtype,
      dim_size,
      use_fp16_guard,
      use_two_pass,
      kTwoPassAutoMinDim);
    if (use_fp16_guard) {
    // Temporary fp16 guard: large buffer reductions can produce invalid fp16
    // outputs; reduce in float and cast back.
    const Tensor input_float = input.to(at::kFloat);
    Tensor reduced_float =
        mean_dim(input_float, buffer_dim, keepdim, c10::ScalarType::Float);
    return reduced_float.to(c10::ScalarType::Half);
  }
    if (use_two_pass) {
    const uint32_t axis =
        safe_downcast<uint32_t>((self.dim() - 1) - buffer_dim);
    return run_mean_buffer_reduction_two_pass(
        context,
        v_input,
        output_size,
        full_output_size,
        keepdim,
        axis,
        safe_downcast<uint32_t>(buffer_dim),
        dim_size);
  }
  if (use_buffer_path) {
    const uint32_t axis =
        safe_downcast<uint32_t>((self.dim() - 1) - buffer_dim);
    const uint32_t reduction_size = dim_size;
    return run_mean_buffer_reduction(
        context,
        v_input,
        output_size,
        full_output_size,
        keepdim,
        axis,
        reduction_size,
        output_dtype);
  }

  vTensor v_output = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::ReductionOutput);
    return vTensor{
        context,
        output_size,
        output_dtype,
    };
  }();

  // Create the params buffer
  const struct Block final {
    uvec2 dim_info;
    int32_t channel;
  } block{
      {normalized_dim, dim_size},
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      keepdim ? VK_KERNEL(mean_dim_keepdim) : VK_KERNEL(mean_dim),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
  return convert(v_output);
}

Tensor mean_dim_IntList(
    const at::Tensor& self,
    const OptionalIntArrayRef opt_dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  TORCH_CHECK(
      opt_dim.has_value(), "Vulkan mean without a dim arg is not implemented");

  std::set<int64_t> dims_set;

  if (opt_dim.has_value()) {
    auto dims = opt_dim.value();
    for (const auto& d : dims) {
      TORCH_CHECK(
          d >= -self.dim() && d < self.dim(),
          "Vulkan mean.dim_IntList dimension out of range expected to be in range of [",
          -self.dim(),
          ",",
          self.dim() - 1,
          "], but got ",
          d);
      int64_t dim_normalized = utils::normalize(d, self.dim());
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      dims_set.insert(dim_normalized);
    }
    Tensor output = self;
    for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
      output = mean_dim(output, *it, keepdim, dtype);
    }
    return output;
  }
  return self;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::mean.dim"), TORCH_FN(mean_dim_IntList));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
