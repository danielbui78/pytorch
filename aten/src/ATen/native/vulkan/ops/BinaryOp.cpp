#ifdef USE_VULKAN_API
#include <ATen/ArrayRef.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/eq.h>
#include <ATen/ops/floor_divide.h>
#endif
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <mutex>
#include <string>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

enum class AddTensorInplaceMode {
  Wrapper,
  Direct,
};

inline AddTensorInplaceMode add_tensor_inplace_mode() {
  static const AddTensorInplaceMode mode = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_ADD_TENSOR_INPLACE_MODE");
    if (env == nullptr || env[0] == '\0') {
      return AddTensorInplaceMode::Direct;
    }

    std::string value(env);
    for (char& ch : value) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (
        value == "wrapper" || value == "legacy") {
      return AddTensorInplaceMode::Wrapper;
    }
    if (value == "direct" || value == "default" || value == "auto") {
      return AddTensorInplaceMode::Direct;
    }

    TORCH_CHECK(
        false,
        "PYTORCH_VULKAN_ADD_TENSOR_INPLACE_MODE must be one of: "
        "wrapper, direct");
  }();
  return mode;
}

inline const char* add_tensor_inplace_mode_name(
    const AddTensorInplaceMode mode) {
  switch (mode) {
    case AddTensorInplaceMode::Wrapper:
      return "wrapper";
    case AddTensorInplaceMode::Direct:
      return "direct";
  }
  return "unknown";
}

inline bool fp16_buffer_storage_enabled() {
  return api::context()->fp16_buffer_storage_enabled();
}

inline bool buffer_dtype_supported(const vTensor& v_tensor) {
  return v_tensor.dtype() == api::kFloat ||
      (v_tensor.dtype() == api::kHalf && fp16_buffer_storage_enabled());
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

inline api::GPUMemoryLayout buffer_output_layout(
    const vTensor& v_self,
    const vTensor& v_other) {
  return (v_self.gpu_memory_layout() == v_other.gpu_memory_layout())
      ? v_self.gpu_memory_layout()
      : api::GPUMemoryLayout::TENSOR_WIDTH_PACKED;
}

inline const api::ShaderInfo& select_buffer_shader(
    const vTensor& v_tensor,
    const api::ShaderInfo& shader_float,
    const api::ShaderInfo& shader_f16) {
  return (v_tensor.dtype() == api::kHalf) ? shader_f16 : shader_float;
}

static Tensor to_buffer_tensor(const Tensor& src_arg) {
  Tensor src_cpu = src_arg.is_vulkan() ? src_arg.cpu() : src_arg;
  Tensor src_contig = src_cpu.contiguous(src_cpu.suggest_memory_format());
  vTensor v_buffer = ops::to_vulkan(src_contig, api::StorageType::BUFFER);
  return convert(v_buffer);
}

static Tensor to_texture_tensor(const Tensor& src_arg) {
  Tensor src_cpu = src_arg.is_vulkan() ? src_arg.cpu() : src_arg;
  Tensor src_contig = src_cpu.contiguous(src_cpu.suggest_memory_format());
  vTensor v_texture = ops::to_vulkan(src_contig, api::StorageType::TEXTURE_3D);
  return convert(v_texture);
}

inline bool binaryop_trace_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("PYTORCH_VULKAN_BINARYOP_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

inline int64_t binaryop_trace_limit() {
  static const int64_t limit = []() -> int64_t {
    const char* value = std::getenv("PYTORCH_VULKAN_BINARYOP_TRACE_LIMIT");
    if (value == nullptr || value[0] == '\0') {
      return 400;
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 ? parsed : 400;
  }();
  return limit;
}

inline int64_t next_binaryop_trace_id() {
  static std::atomic<int64_t> seq{0};
  return ++seq;
}

inline bool update_trace_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("PYTORCH_VULKAN_UPDATE_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

inline int64_t next_update_trace_id() {
  static std::atomic<int64_t> seq{0};
  return ++seq;
}

inline int64_t parse_positive_int_env(
    const char* env_name,
    const int64_t default_value) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  const long parsed = std::strtol(value, nullptr, 10);
  return parsed > 0 ? parsed : default_value;
}

inline int64_t update_guard_force_cpu_copy_at() {
  static const int64_t guard_at = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_GUARD_FORCE_CPU_COPY_AT",
      -1);
  return guard_at;
}

inline int64_t update_guard_force_cpu_copy_start() {
  static const int64_t guard_start = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_GUARD_FORCE_CPU_COPY_START",
      -1);
  return guard_start;
}

inline int64_t update_guard_force_cpu_copy_end() {
  static const int64_t guard_end = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_GUARD_FORCE_CPU_COPY_END",
      -1);
  return guard_end;
}

inline bool should_force_cpu_copy_for_update_id(const int64_t update_id) {
  const int64_t guard_at = update_guard_force_cpu_copy_at();
  if (guard_at > 0 && update_id == guard_at) {
    return true;
  }

  const int64_t guard_start = update_guard_force_cpu_copy_start();
  if (guard_start <= 0 || update_id < guard_start) {
    return false;
  }

  const int64_t guard_end = update_guard_force_cpu_copy_end();
  if (guard_end > 0 && guard_end < guard_start) {
    return false;
  }
  return guard_end <= 0 || update_id <= guard_end;
}

inline int64_t update_ordering_probe_flush_at() {
  static const int64_t guard_at = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_ORDERING_PROBE_FLUSH_AT",
      -1);
  return guard_at;
}

inline int64_t update_ordering_probe_flush_start() {
  static const int64_t guard_start = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_ORDERING_PROBE_FLUSH_START",
      -1);
  return guard_start;
}

inline int64_t update_ordering_probe_flush_end() {
  static const int64_t guard_end = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_ORDERING_PROBE_FLUSH_END",
      -1);
  return guard_end;
}

inline bool should_run_ordering_probe_flush_for_update_id(
    const int64_t update_id) {
  const int64_t guard_at = update_ordering_probe_flush_at();
  if (guard_at > 0 && update_id == guard_at) {
    return true;
  }

  const int64_t guard_start = update_ordering_probe_flush_start();
  if (guard_start <= 0 || update_id < guard_start) {
    return false;
  }

  const int64_t guard_end = update_ordering_probe_flush_end();
  if (guard_end > 0 && guard_end < guard_start) {
    return false;
  }
  return guard_end <= 0 || update_id <= guard_end;
}

enum class UpdateOrderingProbeMode final {
  SubmitAndFlush,
  SubmitOnly,
  SubmitAndWaitIdle,
  SubmitAndWaitIdleCleanup,
};

inline UpdateOrderingProbeMode update_ordering_probe_mode() {
  static const UpdateOrderingProbeMode mode = []() {
    const char* value =
        std::getenv("PYTORCH_VULKAN_UPDATE_ORDERING_PROBE_MODE");
    if (value == nullptr || value[0] == '\0') {
      return UpdateOrderingProbeMode::SubmitAndFlush;
    }
    if (std::strcmp(value, "submit_and_flush") == 0) {
      return UpdateOrderingProbeMode::SubmitAndFlush;
    }
    if (std::strcmp(value, "submit_only") == 0) {
      return UpdateOrderingProbeMode::SubmitOnly;
    }
    if (std::strcmp(value, "submit_and_wait_idle") == 0) {
      return UpdateOrderingProbeMode::SubmitAndWaitIdle;
    }
    if (std::strcmp(value, "submit_and_wait_idle_cleanup") == 0) {
      return UpdateOrderingProbeMode::SubmitAndWaitIdleCleanup;
    }
    return UpdateOrderingProbeMode::SubmitAndFlush;
  }();
  return mode;
}

inline uint32_t update_ordering_probe_cleanup_mask() {
  static const uint32_t mask = []() {
    const char* value =
        std::getenv("PYTORCH_VULKAN_UPDATE_ORDERING_PROBE_CLEANUP_MASK");
    if (value == nullptr || value[0] == '\0') {
      return api::Context::kFlushCleanupAll;
    }

    char* parse_end = nullptr;
    const unsigned long parsed = std::strtoul(value, &parse_end, 10);
    if (parse_end == value || *parse_end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
      return api::Context::kFlushCleanupAll;
    }

    const uint32_t requested_mask = static_cast<uint32_t>(parsed);
    const uint32_t active_mask =
        requested_mask & api::Context::kFlushCleanupAll;
    return (active_mask == 0u) ? api::Context::kFlushCleanupAll : active_mask;
  }();
  return mask;
}

enum class UpdateDescriptorFixMode final {
  Disabled,
  SubmitAndWaitIdleCleanup,
  SubmitAndWaitIdleCleanupPeriodic,
};

inline UpdateDescriptorFixMode update_descriptor_fix_mode() {
  static const UpdateDescriptorFixMode mode = []() {
    const char* value =
        std::getenv("PYTORCH_VULKAN_UPDATE_DESCRIPTOR_FIX_MODE");
    if (value == nullptr || value[0] == '\0') {
      return UpdateDescriptorFixMode::Disabled;
    }
    if (std::strcmp(value, "submit_and_wait_idle_cleanup") == 0 ||
        std::strcmp(value, "1") == 0) {
      return UpdateDescriptorFixMode::SubmitAndWaitIdleCleanup;
    }
    if (std::strcmp(value, "submit_and_wait_idle_cleanup_periodic") == 0 ||
        std::strcmp(value, "periodic") == 0) {
      return UpdateDescriptorFixMode::SubmitAndWaitIdleCleanupPeriodic;
    }
    return UpdateDescriptorFixMode::Disabled;
  }();
  return mode;
}

inline int64_t update_descriptor_fix_at() {
  static const int64_t guard_at = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_DESCRIPTOR_FIX_AT",
      -1);
  return guard_at;
}

inline int64_t update_descriptor_fix_start() {
  static const int64_t guard_start = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_DESCRIPTOR_FIX_START",
      -1);
  return guard_start;
}

inline int64_t update_descriptor_fix_end() {
  static const int64_t guard_end = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_DESCRIPTOR_FIX_END",
      -1);
  return guard_end;
}

inline int64_t update_descriptor_fix_period() {
  static const int64_t period = parse_positive_int_env(
      "PYTORCH_VULKAN_UPDATE_DESCRIPTOR_FIX_PERIOD",
      -1);
  return period;
}

inline int64_t update_descriptor_fix_offset() {
  static const int64_t offset = []() -> int64_t {
    const char* value =
        std::getenv("PYTORCH_VULKAN_UPDATE_DESCRIPTOR_FIX_OFFSET");
    if (value == nullptr || value[0] == '\0') {
      return 0;
    }
    char* parse_end = nullptr;
    const long parsed = std::strtol(value, &parse_end, 10);
    if (parse_end == value || *parse_end != '\0') {
      return 0;
    }
    return static_cast<int64_t>(parsed);
  }();
  return offset;
}

inline bool should_run_update_descriptor_fix_periodic_for_update_id(
    const int64_t update_id) {
  const int64_t period = update_descriptor_fix_period();
  if (period <= 0) {
    return false;
  }

  const int64_t offset = update_descriptor_fix_offset();
  if (offset < 0 || offset >= period) {
    return false;
  }

  return (update_id % period) == offset;
}

inline bool should_run_update_descriptor_fix_for_update_id(
    const int64_t update_id) {
  const UpdateDescriptorFixMode mode = update_descriptor_fix_mode();
  if (mode == UpdateDescriptorFixMode::Disabled) {
    return false;
  }
  if (mode == UpdateDescriptorFixMode::SubmitAndWaitIdleCleanupPeriodic) {
    return should_run_update_descriptor_fix_periodic_for_update_id(update_id);
  }

  const int64_t guard_at = update_descriptor_fix_at();
  if (guard_at > 0) {
    return update_id == guard_at;
  }

  const int64_t guard_start = update_descriptor_fix_start();
  if (guard_start > 0) {
    if (update_id < guard_start) {
      return false;
    }
    const int64_t guard_end = update_descriptor_fix_end();
    if (guard_end > 0 && guard_end < guard_start) {
      return false;
    }
    return guard_end <= 0 || update_id <= guard_end;
  }

  return true;
}

thread_local bool g_update_guard_armed_for_add_tensor = false;
thread_local bool g_update_guard_force_cpu_copy_next = false;
thread_local int64_t g_update_ordering_probe_flush_update_id = -1;
thread_local int64_t g_update_descriptor_fix_update_id = -1;

struct UpdateGuardArm final {
  UpdateGuardArm() {
    g_update_guard_armed_for_add_tensor = true;
  }
  ~UpdateGuardArm() {
    g_update_guard_armed_for_add_tensor = false;
  }
};

inline void arm_update_guard_force_cpu_copy_next() {
  g_update_guard_force_cpu_copy_next = true;
}

inline bool consume_update_guard_force_cpu_copy_next() {
  const bool should_force = g_update_guard_force_cpu_copy_next;
  g_update_guard_force_cpu_copy_next = false;
  return should_force;
}

inline void arm_update_ordering_probe_flush_after_copy(
    const int64_t update_id) {
  g_update_ordering_probe_flush_update_id = update_id;
}

inline int64_t consume_update_ordering_probe_flush_after_copy() {
  const int64_t update_id = g_update_ordering_probe_flush_update_id;
  g_update_ordering_probe_flush_update_id = -1;
  return update_id;
}

inline void arm_update_descriptor_fix_after_copy(const int64_t update_id) {
  g_update_descriptor_fix_update_id = update_id;
}

inline int64_t consume_update_descriptor_fix_after_copy() {
  const int64_t update_id = g_update_descriptor_fix_update_id;
  g_update_descriptor_fix_update_id = -1;
  return update_id;
}

inline const char* storage_type_name(const api::StorageType storage_type) {
  switch (storage_type) {
    case api::StorageType::BUFFER:
      return "BUFFER";
    case api::StorageType::TEXTURE_3D:
      return "TEXTURE_3D";
    default:
      return "OTHER";
  }
}

} // namespace

static Tensor binary_op_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::BinaryOutput);
    return vTensor{
        context,
        v_self.sizes(),
        v_self.dtype(),
    };
  }();

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    int fill0;
    float other;
  } block{
      v_self.extents(),
      0,
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

static Tensor binary_op_preprocess_other_arg(const Tensor& other_arg) {
  // Similar to binary_op_scalar where tensors is mapped to float, we
  // also map known integer types (but not quant types) tensor to float.

  // Such conversion can only to be done before moving to vulkan, since vulkan
  // doesn't yet support integer types.
  Tensor other = other_arg;
  if (!other.is_vulkan()) {
    switch (other.scalar_type()) {
      case at::kByte:
      case at::kChar:
      case at::kShort:
      case at::kInt:
      case at::kLong:
      case at::kDouble:
        other = other.to(kFloat);
        break;
      case at::kFloat:
        // No op for expected type.
        break;
      default:
        TORCH_CHECK(
            false,
            "binary_op_tensor, doesn't support type %s",
            other.scalar_type());
        break;
    }
    other = other.vulkan();
  }

  return other;
}

static Tensor& binary_op_scalar_(
    Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    int fill0;
    float other;
  } block{
      v_self.extents(),
      0,
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

static Tensor binary_op_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor_f16) {
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  Tensor other = binary_op_preprocess_other_arg(other_arg);

  vTensor& v_self_raw = convert(self);
  vTensor& v_other_raw = convert(other);
  const bool any_buffer = v_self_raw.storage_type() == api::StorageType::BUFFER ||
      v_other_raw.storage_type() == api::StorageType::BUFFER;
  const int64_t trace_id =
      binaryop_trace_enabled() ? next_binaryop_trace_id() : -1;
  if (trace_id > 0 && trace_id <= binaryop_trace_limit()) {
    std::cerr << "[vk_binaryop_trace] id=" << trace_id
              << " site=binary_op_tensor.entry any_buffer=" << any_buffer
              << " self_storage=" << storage_type_name(v_self_raw.storage_type())
              << " other_storage=" << storage_type_name(v_other_raw.storage_type())
              << " self_dtype=" << static_cast<int>(v_self_raw.dtype())
              << " other_dtype=" << static_cast<int>(v_other_raw.dtype())
              << " self_numel=" << self.numel() << " other_numel=" << other.numel()
              << "\n";
  }

  if (any_buffer) {
    if (v_self_raw.storage_type() != api::StorageType::BUFFER) {
      self = to_buffer_tensor(self);
    }
    if (v_other_raw.storage_type() != api::StorageType::BUFFER) {
      other = to_buffer_tensor(other);
    }
    vTensor& v_self = convert(self);
    vTensor& v_other = convert(other);

    TORCH_CHECK(
        v_self.storage_type() == api::StorageType::BUFFER &&
            v_other.storage_type() == api::StorageType::BUFFER,
        "Vulkan buffer binary op requires buffer-backed tensors.");
    TORCH_CHECK(
        v_self.dtype() == v_other.dtype(),
        "Vulkan buffer binary op requires matching dtypes.");
    TORCH_CHECK(
        buffer_dtype_supported(v_self),
        "Vulkan buffer binary op requires float32, or fp16 buffer storage to be enabled.");

    vTensor v_output = [&]() {
      api::AllocationTagScope tag_scope(api::AllocationTag::BinaryOutput);
      return vTensor{
          context,
          utils::broadcast_size(self_arg, other_arg),
          v_self.dtype(),
          api::StorageType::BUFFER,
          buffer_output_layout(v_self, v_other),
      };
    }();

    check_storage_buffer_limit(v_self, "binary op input");
    check_storage_buffer_limit(v_other, "binary op other");
    check_storage_buffer_limit(v_output, "binary op output");

    const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0f;
    const struct Block final {
      vec4 alpha;
    } block{
        {alpha, 0.0f, 0.0f, 0.0f},
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    const api::ShaderInfo& buffer_shader =
        select_buffer_shader(v_self, buffer_shader_descriptor, buffer_shader_descriptor_f16);
    if (trace_id > 0 && trace_id <= binaryop_trace_limit()) {
      std::cerr << "[vk_binaryop_trace] id=" << trace_id
                << " site=binary_op_tensor.buffer self_storage="
                << storage_type_name(v_self.storage_type())
                << " other_storage=" << storage_type_name(v_other.storage_type())
                << " use_f16_shader=" << (v_self.dtype() == api::kHalf)
                << " out_numel=" << v_output.gpu_numel() << "\n";
    }

    context->submit_compute_job(
        // shader descriptor
        buffer_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {safe_downcast<uint32_t>(v_output.gpu_numel()), 1u, 1u},
        // local work group size
        {32u, 1u, 1u},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_output.buffer_metadata(),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_self.buffer_metadata(),
        v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other.buffer_metadata(),
        params.buffer());

    return convert(v_output);
  }

  vTensor& v_self = v_self_raw;
  vTensor& v_other = v_other_raw;
  if (trace_id > 0 && trace_id <= binaryop_trace_limit()) {
    std::cerr << "[vk_binaryop_trace] id=" << trace_id
              << " site=binary_op_tensor.texture self_numel=" << self.numel()
              << " other_numel=" << other.numel() << "\n";
  }

  vTensor v_output = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::BinaryOutput);
    return vTensor{
        context,
        utils::broadcast_size(self_arg, other_arg),
        v_self.dtype(),
    };
  }();

  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;
  const struct Block final {
    uvec4 output_tensor_size;
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
    float alpha;
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},

      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha
      safe_downcast<float>(alpha),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

static Tensor quantized_binary_op_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point,
    const api::ShaderInfo& shader_descriptor) {
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);
  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  TORCH_CHECK(v_self.is_quantized(), "Input tensor is not quantized");
  TORCH_CHECK(v_other.is_quantized(), "Input tensor is not quantized");

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      scale,
      zero_point,
      api::kQUInt8,
  };

  const double scale1 = v_self.get_scale();
  const double scale2 = v_other.get_scale();
  const int64_t zero_point1 = v_self.get_zero_point();
  const int64_t zero_point2 = v_other.get_zero_point();
  const struct Block final {
    uvec3 extents;
    uint32_t channelSize;
    uvec3 input1Extents;
    uint32_t channelBatchSize1;
    uvec3 input2Extents;
    uint32_t channelBatchSize2;
    float scale1;
    float scale2;
    int32_t zeroPoint1;
    int32_t zeroPoint2;
    float scale;
    float fill1;
    int32_t zeroPoint;
    int32_t fill2;
  } block{
      v_output.extents(),
      get_dim<Dim4D::Channel>(v_output),
      v_self.extents(),
      get_dim<Dim4D::Channel>(self) * get_dim<Dim4D::Batch>(self),
      v_other.extents(),
      get_dim<Dim4D::Channel>(other) * get_dim<Dim4D::Batch>(other),
      safe_downcast<float>(scale1),
      safe_downcast<float>(scale2),
      safe_downcast<int32_t>(zero_point1),
      safe_downcast<int32_t>(zero_point2),
      safe_downcast<float>(scale),
      0.0f,
      safe_downcast<int32_t>(zero_point),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert_quantized(v_output);
}

static Tensor& binary_op_tensor_(
    Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_inplace_shader_descriptor,
    const api::ShaderInfo& buffer_inplace_shader_descriptor_f16) {
  TORCH_CHECK(
      get_dim<Dim4D::Batch>(self_arg) >= get_dim<Dim4D::Batch>(other_arg) &&
          get_dim<Dim4D::Channel>(self_arg) >=
              get_dim<Dim4D::Channel>(other_arg) &&
          get_dim<Dim4D::Height>(self_arg) >=
              get_dim<Dim4D::Height>(other_arg) &&
          get_dim<Dim4D::Width>(self_arg) >= get_dim<Dim4D::Width>(other_arg),
      "Dimensions of input tensor to Vulkan in-place binary elementwise op "
      "must be less than or equal the dimensions of the underlying tensor.");

  utils::is_broadcastable(self_arg, other_arg);

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  Tensor other = binary_op_preprocess_other_arg(other_arg);
  vTensor& v_other_raw = convert(other);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    if (v_other_raw.storage_type() != api::StorageType::BUFFER) {
      other = to_buffer_tensor(other);
    }
    vTensor& v_other = convert(other);

    TORCH_CHECK(
        v_other.storage_type() == api::StorageType::BUFFER,
        "Vulkan buffer binary op requires buffer-backed other tensor.");
    TORCH_CHECK(
        v_self.dtype() == v_other.dtype(),
        "Vulkan buffer binary op requires matching dtypes.");
    TORCH_CHECK(
        buffer_dtype_supported(v_self),
        "Vulkan buffer binary op requires float32, or fp16 buffer storage to be enabled.");

    check_storage_buffer_limit(v_self, "binary op self");
    check_storage_buffer_limit(v_other, "binary op other");

    const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0f;
    const struct Block final {
      vec4 alpha;
    } block{
        {alpha, 0.0f, 0.0f, 0.0f},
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    auto& self_buffer = v_self.buffer(
        pipeline_barrier,
        api::PipelineStage::COMPUTE,
        api::MemoryAccessType::READ | api::MemoryAccessType::WRITE);

    const api::ShaderInfo& buffer_shader = select_buffer_shader(
        v_self,
        buffer_inplace_shader_descriptor,
        buffer_inplace_shader_descriptor_f16);

    context->submit_compute_job(
        // shader descriptor
        buffer_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {safe_downcast<uint32_t>(v_self.gpu_numel()), 1u, 1u},
        // local work group size
        {32u, 1u, 1u},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        self_buffer,
        v_self.buffer_metadata(),
        v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other.buffer_metadata(),
        params.buffer());

    return self_arg;
  }

  if (v_other_raw.storage_type() == api::StorageType::BUFFER) {
    other = to_texture_tensor(other);
  }
  vTensor& v_other = convert(other);

  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;
  const struct Block final {
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
    float alpha;
  } block{
      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha
      safe_downcast<float>(alpha),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

static Tensor add_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar(
      self_arg, other, std::optional<Scalar>(alpha), VK_KERNEL(add_scalar));
}

static Tensor& add_scalar_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar_(
      self, other, std::optional<Scalar>(alpha), VK_KERNEL(add_scalar_inplace));
}

Tensor quantized_add(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_add));
}

Tensor quantized_sub(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_sub));
}

Tensor quantized_mul(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_mul));
}

Tensor quantized_div(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_div));
}

static Tensor add_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  const bool traced_update =
      self_arg.is_vulkan() && other_arg.is_vulkan() && alpha.to<float>() <= 0.0f;
  if (traced_update) {
    const int64_t update_id = next_update_trace_id();
    const bool cpu_guard_hit = g_update_guard_armed_for_add_tensor &&
        should_force_cpu_copy_for_update_id(update_id);
    const bool ordering_probe_hit = g_update_guard_armed_for_add_tensor &&
        should_run_ordering_probe_flush_for_update_id(update_id);
    const bool descriptor_fix_hit = g_update_guard_armed_for_add_tensor &&
        should_run_update_descriptor_fix_for_update_id(update_id);
    if (cpu_guard_hit) {
      arm_update_guard_force_cpu_copy_next();
      std::cerr << "[vk_update_guard] id=" << update_id
                << " site=add_tensor action=force_cpu_copy\n";
    }
    if (ordering_probe_hit) {
      arm_update_ordering_probe_flush_after_copy(update_id);
      std::cerr << "[vk_update_ordering_probe] id=" << update_id
                << " site=add_tensor action=arm_flush_after_copy\n";
    }
    if (descriptor_fix_hit) {
      arm_update_descriptor_fix_after_copy(update_id);
      std::cerr << "[vk_update_fix_candidate] id=" << update_id
                << " site=add_tensor action=arm_descriptor_cleanup_after_copy\n";
    }
    if (update_trace_enabled()) {
      const bool guard_hit =
          cpu_guard_hit || ordering_probe_hit || descriptor_fix_hit;
      std::cerr << "[vk_update_trace] id=" << update_id
                << " site=add_tensor alpha=" << alpha.to<float>()
                << " self_numel=" << self_arg.numel()
                << " other_numel=" << other_arg.numel()
                << " guard_hit=" << guard_hit
                << " guard_cpu_hit=" << cpu_guard_hit
                << " guard_flush_hit=" << ordering_probe_hit
                << " guard_fix_hit=" << descriptor_fix_hit << "\n";
    }
  }
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add),
      VK_KERNEL(add_buffer),
      VK_KERNEL(add_buffer_f16));
}

static Tensor& add_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  if (self.is_vulkan()) {
    const AddTensorInplaceMode inplace_mode = add_tensor_inplace_mode();
    const int64_t trace_id =
        binaryop_trace_enabled() ? next_binaryop_trace_id() : -1;
    if (trace_id > 0 && trace_id <= binaryop_trace_limit()) {
      const vTensor& v_self = convert(self);
      std::cerr << "[vk_binaryop_trace] id=" << trace_id
                << " site=add_tensor_.entry alpha=" << alpha.to<float>()
                << " inplace_mode="
                << add_tensor_inplace_mode_name(inplace_mode)
                << " self_storage=" << storage_type_name(v_self.storage_type())
                << " self_dtype=" << static_cast<int>(v_self.dtype())
                << " self_numel=" << self.numel()
                << " other_is_vulkan=" << other_arg.is_vulkan() << "\n";
    }
    if (inplace_mode == AddTensorInplaceMode::Direct) {
      return binary_op_tensor_(
          self,
          other_arg,
          std::optional<Scalar>(alpha),
          VK_KERNEL(add_inplace),
          VK_KERNEL(add_buffer_inplace),
          VK_KERNEL(add_buffer_f16_inplace));
    }
    // Preserve the historical wrapper path by default; the direct path is
    // env-gated for add_.Tensor revalidation.
    UpdateGuardArm guard_scope{};
    Tensor out = add_tensor(self, other_arg, alpha);
    const bool force_cpu_copy = consume_update_guard_force_cpu_copy_next();
    const int64_t ordering_flush_update_id =
        consume_update_ordering_probe_flush_after_copy();
    const int64_t descriptor_fix_update_id =
        consume_update_descriptor_fix_after_copy();
    if (trace_id > 0 && trace_id <= binaryop_trace_limit()) {
      const vTensor& v_out = convert(out);
      std::cerr << "[vk_binaryop_trace] id=" << trace_id
                << " site=add_tensor_.after_add out_storage="
                << storage_type_name(v_out.storage_type())
                << " out_dtype=" << static_cast<int>(v_out.dtype())
                << " out_numel=" << out.numel() << "\n";
    }
    if (force_cpu_copy) {
      std::cerr << "[vk_update_guard] site=add_tensor_.copy action=cpu_roundtrip\n";
      Tensor out_cpu = out.cpu();
      self.copy_(out_cpu);
    } else {
      self.copy_(out);
    }
    if (ordering_flush_update_id > 0) {
      api::Context* const context = api::context();
      std::unique_lock<std::mutex> context_lock(context->dispatch_lock());
      const UpdateOrderingProbeMode probe_mode = update_ordering_probe_mode();
      context->submit_cmd_to_gpu(VK_NULL_HANDLE);
      if (probe_mode == UpdateOrderingProbeMode::SubmitOnly) {
        std::cerr << "[vk_update_ordering_probe] id=" << ordering_flush_update_id
                  << " site=add_tensor_.copy action=submit_only\n";
      } else if (probe_mode == UpdateOrderingProbeMode::SubmitAndWaitIdle) {
        VK_CHECK(vkQueueWaitIdle(context->queue()));
        std::cerr << "[vk_update_ordering_probe] id=" << ordering_flush_update_id
                  << " site=add_tensor_.copy action=submit_wait_idle\n";
      } else if (
          probe_mode == UpdateOrderingProbeMode::SubmitAndWaitIdleCleanup) {
        VK_CHECK(vkQueueWaitIdle(context->queue()));
        const uint32_t cleanup_mask = update_ordering_probe_cleanup_mask();
        context->flush_submit_cleanup(cleanup_mask);
        std::cerr << "[vk_update_ordering_probe] id=" << ordering_flush_update_id
                  << " site=add_tensor_.copy action=submit_wait_idle_cleanup"
                  << " cleanup_mask=" << cleanup_mask << "\n";
      } else {
        context->flush();
        std::cerr << "[vk_update_ordering_probe] id=" << ordering_flush_update_id
                  << " site=add_tensor_.copy action=submit_flush\n";
      }
    }
    if (descriptor_fix_update_id > 0) {
      api::Context* const context = api::context();
      std::unique_lock<std::mutex> context_lock(context->dispatch_lock());
      context->submit_cmd_to_gpu(VK_NULL_HANDLE);
      const UpdateDescriptorFixMode descriptor_fix_mode =
          update_descriptor_fix_mode();
      if (
          descriptor_fix_mode ==
              UpdateDescriptorFixMode::SubmitAndWaitIdleCleanup ||
          descriptor_fix_mode ==
              UpdateDescriptorFixMode::SubmitAndWaitIdleCleanupPeriodic) {
        VK_CHECK(vkQueueWaitIdle(context->queue()));
        constexpr uint32_t cleanup_mask =
            api::Context::kFlushCleanupDescriptorPool;
        context->flush_submit_cleanup(cleanup_mask);
        std::cerr << "[vk_update_fix_candidate] id=" << descriptor_fix_update_id
                  << " site=add_tensor_.copy"
                  << " action=submit_wait_idle_cleanup"
                  << " cleanup_mask=" << cleanup_mask << "\n";
      }
    }
    return self;
  }
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add_inplace),
      VK_KERNEL(add_buffer_inplace),
      VK_KERNEL(add_buffer_f16_inplace));
}

static Tensor sub_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar));
}

static Tensor& sub_scalar_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar_inplace));
}

static Tensor sub_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(sub),
      VK_KERNEL(sub_buffer),
      VK_KERNEL(sub_buffer_f16));
}

static Tensor& sub_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  if (self.is_vulkan()) {
    // Avoid unstable Vulkan in-place binary kernels: run out-of-place then copy
    // back into self.
    Tensor out = sub_tensor(self, other_arg, alpha);
    self.copy_(out);
    return self;
  }
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(sub_inplace),
      VK_KERNEL(sub_buffer_inplace),
      VK_KERNEL(sub_buffer_f16_inplace));
}

static Tensor mul_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg, other, std::optional<Scalar>(), VK_KERNEL(mul_scalar));
}

static Tensor& mul_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self, other, std::optional<Scalar>(), VK_KERNEL(mul_scalar_inplace));
}

static Tensor mul_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(mul),
      VK_KERNEL(mul_buffer),
      VK_KERNEL(mul_buffer_f16));
}

static Tensor& mul_tensor_(Tensor& self, const Tensor& other_arg) {
  if (self.is_vulkan()) {
    // Avoid unstable Vulkan in-place binary kernels: run out-of-place then copy
    // back into self.
    Tensor out = mul_tensor(self, other_arg);
    self.copy_(out);
    return self;
  }
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(mul_inplace),
      VK_KERNEL(mul_buffer_inplace),
      VK_KERNEL(mul_buffer_f16_inplace));
}

static Tensor div_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar));
}

static Tensor& div_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar_inplace));
}

static Tensor div_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(div),
      VK_KERNEL(div_buffer),
      VK_KERNEL(div_buffer_f16));
}

static Tensor& div_tensor_(Tensor& self, const Tensor& other_arg) {
  if (self.is_vulkan()) {
    // Avoid unstable Vulkan in-place binary kernels: run out-of-place then copy
    // back into self.
    Tensor out = div_tensor(self, other_arg);
    self.copy_(out);
    return self;
  }
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(div_inplace),
      VK_KERNEL(div_buffer_inplace),
      VK_KERNEL(div_buffer_f16_inplace));
}

static Tensor pow(const Tensor& self, const Tensor& other) {
  return binary_op_tensor(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow),
      VK_KERNEL(pow_buffer),
      VK_KERNEL(pow_buffer_f16));
}

static Tensor& pow_(Tensor& self, const Tensor& other) {
  if (self.is_vulkan()) {
    // Avoid unstable Vulkan in-place binary kernels: run out-of-place then copy
    // back into self.
    Tensor out = pow(self, other);
    self.copy_(out);
    return self;
  }
  return binary_op_tensor_(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_inplace),
      VK_KERNEL(pow_buffer_inplace),
      VK_KERNEL(pow_buffer_f16_inplace));
}

static Tensor pow_tensor_scalar(const Tensor& self, const Scalar& other) {
  return binary_op_scalar(
      self, other, std::optional<Scalar>(), VK_KERNEL(pow_tensor_scalar));
}

static Tensor& pow_tensor_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_tensor_scalar_inplace));
}

static Tensor pow_scalar_tensor(const Scalar& self, const Tensor& other) {
  return binary_op_scalar(
      other, self, std::optional<Scalar>(), VK_KERNEL(pow_scalar_tensor));
}

static Tensor floor_divide_scalar(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar: can't divide by zero");
  return binary_op_scalar(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar));
}

static Tensor& floor_divide_scalar_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar_: can't divide by zero");
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar_inplace));
}

static Tensor floor_divide_tensor(const Tensor& self, const Tensor& other) {
  if (self.is_vulkan() || other.is_vulkan()) {
    const Tensor self_vk = self.is_vulkan() ? self : self.vulkan();
    const Tensor other_vk = other.is_vulkan() ? other : other.vulkan();
    vTensor& v_self = convert(self_vk);
    vTensor& v_other = convert(other_vk);
    const bool any_buffer =
        v_self.storage_type() == api::StorageType::BUFFER ||
        v_other.storage_type() == api::StorageType::BUFFER;
    const bool any_half =
        v_self.dtype() == api::kHalf || v_other.dtype() == api::kHalf;
    if (any_buffer && any_half) {
      Tensor self_cpu = self_vk.cpu().to(at::kFloat);
      Tensor other_cpu = other_vk.cpu().to(at::kFloat);
      Tensor out_cpu = at::floor_divide(self_cpu, other_cpu).to(at::kHalf);
      return out_cpu.to(at::kVulkan);
    }
  }
  return binary_op_tensor(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(floor_divide),
      VK_KERNEL(floor_divide_buffer),
      VK_KERNEL(floor_divide_buffer_f16));
}

static Tensor& floor_divide_tensor_(Tensor& self, const Tensor& other_arg) {
  if (self.is_vulkan()) {
    // Avoid unstable Vulkan in-place binary kernels: run out-of-place then copy
    // back into self.
    Tensor out = floor_divide_tensor(self, other_arg);
    self.copy_(out);
    return self;
  }
  if (self.is_vulkan()) {
    const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
    vTensor& v_self = convert(self);
    vTensor& v_other = convert(other);
    const bool any_buffer =
        v_self.storage_type() == api::StorageType::BUFFER ||
        v_other.storage_type() == api::StorageType::BUFFER;
    const bool any_half =
        v_self.dtype() == api::kHalf || v_other.dtype() == api::kHalf;
    if (any_buffer && any_half) {
      Tensor out = floor_divide_tensor(self, other);
      self.copy_(out);
      return self;
    }
  }
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(floor_divide_inplace),
      VK_KERNEL(floor_divide_buffer_inplace),
      VK_KERNEL(floor_divide_buffer_f16_inplace));
}

static Tensor eq_scalar(const Tensor& self_arg, const Scalar& other) {
  if (!self_arg.is_vulkan()) {
    return at::eq(self_arg, other);
  }

  if (!self_arg.is_floating_point()) {
    const Tensor self_cpu = self_arg.cpu();
    Tensor out_cpu = at::eq(self_cpu, other);
    return out_cpu.to(at::kVulkan);
  }

  api::Context* const context = api::context();
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::BinaryOutput);
    return vTensor{
        context,
        v_self.sizes(),
        api::kBool,
    };
  }();

  const float other_val = other.to<float>();
  const struct Block final {
    uvec3 extents;
    int32_t fill0;
    float other;
  } block{
      v_self.extents(),
      0,
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(eq_scalar),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

static Tensor eq_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  if (!self_arg.is_vulkan() && !other_arg.is_vulkan()) {
    return at::eq(self_arg, other_arg);
  }

  if (!self_arg.is_floating_point() || !other_arg.is_floating_point()) {
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    const Tensor other_cpu = other_arg.is_vulkan() ? other_arg.cpu() : other_arg;
    Tensor out_cpu = at::eq(self_cpu, other_cpu);
    return out_cpu.to(at::kVulkan);
  }

  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);
  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  vTensor v_output = [&]() {
    api::AllocationTagScope tag_scope(api::AllocationTag::BinaryOutput);
    return vTensor{
        context,
        utils::broadcast_size(self_arg, other_arg),
        api::kBool,
    };
  }();

  const struct Block final {
    uvec4 output_tensor_size;
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},
      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},
      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(eq_tensor),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Scalar"), TORCH_FN(add_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Scalar"), TORCH_FN(add_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Scalar"), TORCH_FN(sub_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Scalar"), TORCH_FN(sub_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Scalar"), TORCH_FN(mul_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Scalar"), TORCH_FN(mul_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Scalar"), TORCH_FN(div_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Scalar"), TORCH_FN(div_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow.Tensor_Tensor"), TORCH_FN(pow));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow_.Tensor"), TORCH_FN(pow_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pow.Tensor_Scalar"),
      TORCH_FN(pow_tensor_scalar));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pow_.Scalar"), TORCH_FN(pow_tensor_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow.Scalar"), TORCH_FN(pow_scalar_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide.Scalar"),
      TORCH_FN(floor_divide_scalar));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide_.Scalar"),
      TORCH_FN(floor_divide_scalar_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide"),
      TORCH_FN(floor_divide_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide_.Tensor"),
      TORCH_FN(floor_divide_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::eq.Scalar"), TORCH_FN(eq_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::eq.Tensor"), TORCH_FN(eq_tensor));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
#endif /* USE_VULKAN_API */
