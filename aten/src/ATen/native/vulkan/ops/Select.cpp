#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/nll_loss_forward.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;
Tensor select_batch_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[1], v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (c, h, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, h, c / 4)[c % 4]
  */
  const struct Block final {
    ivec2 batch_info;
  } block{
      {static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index)}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_batch_4d),
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

Tensor select_depth_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[1], v_input_sizes[2]},
      v_input.dtype(),
  };

  const struct Block final {
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_output.extents().data[0u]),
       static_cast<int32_t>(v_output.extents().data[1u]),
       static_cast<int32_t>(v_output.extents().data[2u]),
       static_cast<int32_t>(index)}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_depth_3d),
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

Tensor select_depth_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, h, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, h, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index),
       0}};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_depth_4d),
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

Tensor select_height_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2]},
      v_input.dtype(),
  };
  // Input tensor is a (c, h, w)
  // Output tensor is a (c, w)
  // In shader, the input texture's coordinate is (w, h, c)
  // In shader, the output texture's coordinate is (w, c, 1)
  uint32_t w = v_output.extents().data[0u];
  uint32_t c = v_output.extents().data[1u];
  uint32_t z = 1;
  const struct Block final {
    ivec4 height_info;
  } block{
      {static_cast<int32_t>(w),
       static_cast<int32_t>(c),
       static_cast<int32_t>(z),
       static_cast<int32_t>(index)}};

  // Encoding of c-channel is packed into texel, hence we only call ceil(c/4)
  // times to minimize invocation and read.
  // For the last dimension, it is the selected height. Shader will do a direct
  // lookup based on block.index.
  uvec3 global_workgroup_size{w, api::utils::div_up(c, 4u), z};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_height_3d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_size,
      // local work group size
      adaptive_work_group_size(global_workgroup_size),
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

Tensor select_height_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, c, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, c, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 height_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index),
       0}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_height_4d),
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

Tensor select_width_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1]},
      v_input.dtype(),
  };

  const struct Block final {
    ivec4 width_info;
  } block{
      {static_cast<int32_t>(v_output.extents().data[0u]),
       static_cast<int32_t>(v_output.extents().data[1u]),
       static_cast<int32_t>(v_output.extents().data[2u]),
       static_cast<int32_t>(index)}};

  // Input tensor is a (c, h, w)
  // Output tensor is a (c, h)
  // In shader, the input texture's coordinate is (w, h, c)
  // In shader, the output texture's coordinate is (h, c, 1)
  uint32_t h = v_output.extents().data[0u];
  uint32_t c = v_output.extents().data[1u];

  // Encoding of c-channel is packed into texel, hence we only call ceil(c/4)
  // times to minimize invocation and read.
  // For the last dimension, it is the selected width. Shader will do a direct
  // lookup based on block.index.
  uvec3 global_workgroup_size{h, api::utils::div_up(c, 4u), 1};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_width_3d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_size,
      // local work group size
      adaptive_work_group_size(global_workgroup_size),
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

Tensor select_width_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[2]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, c, h)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (h, c, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 width_info;
  } block{
      static_cast<int32_t>(v_input_sizes[0]),
      static_cast<int32_t>(std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
      static_cast<int32_t>(index),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_width_4d),
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

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  TORCH_CHECK(
      self.dim() == 3 || self.dim() == 4,
      "Vulkan select only supports 3d and 4d tensors!");

  const int64_t size = self.size(dim);

  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        "select(): index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.dim() == 3) {
    if (dim == 0) {
      return select_depth_3d(self, index);
    } else if (dim == 1) {
      return select_height_3d(self, index);
    } else {
      return select_width_3d(self, index);
    }
  } else { // self.dim() == 4
    if (dim == 0) {
      return select_batch_4d(self, index);
    } else if (dim == 1) {
      return select_depth_4d(self, index);
    } else if (dim == 2) {
      return select_height_4d(self, index);
    } else {
      return select_width_4d(self, index);
    }
  }
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
  const Tensor self_vk = self.is_vulkan() ? self : self.vulkan();
  const Tensor index_cpu = index.device().is_cpu() ? index : index.cpu();
  const Tensor output_cpu = self_vk.cpu().index_select(dim, index_cpu);
  return output_cpu.vulkan();
}

Tensor embedding_dense_backward(
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  const Tensor grad_output_cpu =
      grad_output.device().is_cpu() ? grad_output : grad_output.cpu();
  const Tensor indices_cpu = indices.device().is_cpu() ? indices : indices.cpu();
  const Tensor grad_weight_cpu = at::embedding_dense_backward(
      grad_output_cpu,
      indices_cpu,
      num_weights,
      padding_idx,
      scale_grad_by_freq);
  return grad_weight_cpu.vulkan();
}

std::tuple<Tensor, Tensor> nll_loss_forward(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    c10::SymInt ignore_index) {
  const Tensor self_cpu = self.device().is_cpu() ? self : self.cpu();
  const Tensor target_cpu = target.device().is_cpu() ? target : target.cpu();
  const std::optional<Tensor> weight_cpu =
      weight.has_value()
      ? std::optional<Tensor>(
            weight->device().is_cpu() ? *weight : weight->cpu())
      : std::nullopt;

  auto [output_cpu, total_weight_cpu] = at::nll_loss_forward_symint(
      self_cpu,
      target_cpu,
      weight_cpu,
      reduction,
      ignore_index);

  return std::make_tuple(output_cpu.vulkan(), total_weight_cpu.vulkan());
}

std::tuple<Tensor&, Tensor&> nll_loss_forward_output(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    c10::SymInt ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  auto [output_bridge, total_weight_bridge] = nll_loss_forward(
      self,
      target,
      weight,
      reduction,
      ignore_index);
  output.copy_(output_bridge);
  total_weight.copy_(total_weight_bridge);
  return std::tuple<Tensor&, Tensor&>{output, total_weight};
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::select.int"), TORCH_FN(select));
  m.impl(TORCH_SELECTIVE_NAME("aten::index_select"), TORCH_FN(index_select));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::embedding_dense_backward"),
      TORCH_FN(embedding_dense_backward));
  m.impl(TORCH_SELECTIVE_NAME("aten::nll_loss_forward"), TORCH_FN(nll_loss_forward));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::nll_loss_forward.output"),
      TORCH_FN(nll_loss_forward_output));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
