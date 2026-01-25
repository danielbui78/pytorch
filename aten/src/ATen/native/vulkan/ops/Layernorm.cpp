#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <ATen/Context.h>
#include <c10/util/irange.h>

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

LayernormPackedContext::LayernormPackedContext(
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    double eps)
    : unpacked_{c10::AnyType::get()} {
  packed_.reserve(ListArgs::kNumArgs);

  TORCH_CHECK(weight, "Weight must be provided!");
  packed_.emplace_back(weight->vulkan());
  TORCH_CHECK(bias, "Bias must be provided!");
  packed_.emplace_back(bias->vulkan());
  packed_.emplace_back(eps);

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(ListArgs::kNumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
    unpacked_.emplace_back(eps);
  }
}

LayernormPackedContext LayernormPackedContext::pack(
    c10::impl::GenericList unpacked) {
  return LayernormPackedContext(
      get_optional_tensor(unpacked, ListArgs::kWeight),
      get_optional_tensor(unpacked, ListArgs::kBias),
      unpacked.get(ListArgs::kEps).toDouble());
}

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps) {
  return c10::make_intrusive<LayernormPackedContext>(
      LayernormPackedContext(weight, bias, eps));
}

Tensor run_layernorm_context(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context) {
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  const std::optional<Tensor>& weight_opt =
      layernorm_context->get_val(LayernormPackedContext::ListArgs::kWeight)
          .toTensor();
  const std::optional<Tensor>& bias_opt =
      layernorm_context->get_val(LayernormPackedContext::ListArgs::kBias)
          .toTensor();
  const float eps = api::utils::safe_downcast<float>(
      layernorm_context->get_val(LayernormPackedContext::ListArgs::kEps)
          .toDouble());

  // We invoke native_layer_norm which returns a tuple of tensors: <layer_norm,
  // mean, 1/sqrt(var+eps)>, but we only need the first tensor (layer_norm).
  std::tuple<Tensor, Tensor, Tensor> native_layer_norm_output =
      at::native_layer_norm(input, normalized_shape, weight_opt, bias_opt, eps);
  return std::get<0>(native_layer_norm_output);
}

static Tensor layer_norm(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  return run_layernorm_context(
      input_arg,
      normalized_shape,
      c10::make_intrusive<LayernormPackedContext>(
          LayernormPackedContext(weight_opt, bias_opt, eps)));
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
    const Tensor& grad_out,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    std::array<bool, 3> output_mask) {
  const Tensor grad_cpu = grad_out.is_vulkan() ? grad_out.cpu() : grad_out;
  const Tensor input_cpu = input.is_vulkan() ? input.cpu() : input;
  const Tensor mean_cpu = mean.is_vulkan() ? mean.cpu() : mean;
  const Tensor rstd_cpu = rstd.is_vulkan() ? rstd.cpu() : rstd;
  const std::optional<Tensor> weight_cpu =
      (weight && weight->is_vulkan()) ? std::optional<Tensor>(weight->cpu())
                                      : weight;
  const std::optional<Tensor> bias_cpu =
      (bias && bias->is_vulkan()) ? std::optional<Tensor>(bias->cpu())
                                   : bias;

  auto grads = at::native_layer_norm_backward(
      grad_cpu,
      input_cpu,
      normalized_shape,
      mean_cpu,
      rstd_cpu,
      weight_cpu,
      bias_cpu,
      output_mask);

  Tensor grad_input = std::get<0>(grads);
  Tensor grad_weight = std::get<1>(grads);
  Tensor grad_bias = std::get<2>(grads);

  if (output_mask[0] && grad_input.defined()) {
    grad_input = grad_input.to(at::kVulkan);
  }
  if (output_mask[1] && grad_weight.defined()) {
    grad_weight = grad_weight.to(at::kVulkan);
  }
  if (output_mask[2] && grad_bias.defined()) {
    grad_bias = grad_bias.to(at::kVulkan);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// NOTE: We intentionally do not register aten::layer_norm for Vulkan here.
// This avoids CompositeImplicitAutograd dispatch ambiguity and allows
// composite lowering to select Vulkan-supported primitives.

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_layer_norm_backward"),
      TORCH_FN(native_layer_norm_backward));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
