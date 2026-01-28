#version 450 core

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#include "indexing.h"

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float16_t data[];
}
uOutput;

/*
 * Output Buffer Metadata
 */
layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uOutMeta;

/*
 * Input Buffer
 */
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  float16_t data[];
}
uInput;

/*
 * Input Buffer Metadata
 */
layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uInMeta;

/*
 * Weight Buffer
 */
layout(set = 0, binding = 4) buffer PRECISION restrict readonly WBuffer {
  float16_t data[];
}
uWeight;

/*
 * Weight Buffer Metadata
 */
layout(set = 0, binding = 5) uniform PRECISION restrict WMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uWMeta;

/*
 * Bias Buffer
 */
layout(set = 0, binding = 6) buffer PRECISION restrict readonly BiasBuffer {
  float16_t data[];
}
uBias;

/*
 * Bias Buffer Metadata
 */
layout(set = 0, binding = 7) uniform PRECISION restrict BiasMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uBiasMeta;

/*
 * Params
 */
layout(set = 0, binding = 8) uniform PRECISION restrict Params {
  vec4 alpha_beta;
}
uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint col = gl_GlobalInvocationID.x;
  const uint row = gl_GlobalInvocationID.y;

  const uint M = uInMeta.sizes.y;
  const uint K = uInMeta.sizes.x;
  const uint N = uWMeta.sizes.x;

  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0;
  for (uint k = 0; k < K; ++k) {
    const uint idxA = coord_to_idx(uvec4(k, row, 0, 0), uInMeta.strides);
    const uint idxB = coord_to_idx(uvec4(col, k, 0, 0), uWMeta.strides);
    acc += float(uInput.data[idxA]) * float(uWeight.data[idxB]);
  }

  const uint bias_len = uBiasMeta.sizes.x;
  const uint bias_idx = (bias_len == 1u)
      ? 0u
      : coord_to_idx(uvec4(col, 0, 0, 0), uBiasMeta.strides);
  const float bias_val = float(uBias.data[bias_idx]);

  const float alpha = uParams.alpha_beta.x;
  const float beta = uParams.alpha_beta.y;

  const uint idxOut = coord_to_idx(uvec4(col, row, 0, 0), uOutMeta.strides);
  uOutput.data[idxOut] = float16_t(acc * alpha + bias_val * beta);
}
