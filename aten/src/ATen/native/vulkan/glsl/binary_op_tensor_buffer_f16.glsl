#version 450 core

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

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
 * Other Buffer
 */
layout(set = 0, binding = 4) buffer PRECISION restrict readonly OtherBuffer {
  float16_t data[];
}
uOther;

/*
 * Other Buffer Metadata
 */
layout(set = 0, binding = 5) uniform PRECISION restrict OtherMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uOtherMeta;

/*
 * Params
 */
layout(set = 0, binding = 6) uniform PRECISION restrict Params {
  vec4 alpha;
}
uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.buf_length) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.strides, uOutMeta.sizes);

  uvec4 in_coord = write_coord;
  in_coord.x = (uInMeta.sizes.x == 1u) ? 0u : in_coord.x;
  in_coord.y = (uInMeta.sizes.y == 1u) ? 0u : in_coord.y;
  in_coord.z = (uInMeta.sizes.z == 1u) ? 0u : in_coord.z;
  in_coord.w = (uInMeta.sizes.w == 1u) ? 0u : in_coord.w;

  uvec4 other_coord = write_coord;
  other_coord.x = (uOtherMeta.sizes.x == 1u) ? 0u : other_coord.x;
  other_coord.y = (uOtherMeta.sizes.y == 1u) ? 0u : other_coord.y;
  other_coord.z = (uOtherMeta.sizes.z == 1u) ? 0u : other_coord.z;
  other_coord.w = (uOtherMeta.sizes.w == 1u) ? 0u : other_coord.w;

  const uint in_idx = coord_to_idx(in_coord, uInMeta.strides);
  const uint other_idx = coord_to_idx(other_coord, uOtherMeta.strides);

  const float in_val = float(uInput.data[in_idx]);
  const float other_val = float(uOther.data[other_idx]);

  uOutput.data[write_idx] =
      float16_t(OP(in_val, other_val, uParams.alpha.x));
}
