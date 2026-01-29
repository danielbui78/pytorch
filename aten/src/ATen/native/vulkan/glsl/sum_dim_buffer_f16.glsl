#version 450 core

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

#define FP_PRECISION ${PRECISION}   // mediump/highp for floats if desired
#define INT_PRECISION highp         // always high precision for indices/metadata
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0) buffer FP_PRECISION restrict writeonly OutBuffer {
  float16_t data[];
}
uOutput;

layout(set = 0, binding = 1) uniform INT_PRECISION restrict OutMeta {
  highp uvec4 sizes;
  highp uvec4 strides;
  highp uint ndim;
  highp uint buf_length;
}
uOutMeta;

layout(set = 0, binding = 2) buffer FP_PRECISION restrict readonly InBuffer {
  float16_t data[];
}
uInput;

layout(set = 0, binding = 3) uniform INT_PRECISION restrict InMeta {
  highp uvec4 sizes;
  highp uvec4 strides;
  highp uint ndim;
  highp uint buf_length;
}
uInMeta;

layout(set = 0, binding = 4) uniform INT_PRECISION restrict Block {
  highp uvec2 dim_info;
}
uBlock;

#include "indexing.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.buf_length) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.strides, uOutMeta.sizes);
  highp uvec4 in_coord = write_coord;

  highp float accumulator = 0.0;
  for (highp uint i = 0u; i < uBlock.dim_info.y; ++i) {
    in_coord[uBlock.dim_info.x] = i;
    const highp uint in_idx = coord_to_idx(in_coord, uInMeta.strides);
    if (in_idx >= uInMeta.buf_length) {
      continue;
    }
    accumulator += float(uInput.data[in_idx]);
  }

  uOutput.data[write_idx] = float16_t(accumulator);
}
