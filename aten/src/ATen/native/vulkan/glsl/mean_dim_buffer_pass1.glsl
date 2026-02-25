#version 450 core

#define FP_PRECISION ${PRECISION}
#define INT_PRECISION highp
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0) buffer FP_PRECISION restrict writeonly OutBuffer {
  float data[];
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
  float data[];
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
  highp uint block_size;
}
uBlock;

#include "indexing.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const highp uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.buf_length) {
    return;
  }

  const highp uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.strides, uOutMeta.sizes);
  highp uvec4 in_coord = write_coord;

  const highp uint axis = uBlock.dim_info.x;
  const highp uint dim_size = uBlock.dim_info.y;
  const highp uint block_id = write_coord[axis];
  const highp uint start = block_id * uBlock.block_size;
  const highp uint end = min(start + uBlock.block_size, dim_size);

  highp float accumulator = 0.0;
  for (highp uint i = start; i < end; ++i) {
    in_coord[axis] = i;
    const highp uint in_idx = coord_to_idx(in_coord, uInMeta.strides);
    if (in_idx < uInMeta.buf_length) {
      accumulator += uInput.data[in_idx];
    }
  }

  uOutput.data[write_idx] = accumulator;
}
