#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uOutMeta;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uInMeta;

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  uvec2 dim_info;
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
  uvec4 in_coord = write_coord;

  float accumulator = 0.0;
  for (uint i = 0u; i < uBlock.dim_info.y; ++i) {
    in_coord[uBlock.dim_info.x] = i;
    const uint in_idx = coord_to_idx(in_coord, uInMeta.strides);
    accumulator += uInput.data[in_idx];
  }

  uOutput.data[write_idx] = accumulator;
}
