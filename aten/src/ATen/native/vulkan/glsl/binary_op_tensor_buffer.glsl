#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define INPLACE ${INPLACE}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

#include "indexing.h"

layout(std430) buffer;

// clang-format off
$if not INPLACE:
  /*
   * Output Buffer
   */
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    float data[];
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
    float data[];
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
    float data[];
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
$else:
  /*
   * Output Buffer
   */
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    float data[];
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
   * Other Buffer
   */
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly OtherBuffer {
    float data[];
  }
  uOther;

  /*
   * Other Buffer Metadata
   */
  layout(set = 0, binding = 3) uniform PRECISION restrict OtherMeta {
    uvec4 sizes;
    uvec4 strides;
    uint ndim;
    uint buf_length;
  }
  uOtherMeta;

  /*
   * Params
   */
  layout(set = 0, binding = 4) uniform PRECISION restrict Params {
    vec4 alpha;
  }
  uParams;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.buf_length) {
    return;
  }

  const uvec4 logical_strides = uvec4(
      1u,
      uOutMeta.sizes.x,
      uOutMeta.sizes.x * uOutMeta.sizes.y,
      uOutMeta.sizes.x * uOutMeta.sizes.y * uOutMeta.sizes.z);

  const uvec4 write_coord =
      idx_to_coord(write_idx, logical_strides, uOutMeta.sizes);

  uvec4 other_coord = write_coord;
  other_coord.x = (uOtherMeta.sizes.x == 1u) ? 0u : other_coord.x;
  other_coord.y = (uOtherMeta.sizes.y == 1u) ? 0u : other_coord.y;
  other_coord.z = (uOtherMeta.sizes.z == 1u) ? 0u : other_coord.z;
  other_coord.w = (uOtherMeta.sizes.w == 1u) ? 0u : other_coord.w;

  const bool other_same_shape = all(equal(uOtherMeta.sizes, uOutMeta.sizes));
  const uint other_idx = other_same_shape
      ? write_idx
      : coord_to_idx(other_coord, uOtherMeta.strides);
  const float other_val = uOther.data[other_idx];
  // clang-format off
  $if not INPLACE:
    uvec4 in_coord = write_coord;
    in_coord.x = (uInMeta.sizes.x == 1u) ? 0u : in_coord.x;
    in_coord.y = (uInMeta.sizes.y == 1u) ? 0u : in_coord.y;
    in_coord.z = (uInMeta.sizes.z == 1u) ? 0u : in_coord.z;
    in_coord.w = (uInMeta.sizes.w == 1u) ? 0u : in_coord.w;
    const bool in_same_shape = all(equal(uInMeta.sizes, uOutMeta.sizes));
    const uint in_idx = in_same_shape
        ? write_idx
        : coord_to_idx(in_coord, uInMeta.strides);
    const float in_val = uInput.data[in_idx];
  $else:
    const float in_val = uOutput.data[write_idx];
  // clang-format on
  uOutput.data[write_idx] = OP(in_val, other_val, uParams.alpha.x);
}
