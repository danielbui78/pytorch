#version 450 core
#define PRECISION ${PRECISION}

#include "texel_access.h"

layout(std430) buffer;

layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uOther;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 output_sizes;
  ivec4 input_sizes;
  ivec4 other_sizes;
}
uArgs;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  ivec3 output_extents;
  output_extents.xy = uArgs.output_sizes.xy;
  output_extents.z =
      uArgs.output_sizes.w * int(ceil(uArgs.output_sizes.z / 4.0));

  if (any(greaterThanEqual(pos, output_extents.xyz))) {
    return;
  }

  ivec3 other_pos =
      map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.other_sizes);
  vec4 other_texel =
      load_texel(other_pos, uArgs.output_sizes, uArgs.other_sizes, uOther);

  ivec3 input_pos =
      map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.input_sizes);
  vec4 in_texel =
      load_texel(input_pos, uArgs.output_sizes, uArgs.input_sizes, uInput);

  const bvec4 cmp = equal(in_texel, other_texel);
  imageStore(uOutput, pos, ivec4(cmp));
}