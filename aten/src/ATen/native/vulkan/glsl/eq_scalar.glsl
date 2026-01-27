#version 450 core
#define PRECISION ${PRECISION}

layout(std430) buffer;

layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  float other;
}
uArgs;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uArgs.extents.xyz))) {
    return;
  }

  const vec4 in_texel = texelFetch(uInput, pos, 0);
  const bvec4 cmp = equal(in_texel, vec4(uArgs.other));
  imageStore(uOutput, pos, ivec4(cmp));
}