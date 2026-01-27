#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Params Buffer
 */
layout(set = 0, binding = 1) uniform PRECISION restrict Block {
  uvec3 extents;
  int fill0;
  vec4 value;
}
uArgs;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Returns a tensor filled with a scalar value
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, ivec3(uArgs.extents)))) {
    return;
  }

  imageStore(uOutput, pos, uArgs.value);
}
