#version 430

struct Charge {
    vec2 position;
    float magnitude;
};

layout(std430, binding = 0) buffer ChargesBuffer {
    Charge charges[];
};

layout(std430, binding = 1) buffer PositionsBuffer {
    vec2 r[];
};

layout(std430, binding = 2) buffer ForceBuffer {
    vec2 force[];
};

layout(local_size_x = 1, local_size_y = 1) in;

uniform uint numPositions;

vec2 GetForce(uint cIndex, uint pIndex){
    float k = 8.987551 * pow(10.0f, 9.0f);
    vec2 r1 = r[pIndex] - charges[cIndex].position;
    float r1MagCubed = pow(length(r1), 3.0f);
    float forceConst = (k * charges[cIndex].magnitude) / r1MagCubed;

    return forceConst*r1;
}

void main() {
    uint cIndex = gl_GlobalInvocationID.x;
    uint pIndex = gl_GlobalInvocationID.y;

    uint index = pIndex + cIndex * numPositions;
    force[index] = GetForce(cIndex, pIndex);
}