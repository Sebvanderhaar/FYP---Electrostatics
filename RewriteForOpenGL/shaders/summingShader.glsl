#version 430

layout(std430, binding = 2) buffer ForceBuffer {
    vec2 force[];
};

layout(std430, binding = 3) buffer ResultBuffer {
    vec2 result[];
};

layout(local_size_x = 1) in;

uniform uint numPos;
uniform uint numCharges;

void main() {
    uint pIndex = gl_GlobalInvocationID.x;
    vec2 sum = vec2(0.0f, 0.0f);
    for (uint sumIter = 0; sumIter < numCharges; sumIter++){
        sum += force[pIndex + sumIter * numPos];
    }
    result[pIndex] = sum;
}