#version 430

layout(std430, binding = 3) buffer ResultBuffer {
    vec2 result[];
};

layout(std430, binding = 4) buffer LineBuffer {
    vec2 line[];
};

layout(std430, binding = 1) buffer PositionBuffer {
    vec2 position[];
};

layout(local_size_x = 1) in;

uniform uint i;
uniform uint totIters;

void main() {
    uint pIndex = gl_GlobalInvocationID.x;

    vec2 nextPoint = line[i + pIndex*(totIters + 1)] + 0.01*result[pIndex]/length(result[pIndex]);

    line[i + pIndex*(totIters + 1) + 1] = nextPoint;
    position[pIndex] = nextPoint;
}