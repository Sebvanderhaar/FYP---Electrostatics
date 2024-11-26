#version 430

struct Charge {
    vec2 position;
    float magnitude;
};

layout(std430, binding = 0) buffer ChargesBuffer {
    Charge charges[];
};

layout(std430, binding = 1) buffer ForceBuffer {
    vec2 force[];
};

layout(std430, binding = 2) buffer PositionsBuffer {
    vec2 r[];
};


layout(local_size_x = 1) in;

vec2 GetForce(uint i, uint j){
    float k = 8.987551 * pow(10.0f, 9.0f);
    vec2 r1 = r[j*2] - charges[i].position;
    float r1MagCubed = pow(length(r1), 3.0f);
    float forceConst = (k * charges[i].magnitude) / r1MagCubed;

    return forceConst*r1
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;

    force[i*j] = GetForce(i, j)
}