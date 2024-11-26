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

layout(local_size_x = 1, local_size_y = 1) in;

vec2 GetForce(uint i, uint j){
    float k = 8.987551 * pow(10.0f, 9.0f); // Coulomb constant
    vec2 r1 = r[j] - charges[i].position; // Position vector difference
    float r1MagCubed = pow(length(r1), 3.0f); // Magnitude cubed
    float forceConst = (k * charges[i].magnitude) / r1MagCubed; // Force constant

    return forceConst * r1; // Calculate force
}

void main() {
    uint i = gl_GlobalInvocationID.x; // Index for charges
    uint j = gl_GlobalInvocationID.y; // Index for positions

    // Calculate force and store it in the appropriate location
    force[i * gl_NumWorkGroups.y + j] = GetForce(i, j); // Store in a 1D array for force
}
