#version 430 core
layout(std430, binding = 4) buffer LineBuffer{
    vec2 line[];
};

uniform uint skip_interval;

out vec2 startPoint;
out vec2 endPoint;
void main() {
    int line_index = gl_VertexID;

    if ((line_index + 1) % skip_interval != 0) {
        startPoint = line[gl_VertexID];
        endPoint = line[gl_VertexID + 1];
    }
}