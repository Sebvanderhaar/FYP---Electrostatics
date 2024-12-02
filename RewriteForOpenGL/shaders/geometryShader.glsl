#version 430 core
layout(lines) in;
layout(line_strip, max_vertices = 2) out;
in vec2 startPoint[];
in vec2 endPoint[];
void main() {
    gl_Position = vec4(startPoint[0], 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(endPoint[0], 0.0, 1.0);
    EmitVertex();
    EndPrimitive();
}