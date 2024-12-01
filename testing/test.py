import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# Initialize PyGame and OpenGL
pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
glViewport(0, 0, 800, 600)
glClearColor(0.1, 0.1, 0.1, 1.0)

# Example point data (vec2s)
points = np.array([
    [0.0, 0.0],
    [0.25, 0.25],
    [-0.5, 0.5],
    [-0.25, -0.25],
    [0.5, -0.5],
    [1, -1]
], dtype=np.float32)

# Create SSBO
ssbo = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
glBufferData(GL_SHADER_STORAGE_BUFFER, points.nbytes, points, GL_STATIC_DRAW)
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

# Vertex Shader
vertex_shader_code = """
#version 430 core
layout(std430, binding = 0) buffer PointBuffer {
    vec2 points[];
};
out vec2 startPoint;
out vec2 endPoint;
void main() {
    startPoint = points[gl_VertexID];
    endPoint = points[gl_VertexID + 1];
}
"""

# Geometry Shader
geometry_shader_code = """
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
"""

# Fragment Shader
fragment_shader_code = """
#version 430 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color
}
"""

# Compile and link the shader program
shader_program = compileProgram(
    compileShader(vertex_shader_code, GL_VERTEX_SHADER),
    compileShader(geometry_shader_code, GL_GEOMETRY_SHADER),
    compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT)

    # Use the shader program
    glUseProgram(shader_program)

    # Draw lines
    glDrawArrays(GL_LINE_STRIP, 0, len(points))

    # Swap buffers
    pygame.display.flip()

# Cleanup
glDeleteBuffers(1, [ssbo])
pygame.quit()
