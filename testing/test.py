import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# Initialize PyGame
pygame.init()

# Set the window size
width, height = 800, 600
pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)

# Set OpenGL settings
glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
glPointSize(5)  # Set point size for debugging purposes

# Example points: 10 points with (x, y) coordinates
points = np.random.rand(10, 2).astype(np.float32)  # 10 random points in the range [0, 1]

# Create SSBO buffer
ssbo = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
glBufferData(GL_SHADER_STORAGE_BUFFER, points.nbytes, points, GL_STATIC_DRAW)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

# Define vertex and fragment shaders
vertex_shader_source = """
#version 430 core
layout(std140, binding = 0) buffer Points {
    vec2 points[];
};

void main() {
    uint idx = gl_VertexID / 2;  // Each line segment has two vertices
    uint pair_idx = gl_VertexID % 2;  // Either 0 or 1 for the pair of points

    gl_Position = vec4(points[idx * 2 + pair_idx], 0.0, 1.0);  // Access point pairs
    gl_PointSize = 5.0;  // Optional point size, not needed for lines
}
"""

fragment_shader_source = """
#version 430 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red color for lines
}
"""

# Compile shader program
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        print(glGetShaderInfoLog(shader))
        raise RuntimeError("Shader compilation failed")
    return shader

# Compile shaders
vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

# Create and link program
shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)

if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
    print(glGetProgramInfoLog(shader_program))
    raise RuntimeError("Program linking failed")

glUseProgram(shader_program)

# Main render loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)

    # Bind the SSBO buffer
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)

    # Draw line segments (each pair of points creates a line segment)
    glDrawArrays(GL_LINES, 0, len(points))  # Use the points array as line segments

    # Unbind the SSBO buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Swap buffers to display the result
    pygame.display.flip()

# Clean up and quit
pygame.quit()
