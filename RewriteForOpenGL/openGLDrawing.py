from OpenGL.GL import *
import numpy as np

def compile_shader_program(vertex_src, fragment_src):
    vertex = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex, vertex_src)
    glCompileShader(vertex)
    if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(vertex))

    fragment = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment, fragment_src)
    glCompileShader(fragment)
    if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(fragment))

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex)
    glAttachShader(shader_program, fragment)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(shader_program))

    glDeleteShader(vertex)
    glDeleteShader(fragment)
    return shader_program

def drawShape(shader, vertices, primitive_type):
    # Create and bind a temporary VAO and VBO for each shape
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

    # Use the shader program and draw the shape
    glUseProgram(shader)
    glDrawArrays(primitive_type, 0, len(vertices))

    # Clean up by deleting the VBO and VAO after drawing
    glDeleteBuffers(1, [vbo])
    glDeleteVertexArrays(1, [vao])

def drawPolygon(shader, sides, centre, radius, rotOffset):
    vertices = np.zeros((sides, 2), dtype=np.float32)
    for i in range(sides):
        offsetX = radius * np.sin(((i*2*np.pi)/sides) + rotOffset)
        offsetY = radius * np.cos(((i*2*np.pi)/sides) + rotOffset)

        vertices[i] = np.array([centre[0] + offsetX, centre[1] + offsetY])

    drawShape(shader, vertices, GL_TRIANGLE_FAN)

def drawCircle(shader, centre, radius):
    drawPolygon(shader, 30, centre, radius, 0)

def drawLine(shader, start, end, width):
    vertices = np.array([start, end], dtype=np.float32)
    glLineWidth(width)
    drawShape(shader, vertices, GL_LINES)