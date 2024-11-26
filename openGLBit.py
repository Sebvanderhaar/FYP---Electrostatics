import glfw
import glm
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

def compileShader(filename):
    shaderCode = open(filename, "r").read()
    shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader, shaderCode)
    glCompileShader(shader)
    checkOpenGLError()
    return shader

def checkOpenGLError():
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL Error: {error}")

def initOpenGL():
    OpenGL.ERROR_CHECKING = True
    glfw.init()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(1,1, "Window", None, None)
    glfw.hide_window(window)
    glfw.make_context_current(window)

    shader = compileShader("shader.glsl")
    program = glCreateProgram()
    checkOpenGLError()
    glAttachShader(program, shader)
    checkOpenGLError()
    glLinkProgram(program)
    checkOpenGLError()

    return program, shader

def BindBuffers(charges, forces, positions):
    chargesBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chargesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, charges.nbytes, charges, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, chargesBuffer)

    forcesBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forcesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, forces.nbytes, None, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, forcesBuffer)

    positionsBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionsBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, positions.nbytes, positions, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, positionsBuffer)

    checkOpenGLError()

    return chargesBuffer, forcesBuffer, positionsBuffer

def getResults(program, forces, charges, positions, forcesBuffer):
    glUseProgram(program)

    glDispatchCompute(len(charges), len(positions), 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forcesBuffer)
    forces = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, forces.nbytes)

    return forces

def cleanUp(program, shader, chargesBuffer, forcesBuffer, positionsBuffer):
    glDeleteBuffers(1, [chargesBuffer, forcesBuffer, positionsBuffer])
    glDeleteProgram(program)
    glDeleteShader(shader)

    glfw.terminate()



program, shader = initOpenGL()

charges = np.zeros(6, dtype=[('position', np.float32, 2), ('magnitude', np.float32), ('padding', np.float32)])
charges['position'] = [[0,0], [10,0], [0,10], [0,0], [10,0], [0,10]]
charges['magnitude'] = [1,1,1,-1,-1,-1]

posArray = np.array([100, 100, 50, 50, 25, 25], dtype=np.float32)

forces = np.zeros(len(charges)*len(posArray)*2, dtype=np.float32)


chargesBuffer, positionBuffer, forcesBuffer = BindBuffers(charges, forces, posArray)

forces = getResults(program, forces, charges, posArray, forcesBuffer)

results = np.frombuffer(forces, dtype=np.float32).reshape(6, 3, 2)

print("Results from compute shader (first 10 vec2s):\n", results[:6])


