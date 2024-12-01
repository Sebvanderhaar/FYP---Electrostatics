import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
from openGLDrawing import *
from OpenGL.GL.shaders import compileShader, compileProgram
import time

vertex_shader = """
#version 430
layout(location = 0) in vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_shader = """
#version 430
out vec4 color;
void main()
{
    color = vec4(0.8, 0.2, 0.5, 1.0);  // Color for triangles and square (purple-pink)
}
"""

physics_shader = """
#version 430

struct Charge {
    vec2 position;
    float magnitude;
};

layout(std430, binding = 0) buffer ChargesBuffer {
    Charge charges[];
};

layout(std430, binding = 1) buffer PositionsBuffer {
    vec2 r[];
};

layout(std430, binding = 2) buffer ForceBuffer {
    vec2 force[];
};

layout(local_size_x = 1, local_size_y = 1) in;

uniform uint numPositions;

vec2 GetForce(uint cIndex, uint pIndex){
    float k = 8.987551 * pow(10.0f, 9.0f);
    vec2 r1 = r[pIndex] - charges[cIndex].position;
    float r1MagCubed = pow(length(r1), 3.0f);
    float forceConst = (k * charges[cIndex].magnitude) / r1MagCubed;

    return forceConst*r1;
}

void main() {
    uint cIndex = gl_GlobalInvocationID.x;
    uint pIndex = gl_GlobalInvocationID.y;

    uint index = pIndex + cIndex * numPositions;
    force[index] = GetForce(cIndex, pIndex);
}
"""

summing_shader = """
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
"""

line_shader = """
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
"""

vertex_shader_lines = """
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
"""

geometry_shader_lines = """
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

fragment_shader_lines = """
#version 430 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 1.0, 1.0, 1.0); // White color
}
"""

class Charge:
    def __init__(self, position: np.ndarray, magnitude: float, radius: float, colour: tuple):
        self.position: np.ndarray = position
        self.magnitude: float = magnitude
        self.radius: float = radius
        self.colour: tuple = colour
        self.dragging: bool = False

    def Draw(self, shader) -> None:
        drawCircle(shader, self.position, self.radius)

    def HandleDragging(self, event: pygame.event) -> None:
        mousePos = np.array(pygame.mouse.get_pos())
        xMousePosOpenGL = ((mousePos[0])/400) - 1
        yMousePosOpenGL = 1 - ((mousePos[1])/400)

        distToMouse = np.linalg.norm(np.array([self.position[0] - xMousePosOpenGL, self.position[1] - yMousePosOpenGL]))

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and distToMouse < self.radius:
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if self.dragging == True:
            self.position = np.array([xMousePosOpenGL, yMousePosOpenGL])

class Line:
    def __init__(self, x: np.float32, y: np.float32):
        self.points: np.ndarray = np.array([(x, y)], dtype="2f4")
        self.colour: np.ndarray = (200, 200, 200)
        self.width: float = 0.01

    def Draw(self, shader) -> None:
        for startPos, endPos in zip(self.points, self.points[1:]):
            drawLine(shader, startPos, endPos, self.width)

    def Append(self, x: np.float32, y: np.float32):
        self.points = np.concatenate((self.points, np.array([(x, y)], dtype="2f4")))

def CreateCharges() -> list:
    constRadius = 0.03
    constColour = (200,200,200)

    charge1 = Charge(np.array([0.25, 0.25]), 1, 0.05, constColour)
    charge2 = Charge(np.array([0.75, 0.75]), 1, constRadius, constColour)
    charge3 = Charge(np.array([0.25, 0.75]), -2, constRadius, constColour)

    chargeList = [charge1, charge2, charge3]

    return chargeList

def initLines(chargeList, linesPerCharge) -> list:
    lineList = []
    for charge in chargeList:
        if charge.magnitude > 0:
            totLines = linesPerCharge*charge.magnitude
            for i in range(totLines):
                initX = charge.position[0] + np.sin(2*np.pi*(i/totLines))*0.05
                initY = charge.position[1] + np.cos(2*np.pi*(i/totLines))*0.05

                line = Line(initX, initY)
                lineList.append(line)

    return lineList

def lineExtendOpenGL(lineList, chargeList, physicsProgram, summingProgram, lineProgram, i, iterations):
    numColsLoc = glGetUniformLocation(physicsProgram, "numPositions")
    glUseProgram(physicsProgram)
    glUniform1ui(numColsLoc, len(lineList))

    glDispatchCompute(len(chargeList), len(lineList), 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)


    numChargesLoc = glGetUniformLocation(summingProgram, "numCharges")
    numPosLoc = glGetUniformLocation(summingProgram, "numPos")
    glUseProgram(summingProgram)
    glUniform1ui(numChargesLoc, len(chargeList))
    glUniform1ui(numPosLoc, len(lineList))

    glDispatchCompute(len(lineList), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)


    iterationLoc = glGetUniformLocation(lineProgram, "i")
    totItersLoc = glGetUniformLocation(lineProgram, "totIters")
    glUseProgram(lineProgram)
    glUniform1ui(iterationLoc, i)
    glUniform1ui(totItersLoc, iterations)

    glDispatchCompute(len(lineList), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

def bindBuffers(chargeList, lineList, iterations) -> tuple:
    chargesBuffer, positionsBuffer, forcesBuffer, summingBuffer, lineBuffer = glGenBuffers(5)
    
    chargesDtype = [("position", "2f4"), ("magnitude", "f4"), ("buffer", "f4")]
    chargesArray = np.zeros(len(chargeList), dtype=chargesDtype)
    positionsArray = np.zeros(len(lineList), dtype="2f4")
    forcesArray = np.zeros(len(chargeList) * len(lineList), dtype="2f4")
    summingArray = np.zeros(len(lineList), dtype="2f4")
    linesArray = np.zeros(len(lineList) * (iterations+1), dtype="2f4")

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chargesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, chargesArray.nbytes, chargesArray, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, chargesBuffer)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionsBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, positionsArray.nbytes, positionsArray, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, positionsBuffer)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forcesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, forcesArray.nbytes, forcesArray, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, forcesBuffer)
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, summingBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, summingArray.nbytes, summingArray, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, summingBuffer)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lineBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, linesArray.nbytes, linesArray, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, lineBuffer)

    return chargesBuffer, positionsBuffer, forcesBuffer, summingBuffer, lineBuffer

def writePosChargesLines(chargesBuffer, positionsBuffer, lineBuffer, chargeList, lineList, iterations):
    chargesDtype = [("position", "2f4"), ("magnitude", "f4"), ("buffer", "f4")]
    charges = np.zeros(len(chargeList), dtype=chargesDtype)
    positions = np.zeros(len(lineList), dtype="2f4")
    lines = np.zeros(len(lineList)*(iterations+1), dtype="2f4")

    for index, charge in enumerate(chargeList):
        charges[index] = ([charge.position[0], charge.position[1]], charge.magnitude, 0)

    for index, line in enumerate(lineList):
        positions[index] = line.points[-1]

    for index, line in enumerate(lineList):
        lines[index*(iterations+1)] = line.points[-1]

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chargesBuffer)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, charges.nbytes, charges)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionsBuffer)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, positions.nbytes, positions)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lineBuffer)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, lines.nbytes, lines)

def main():
    global windowWidth
    global windowHeight
    windowWidth, windowHeight = 800, 800
    pygame.init()
    pygame.display.set_mode((windowWidth, windowHeight), DOUBLEBUF | OPENGL)
    glViewport(0, 0, windowWidth, windowHeight)

    iterations = 200

    chargeList = CreateCharges()
    lineList = initLines(chargeList, 8)

    shader = compile_shader_program(vertex_shader, fragment_shader)

    shader_program_lines = compileProgram(
        compileShader(vertex_shader_lines, GL_VERTEX_SHADER),
        compileShader(geometry_shader_lines, GL_GEOMETRY_SHADER),
        compileShader(fragment_shader_lines, GL_FRAGMENT_SHADER)
        )

    physicsShader = compileShader(physics_shader, GL_COMPUTE_SHADER)
    physicsProgram = compileProgram(physicsShader)

    summingShader = compileShader(summing_shader, GL_COMPUTE_SHADER)
    summingProgram = compileProgram(summingShader)

    lineShader = compileShader(line_shader, GL_COMPUTE_SHADER)
    lineProgram = compileProgram(lineShader)

    chargesBuffer, positionsBuffer, forcesBuffer, summingBuffer, lineBuffer = bindBuffers(chargeList, lineList, iterations)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            for charge in chargeList:
                charge.HandleDragging(event)

        glClear(GL_COLOR_BUFFER_BIT)

        lineList = initLines(chargeList, 8)
        
        writePosChargesLines(chargesBuffer, positionsBuffer, lineBuffer, chargeList, lineList, iterations)

        for i in range(iterations):
            lineExtendOpenGL(lineList, chargeList, physicsProgram, summingProgram, lineProgram, i, iterations)

        skip_interval_loc = glGetUniformLocation(shader_program_lines, "skip_interval")
        glUseProgram(shader_program_lines)
        glUniform1ui(skip_interval_loc, iterations + 1)
        glDrawArrays(GL_LINE_STRIP, 0, len(lineList)*iterations)

        for i, charge in enumerate(chargeList):
            charge.Draw(shader)

        #Flip display
        pygame.display.flip()
        pygame.time.wait(10)

    glDeleteProgram(shader)
    pygame.quit()

if __name__ == "__main__":
    main()
