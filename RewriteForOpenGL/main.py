import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
from openGLDrawing import *
import scipy.constants as cnst
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
    charge2 = Charge(np.array([0.75, 0.75]), -1, constRadius, constColour)
    charge3 = Charge(np.array([0.25, 0.75]), 2, constRadius, constColour)

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

def GetForce(r, chargeList):
    k = 1/(4*np.pi*cnst.epsilon_0)
    forceArr = np.zeros((len(chargeList),2))
    for index, charge in enumerate(chargeList):
        r1 = r - charge.position
        r1MagCubed = np.linalg.norm(r1)**3
        constForce = (charge.magnitude*k)/r1MagCubed

        forceArr[index] = constForce*r1
    return np.sum(forceArr, axis=0)

def lineExtend(line, chargeList):
    running = True
    dl = 0.01

    j = 0
    while(running):
        j += 1
        force = GetForce(line.points[-1], chargeList)
        forceMag = np.linalg.norm(force)

        nextElem = line.points[-1] + (force/forceMag)*dl
        line.points = np.vstack([line.points, nextElem])

        for charge in chargeList:
            vec = charge.position - nextElem
            if j > 200 or np.linalg.norm(nextElem) > 5:
                running = False

    return line

def lineExtendOpenGL(lineList, chargeList, physicsProgram, summingProgram):
    start = time.perf_counter()
    dl = 0.1

    chargesDtype = [("position", "2f4"), ("magnitude", "f4"), ("buffer", "f4")]
    charges = np.zeros(len(chargeList), dtype=chargesDtype)
    positions = np.zeros(len(lineList), dtype="2f4")

    for index, charge in enumerate(chargeList):
        charges[index] = ([charge.position[0], charge.position[1]], charge.magnitude, 0)

    for index, line in enumerate(lineList):
        positions[index] = line.points[-1]

    forces = np.zeros((len(charges)*len(positions)), dtype="2f4")

    results = np.zeros(len(positions), dtype="2f4")

    chargesBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chargesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, charges.nbytes, charges, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, chargesBuffer)

    positionsBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionsBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, positions.nbytes, positions, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, positionsBuffer)

    forcesBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forcesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, forces.nbytes, forces, GL_STREAM_READ)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, forcesBuffer)

    resultsBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultsBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, results.nbytes, results, GL_STREAM_READ)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resultsBuffer)

    bufferTime = time.perf_counter()

    numColsLoc = glGetUniformLocation(physicsProgram, "numPositions")
    glUseProgram(physicsProgram)
    glUniform1ui(numColsLoc, len(positions))
    startMem = time.perf_counter()
    glDispatchCompute(len(charges), len(positions), 1)

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    endMem = time.perf_counter()

    numChargesLoc = glGetUniformLocation(summingProgram, "numCharges")
    numPosLoc = glGetUniformLocation(summingProgram, "numPos")
    glUseProgram(summingProgram)
    glUniform1ui(numChargesLoc, len(charges))
    glUniform1ui(numPosLoc, len(positions))
    glDispatchCompute(len(positions), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    # Retrieve the results
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultsBuffer)
    startRet = time.perf_counter()
    result = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, results.nbytes), dtype='2f4')
    endRet = time.perf_counter()
    #result = result.reshape((len(positions), 2), order="F")

    outList = []
    for index, line in enumerate(lineList):
        force = result[index]
        forceMag = np.linalg.norm(force)

        nextElem = line.points[-1] + (force/forceMag)*dl

        line.Append(nextElem[0], nextElem[1])

        outList.append(line)

    print("retrieval: ", endRet - startRet)
    print("Physics runtime: ", endMem - startMem)
    print("Buffers: ", bufferTime - start)

    return outList

def defineBuffers(chargeList, lineList):
    chargesDtype = [("position", "2f4"), ("magnitude", "f4"), ("buffer", "f4")]
    charges = np.zeros(len(chargeList), dtype=chargesDtype)

    positions = np.zeros(len(lineList), dtype="2f4")

    forces = np.zeros((len(charges)*len(positions)), dtype="2f4")

    results = np.zeros(len(positions), dtype="2f4")

    chargesBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chargesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, charges.nbytes, charges, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, chargesBuffer)

    positionsBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionsBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, positions.nbytes, positions, GL_STREAM_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, positionsBuffer)

    forcesBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forcesBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, forces.nbytes, forces, GL_STREAM_READ)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, forcesBuffer)

    resultsBuffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultsBuffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, results.nbytes, results, GL_STREAM_READ)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, resultsBuffer)

def main():
    global windowWidth
    global windowHeight
    windowWidth, windowHeight = 800, 800
    pygame.init()
    pygame.display.set_mode((windowWidth, windowHeight), DOUBLEBUF | OPENGL)
    glViewport(0, 0, windowWidth, windowHeight)

    shader = compile_shader_program(vertex_shader, fragment_shader)
    chargeList = CreateCharges()

    physicsShader = compileShader(physics_shader, GL_COMPUTE_SHADER)
    physicsProgram = compileProgram(physicsShader)

    summingShader = compileShader(summing_shader, GL_COMPUTE_SHADER)
    summingProgram = compileProgram(summingShader)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            for charge in chargeList:
                charge.HandleDragging(event)

        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        lineList = initLines(chargeList, 8)
        
        for _ in range(10):
            lineList = lineExtendOpenGL(lineList, chargeList, physicsProgram, summingProgram)

        for line in lineList:
            line.Draw(shader)

        
        #Drawing
        for i, charge in enumerate(chargeList):
            charge.Draw(shader)

        #Flip display
        pygame.display.flip()
        pygame.time.wait(10)

    glDeleteProgram(shader)
    pygame.quit()

if __name__ == "__main__":
    main()
