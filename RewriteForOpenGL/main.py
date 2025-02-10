import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
from openGLDrawing import *
from OpenGL.GL.shaders import compileShader, compileProgram
import time
import os
import csv

import tkinter as tk
from threading import Thread
from queue import Queue

#Container for all information and methods for a single charge
class Charge:
    def __init__(self, position: np.ndarray, magnitude: float, radius: float, colour: tuple, draw: bool):
        self.position: np.ndarray = position
        self.magnitude: float = magnitude
        self.radius: float = radius
        self.colour: tuple = colour
        self.dragging: bool = False
        self.draw: bool = draw

    def Draw(self, shader) -> None:
        #from openGLDrawing.py
        if self.draw:
            drawCircle(shader, self.position, self.radius)

    #Called every frame, uses pygame events tracker
    def HandleDragging(self, event: pygame.event, screen: pygame.display) -> None:
        screenX, screenY = screen.get_size()
        mousePos: tuple = np.array(pygame.mouse.get_pos())
        xMousePosOpenGL: int = ((mousePos[0])/(screenX/2)) - 1
        yMousePosOpenGL: int = 1 - ((mousePos[1])/(screenY/2))

        distToMouse: float = np.linalg.norm(np.array([self.position[0] - xMousePosOpenGL, self.position[1] - yMousePosOpenGL]))

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

    charge1 = Charge(np.array([0.25, 0.25]), 1, 0.05, constColour, True)
    charge2 = Charge(np.array([0.75, 0.75]), 1, constRadius, constColour, True)
    charge3 = Charge(np.array([0.25, 0.75]), 1, constRadius, constColour, True)
    charge4 = Charge(np.array([-0.75, -0.75]), 1, constRadius, constColour, True)
    charge5 = Charge(np.array([-0.25, -0.75]), 1, constRadius, constColour, True)

    chargeList = [charge1, charge2, charge3, charge4, charge5]

    #chargeList = []

    return chargeList

def initLines(chargeList, linesPerCharge) -> list:
    lineList = []
    for charge in chargeList:
        if charge.magnitude > 0:
            totLines = round(np.ceil(linesPerCharge*charge.magnitude)) + 1
            for i in range(totLines):
                initX = charge.position[0] + np.sin(2*np.pi*(i/totLines))*0.001
                initY = charge.position[1] + np.cos(2*np.pi*(i/totLines))*0.001

                line = Line(initX, initY)
                lineList.append(line)

    return lineList

def initLinesCustom(numberOfLines) -> list:
    lineList = []
    xStart = -0.5
    xEnd = 0.5

    for i in range(numberOfLines):
        lineX = xStart + i/numberOfLines
        lineY = 0.75

        lineList.append(Line(lineX, lineY))

    return lineList

def lineExtendOpenGL(lineList, chargeList, physicsProgram, summingProgram, lineProgram, i, iterations, lineLength):
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
    lineLengthLoc = glGetUniformLocation(lineProgram, "lineLength")
    glUseProgram(lineProgram)
    glUniform1ui(iterationLoc, i)
    glUniform1ui(totItersLoc, iterations)
    glUniform1f(lineLengthLoc, lineLength)

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

def compileAllShader() -> tuple:
    shaderFileNames = ["fragmentShader.glsl", "fragmentShaderLines.glsl", "geometryShader.glsl", "lineShader.glsl", "physicsShader.glsl", "summingShader.glsl", "vertexShader.glsl", "vertexShaderLines.glsl"]
    shaderFileDict = {}

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for filename in shaderFileNames:
        filepath = os.path.join(base_dir, "shaders", filename)
        with open(filepath, "r") as file:
            shaderFileDict[filename] = file.read()
    
    mainShaderProgram = compileProgram(
        compileShader(shaderFileDict["vertexShader.glsl"], GL_VERTEX_SHADER),
        compileShader(shaderFileDict["fragmentShader.glsl"], GL_FRAGMENT_SHADER)
        )

    linesShaderProgram = compileProgram(
        compileShader(shaderFileDict["vertexShaderLines.glsl"], GL_VERTEX_SHADER),
        compileShader(shaderFileDict["geometryShader.glsl"], GL_GEOMETRY_SHADER),
        compileShader(shaderFileDict["fragmentShaderLines.glsl"], GL_FRAGMENT_SHADER)
        )

    physicsProgram = compileProgram(compileShader(shaderFileDict["physicsShader.glsl"], GL_COMPUTE_SHADER))

    summingProgram = compileProgram(compileShader(shaderFileDict["summingShader.glsl"], GL_COMPUTE_SHADER))

    lineProgram = compileProgram(compileShader(shaderFileDict["lineShader.glsl"], GL_COMPUTE_SHADER))

    return mainShaderProgram, linesShaderProgram, physicsProgram, summingProgram, lineProgram 

def createChargeLine(startPos, endPos, mag):
    newCharges = []

    nCharges = 10
    chargeDensity = 0.01 * mag
    radius = 0.01
    colour = (255, 255, 255)

    for i in range(nCharges):
        X = startPos[0] + (endPos[0] - startPos[0])*(i/nCharges)
        Y = startPos[1] + (endPos[1] - startPos[1])*(i/nCharges)
        if i % 10 == 0:
            charge = Charge(np.array([X, Y]), chargeDensity/(nCharges/np.linalg.norm(endPos - startPos)), radius, colour, True)
        else:
            charge = Charge(np.array([X, Y]), chargeDensity/(nCharges/np.linalg.norm(endPos - startPos)), radius, colour, False)
        newCharges.append(charge)

    return newCharges

def createParallelPlate(chargeList) -> list:
    startPos: np.ndarray = np.array([-0.5, 0.8])
    endPos: np.ndarray = np.array([0.5, 0])

    chargeLine: list = createChargeLine(startPos, endPos, 1)

    for item in chargeLine:
        chargeList.append(item)
    
    startPos: np.ndarray = np.array([-0.5, -0.8])
    endPos: np.ndarray = np.array([0.5, -0.8])

    chargeLine: list = createChargeLine(startPos, endPos, -1)

    for item in chargeLine:
        chargeList.append(item)

    return chargeList

def drawLines(shader, lineList, iterations):
    skip_interval_loc = glGetUniformLocation(shader, "skip_interval")
    glUseProgram(shader)
    glUniform1ui(skip_interval_loc, iterations + 1)
    glDrawArrays(GL_LINE_STRIP, 0, len(lineList)*iterations)

def createDipole(chargeList, centre) -> list:
    magnitude = 0.001
    split = 0.01
    radius = 0.001
    colour = (255, 255, 255)
    
    X = centre[0]
    Ylower = centre[1] - split/2
    Yupper = centre[1] + split/2

    chargeUpper = Charge(np.array([X,Yupper]), -magnitude, radius, colour, False)
    chargeLower = Charge(np.array([X,Ylower]), magnitude, radius, colour, False)

    chargeList.append(chargeUpper)
    chargeList.append(chargeLower)

    return chargeList

def createDipoleGrid(chargeList, xCount, yCount, xStart, yStart, xEnd, yEnd) -> list:
    for i in range(xCount):
        for j in range(yCount):
            X = xStart + ((xEnd - xStart)/xCount)*i
            Y = yStart + ((yEnd - yStart)/yCount)*j
            
            centre = [X, Y]

            chargeList = createDipole(chargeList, centre)

    return chargeList

def logPerformance(setupDict, timingDataDict):
    combinedDict = {**setupDict, **timingDataDict}
    fieldNames = list(combinedDict.keys())
    fileName = "Setup1Run1"
    filePath = f"RewriteForOpenGL\PerformanceLoggingData\\{fileName}.csv"
    isFile = os.path.isfile(filePath)

    with open(filePath, mode="a" if isFile else "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldNames)
        if not isFile:
            writer.writeheader()

        writer.writerow(combinedDict)

def runTkinter(queue):
    def sliderChanged(value, type):
        if type == 1:
            queue.put(f"Lines:{value}")
        elif type == 2:
            queue.put(f"Line Length:{value}")
        elif type == 3:
            queue.put(f"Segment Count:{value}")

    def sendText():
        text = entry.get()
        if text.strip():
            queue.put(f"xText:{text}")
            entry.delete(0, tk.END)

    def check_queue():
        for _ in range(queue.size):
            message = queue.get()
            if message.startswith("fps:"):
                fps_value = message.split(":")[1]
                label.config(text=f"FPS: {fps_value}")

        root.after(1, check_queue)

    root = tk.Tk()
    root.title("Tkinter UI")
    root.geometry("800x400")
    
    # Example of adding widgets
    label = tk.Label(root, text="FPS: --", font=("Arial", 14))
    label.pack(pady=10)
    
    button = tk.Button(root, text="New charge", command=sendText)
    button.pack(pady=10)
    
    tk.Label(root, text="Line count").pack(pady=10)
    lineCountSlider = tk.Scale(root, from_= 1, to=500, orient="horizontal", command=lambda v: sliderChanged(v, 1))
    lineCountSlider.set(10)
    lineCountSlider.pack(pady=10)

    tk.Label(root, text="Line Length").pack(pady=5)
    lineLengthVar = tk.DoubleVar(value=0.01)  # Variable to store float value
    lineLengthSlider = tk.Scale(root, from_=0.00001, to=0.01, resolution=0.00001, orient="horizontal",
                                variable=lineLengthVar, command=lambda v: sliderChanged(float(v), 2), length=600)
    lineLengthSlider.set(0.0001)
    lineLengthSlider.pack(pady=10)

    tk.Label(root, text="Segment count").pack(pady=10)
    lineCountSlider = tk.Scale(root, from_= 1, to=10000, orient="horizontal", command=lambda v: sliderChanged(v, 3))
    lineCountSlider.set(10)
    lineCountSlider.pack(pady=10)

    entry = tk.Entry(root)
    entry.pack(pady=5)
    
    root.after(1, check_queue)

    root.mainloop()

def main():
    global windowWidth
    global windowHeight
    windowWidth, windowHeight = 1200, 1200

    pygame.init()

    sceen: pygame.display = pygame.display.set_mode((windowWidth, windowHeight), DOUBLEBUF | OPENGL)
    clock: pygame.time.Clock = pygame.time.Clock()
    pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)

    iterations: int = 1000
    linesPerUnitCharge: int = 1000

    chargeList: list = CreateCharges()
    #chargeList: list = createDipoleGrid(chargeList, 200, 20, -1, -0.24, 1, 0.24)
    #chargeList: list = createParallelPlate(chargeList)
    lineList: list = initLines(chargeList, linesPerUnitCharge)
    #lineList: list = initLinesCustom(linesPerUnitCharge)

    shader, shader_program_lines, physicsProgram, summingProgram, lineProgram = compileAllShader()
    
    chargesBuffer, positionsBuffer, forcesBuffer, summingBuffer, lineBuffer = bindBuffers(chargeList, lineList, 10000)

    running: bool = True
    frameIndex: int = -1
    lineLength: float = 0.01

    queue = Queue()

    tk_thread = Thread(target=runTkinter, args=(queue,), daemon=True)
    tk_thread.start()

    while running:
        frameIndex += 1
        #if frameIndex % 100 == 0:
            #iterations += 10
        #if frameIndex == 10000:
            #running = False

        frameStart = time.perf_counter()
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            for charge in chargeList:
                charge.HandleDragging(event, sceen)
        draggingHandleEnd = time.perf_counter()

        while not queue.empty():
            message = queue.get()
            if message.startswith("Lines:"):
                linesPerUnitCharge = int(message.split(":")[1]) - 1
            elif message.startswith("xText:"):
                xPos = float(message.split(":")[1])
            elif message.startswith("Line Length:"):
                lineLength = float(message.split(":")[1])
            elif message.startswith("Segment Count:"):
                iterations = int(message.split(":")[1])

        print(linesPerUnitCharge)

        glClear(GL_COLOR_BUFFER_BIT)

        #Gets the initial position of each line, using chargeList
        lineList = initLines(chargeList, linesPerUnitCharge)
        lineListSetStart = time.perf_counter()
        #lineList: list = initLinesCustom(linesPerUnitCharge)
        lineListSetEnd = time.perf_counter()

        #Writes the required data to the Charges, positions and line buffers
        #Positions is the init positions of each line
        #Charges is a struct containing each charges pos and magnitude
        #Lines is the init positions of each line, spread out such to fit each line's full data
        buffersStart = time.perf_counter()
        writePosChargesLines(chargesBuffer, positionsBuffer, lineBuffer, chargeList, lineList, iterations)
        buffersEnd = time.perf_counter()

        #Extends each line with number of iterations given
        
        physicsStart = time.perf_counter()
        for i in range(iterations):
            lineExtendOpenGL(lineList, chargeList, physicsProgram, summingProgram, lineProgram, i, iterations, lineLength)
        physicsEnd = time.perf_counter()

        #Draw the lines, uses openGLs draw array method, with data from buffers already on GPU
        drawLinesStart = time.perf_counter()
        drawLines(shader_program_lines, lineList, iterations)
        drawLinesEnd = time.perf_counter()

        #Draws each charge, pulls data from python chargeList
        drawChargesStart = time.perf_counter()
        for i, charge in enumerate(chargeList):
            charge.Draw(shader)
        drawChargesEnd = time.perf_counter()

        flippingStart = time.perf_counter()
        pygame.display.flip()
        clock.tick()
        flippingEnd = time.perf_counter()

        clearingPrintStart = time.perf_counter()
        fps = clock.get_fps()
        #os.system('cls' if os.name == 'nt' else 'clear')
        clearingPrintEnd = time.perf_counter()

        """
        printBlockStart = time.perf_counter()
        print("FPS: ", fps)
        draggingPercent = (draggingHandleEnd - frameStart)*100*fps
        print("Dragging managing: ", draggingPercent)
        lineListSetPercent = (lineListSetEnd - lineListSetStart)*100*fps
        print("Line List Set: ", lineListSetPercent)
        buffersPercent = (buffersEnd - buffersStart)*100*fps
        print("Buffers: ", buffersPercent)
        physicsPercent = (physicsEnd - physicsStart)*100*fps
        print("Physics: ", physicsPercent)
        linesDrawingPercent = (drawLinesEnd - drawLinesStart)*100*fps
        print("Lines drawing: ", linesDrawingPercent)
        chargesDrawingPercent = (drawChargesEnd - drawChargesStart)*100*fps
        print("Charges drawing: ", chargesDrawingPercent)
        clearingPrintPercent = (clearingPrintEnd - clearingPrintStart)*100*fps
        print("Clearing Print: ", clearingPrintPercent)
        flippingPercent = (flippingEnd - flippingStart)*100*fps
        print("Flipping: ", flippingPercent)
        print("Iterations: ", iterations)

        setupDict = {}
        setupDict["Iterations"] = iterations
        setupDict["ChargeCount"] = len(chargeList)
        setupDict["LineCount"] = len(lineList)
        setupDict["DrawChargeCount"] = len([item for item in chargeList if item.draw])

        timingDataDict = {}
        timingDataDict["frameIndex"] = frameIndex
        timingDataDict["FPS"] = fps
        timingDataDict["dragging"] = draggingPercent
        timingDataDict["lineListSet"] = lineListSetPercent
        timingDataDict["buffers"] = buffersPercent
        timingDataDict["physics"] = physicsPercent
        timingDataDict["lineDrawing"] = linesDrawingPercent
        timingDataDict["chargeDrawing"] = chargesDrawingPercent
        timingDataDict["clearingPrint"] = clearingPrintPercent
        timingDataDict["flipping"] = flippingPercent

        #logPerformance(setupDict, timingDataDict)
 
        printBlockEnd = time.perf_counter()

        printBlockPercent = (printBlockEnd - printBlockStart)*100*fps
        print("Print Block: ", printBlockPercent)
        print("Frame total: ", (time.perf_counter() - frameStart)*100*fps)
        """
        queue.put(f"fps:{str(fps)}")
        print(fps)


    glDeleteProgram(shader)
    pygame.quit()

if __name__ == "__main__":
    main()
