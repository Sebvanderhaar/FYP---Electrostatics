import numpy as np
import scipy.constants as cnst
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import datetime
import pygame
import sys
import math
import time
import ctypes
from openGLBit import *
import glfw
import glm
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

class Charge:
    def __init__(self, position: np.ndarray, magnitude: float, radius: float, colour: tuple):
        self.position: np.ndarray = position
        self.magnitude: float = magnitude
        self.radius: float = radius
        self.colour: tuple = colour
        self.dragging: bool = False

    def Draw(self, surface: pygame.surface) -> None:
        pygame.draw.circle(surface, self.colour, self.position, self.radius)

    def HandleDragging(self, event: pygame.event) -> None:
        mousePos = np.array(pygame.mouse.get_pos())
        distToMouse = np.linalg.norm(mousePos - self.position)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and distToMouse < self.radius:
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

        if self.dragging == True:
            self.position = mousePos 

class ChargeC(ctypes.Structure):
    _fields_ = [("position", ctypes.c_double * 2), ("magnitude", ctypes.c_double)]

class Line:
    def __init__(self, points: np.ndarray):
        self.points: np.ndarray = points
        self.colour: np.ndarray = (200, 200, 200)
        self.width: float = 1

    def Draw(self, surface: pygame.surface) -> None:
        for startPos, endPos in zip(self.points, self.points[1:]):
            pygame.draw.line(surface, self.colour, startPos, endPos, self.width)

def GetForce(r: np.ndarray) -> np.ndarray:
    k = 1/(4*np.pi*cnst.epsilon_0)

    force: np.ndarray = np.zeros(2)

    for charge in chargeList:
        r1: np.ndarray = r - charge.position
        r1MagCubed: float = np.linalg.norm(r1)**3 
        force += k*charge.magnitude*(r1/r1MagCubed)
    
    return force

def CreateCharges() -> list:
    constRadius = 5
    constColour = (200,200,200)

    charge1 = Charge(np.array([250,250]), 1, constRadius, constColour)
    charge2 = Charge(np.array([750, 750]), -1, constRadius, constColour)
    charge3 = Charge(np.array([250, 750]), 2, constRadius, constColour)

    chargeList = [charge1, charge2, charge3]

    return chargeList

def dfdx(t: float, f: np.ndarray) -> np.ndarray:
    fout: np.ndarray = np.array([0,0])

    for charge in chargeList:
        derivative = f - charge.position
        distCubed = np.linalg.norm(derivative)
        fout += k * charge.magnitude*derivative/distCubed

    return f

def CreateFieldLinesSolveIVP() -> list: #3 - 6ms per line
    global iters
    def dfdx(t: float, f: np.ndarray) -> np.ndarray:
        fout: np.ndarray = np.array([0,0], dtype=np.float64)

        for charge in chargeList:
            derivative = f - charge.position
            distCubed = np.linalg.norm(derivative)**3
            fout += k * charge.magnitude*derivative/distCubed

        return fout / np.linalg.norm(fout)

    def stop_near_charge(t, f):
        min_distance = 5 # Set the minimum distance to stop the line
        infinity = 100000
        for charge in chargeList:
            distance = np.linalg.norm(f - charge.position)
            if distance < min_distance or np.linalg.norm(f - [500,500]) > infinity:
                return 0  # Stop the integration
        return 1  # Continue integration

    stop_near_charge.terminal = True
    stop_near_charge.direction = -1

    constLinesPerUnitCharge = 8
    constMaxT = 10000

    k = 1/(4*np.pi*cnst.epsilon_0)
    fieldLineList = []

    for charge in chargeList:
        if charge.magnitude > 0:
            for i in range(constLinesPerUnitCharge * abs(charge.magnitude)):
                start = time.time()
                initX = charge.position[0] + np.cos(i * 2 * np.pi / constLinesPerUnitCharge * abs(charge.magnitude)) * 5
                initY = charge.position[1] + np.sin(i * 2 * np.pi / constLinesPerUnitCharge * abs(charge.magnitude)) * 5
                initPos: np.ndarray = np.array([initX, initY])

                timeSpan = [0, constMaxT]
                num_points = 1000
                t_eval = np.linspace(timeSpan[0], timeSpan[1], num_points)

                solution = solve_ivp(dfdx, timeSpan, initPos, method='RK23', events=stop_near_charge, t_eval=t_eval)

                iters += 1

                fieldLine = Line(np.stack(solution.y, axis=-1))
                fieldLineList.append(fieldLine)
                print(time.time() - start)
    
    return fieldLineList

def CreateFieldLinesODE() -> list: #Not currently working
    def dfdx(t: float, f: np.ndarray) -> np.ndarray:
        fout: np.ndarray = np.array([0,0], dtype=np.float64)

        for charge in chargeList:
            derivative = f - charge.position
            distCubed = np.linalg.norm(derivative)
            fout += k * charge.magnitude*derivative/distCubed

        return fout
    
    constLinesPerUnitCharge = 8
    constMaxT = 5.01
    k = 1 / (4 * np.pi * cnst.epsilon_0)
    fieldLineList = []

    for charge in chargeList:
        if charge.magnitude > 0:
            for i in range(constLinesPerUnitCharge):
                start = time.time()
                initX = charge.position[0] + np.cos(i * 2 * np.pi / constLinesPerUnitCharge) * 50
                initY = charge.position[1] + np.sin(i * 2 * np.pi / constLinesPerUnitCharge) * 50
                initPos = np.array([initX, initY])

                # Initialize the ODE solver
                solver = ode(dfdx)
                solver.set_integrator('vode', method='bdf', order=5, nsteps=3000)  # Choose 'vode' as the solver
                solver.set_initial_value(initPos, 5)

                points = [initPos]  # List to store computed points
                start_time = datetime.datetime.now()

                # Integrate the ODE step-by-step
                while solver.successful() and solver.t < constMaxT:
                    step = 10**-5
                    solver.integrate(solver.t + step)
                    points.append(solver.y)

                print(datetime.datetime.now() - start_time, i, initPos)

                # Convert the list of points to a Line object
                fieldLine = Line(np.array(points))
                fieldLineList.append(fieldLine)
                print(time.time() - start)
        
        return fieldLineList

def CreateFieldLinesIter() -> list: #1 - 12ms per line
    linesPerUnitCharge = 8
    initRadius = 5
    dl = 5

    fieldLineList = []

    for charge in chargeList:
        if charge.magnitude > 0:
            for i in range(charge.magnitude * linesPerUnitCharge):
                initX: float = charge.position[0] + np.cos(i * 2 * np.pi / (charge.magnitude * linesPerUnitCharge)) * initRadius
                initY: float = charge.position[1] + np.sin(i * 2 * np.pi / (charge.magnitude * linesPerUnitCharge)) * initRadius
                initPos: np.ndarray = np.array([initX, initY])

                solutionArray: list = []

                solutionArray.append(initPos)

                ended: bool = False

                currentPos: np.ndarray = initPos

                start = time.time()
                j = 0
                while not ended: #between 0.3 and 0.8ms per loop ~50 loops
                    force: np.ndarray = GetForce(currentPos)
                    forceMag: float = np.linalg.norm(force)
                    endPos: np.ndarray = currentPos + (force/forceMag) * dl
                    solutionArray.append(endPos)
                    
                    for charge in chargeList:
                        if np.linalg.norm(endPos - charge.position) < 10 or np.linalg.norm(endPos) > 2000:
                            ended = True
                    currentPos = endPos
                    j += 1
                end = time.time()
                print((end - start) / j, j)
                fieldLine: Line = Line(np.array(solutionArray))
                fieldLineList.append(fieldLine)

    return fieldLineList

def CreateFieldLinesC() -> list:
    linesPerUnitCharge = 10
    initRadius = 5
    dl = 5
    windowX = windowSize[0]
    windowY = windowSize[1]

    fieldLineList = []

    chargeListC = ConvertChargeListToC(chargeList)

    for chargeIndex, charge in enumerate(chargeList):
        if charge.magnitude > 0:
            for i in range(charge.magnitude * linesPerUnitCharge):
                solutionArray = iterationMethod.getSolutionArray(chargeListC[chargeIndex], chargeListC, len(chargeListC), i, linesPerUnitCharge, initRadius, dl, windowX, windowY)
                j = 0
                solutionArrayList = []
                while solutionArray[j][0] != -1 and solutionArray[j][1] != -1 and j < 999:
                    solutionArrayList.append(solutionArray[j])
                    j += 1
                fieldLine: Line = Line(np.array(solutionArrayList))
                fieldLineList.append(fieldLine)

    return fieldLineList

def ConvertChargeListToC(chargeList) -> ctypes.Array:


    chargeListC = []
    for charge in chargeList:
        posArray = (ctypes.c_double * len(charge.position))(*charge.position)
        chargeListC.append(ChargeC(posArray, charge.magnitude))

    ChargeCArrayType = ChargeC * len(chargeListC)

    return ChargeCArrayType(*chargeListC)


def CreateFieldLinesOpenGL() -> list:
    linesPerUnitCharge = 8
    initRadius = 5
    dl = 5
    ended: bool = False 
    fieldLineList = []

    charges = np.zeros(len(chargeList), dtype=[('position', np.float32, 2), ('magnitude', np.float32), ('padding', np.float32)])
    charges['position'] = [charge.position for charge in chargeList]
    charges['magnitude'] = [charge.magnitude for charge in chargeList]


    posList = []

    for charge in chargeList:
        if charge.magnitude > 0:
            for i in range(charge.magnitude * linesPerUnitCharge): 
                initX: np.float32 = charge.position[0] + np.cos(i * 2 * np.pi / (charge.magnitude * linesPerUnitCharge)) * initRadius
                initY: np.float32 = charge.position[1] + np.sin(i * 2 * np.pi / (charge.magnitude * linesPerUnitCharge)) * initRadius
                initPos: np.ndarray = np.array([initX, initY])

                posList.append(initPos)

    posArray = np.array(posList).flatten()

    print(pos for pos in posArray)

    j = 0
    while not ended:
        j += 1
        
        forces = np.zeros(len(charges)*len(posArray)*2, dtype=np.float32)

        chargesBuffer, forcesBuffer, positionsBuffer = BindBuffers(charges, forces, posArray)
        forces = getResults(program, forces, charges, posArray, forcesBuffer)

        results = np.frombuffer(forces, dtype=np.float32).reshape(len(chargeList), len(posArray), 2)

        force: np.ndarray = np.sum(results, axis=1)
        print("force ", force)
        #forceMag: float = np.linalg.norm(force)
        #endPos: np.ndarray = currentPos + (force/forceMag) * dl

        if j > 2:
            ended = True

    #solutionArray.append(endPos)
    #end = time.time()
    #fieldLine: Line = Line(np.array(solutionArray))
    #fieldLineList.append(fieldLine)
    return fieldLineList


program, shader = initOpenGL()


iterationMethod = ctypes.CDLL("./iterationMethod.dll")

iterationMethod.getSolutionArray.argtypes = [ChargeC, ctypes.POINTER(ChargeC), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
iterationMethod.getSolutionArray.restype = ctypes.POINTER(ctypes.c_double * 2)

windowSize: list = [2000, 1300]

pygame.init()

screen: pygame.surface = pygame.display.set_mode(windowSize)
clock: pygame.time.Clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)

chargeList = CreateCharges()

while True:
    iters = 0
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        for charge in chargeList:
            charge.HandleDragging(event)


    fieldLineList = CreateFieldLinesOpenGL()


    for charge in chargeList:
        charge.Draw(screen)

    for fieldLine in fieldLineList:
        fieldLine.Draw(screen)

    fps = int(clock.get_fps())
    fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))  # Display at top-left corner

    clock.tick(165)
    pygame.display.flip()