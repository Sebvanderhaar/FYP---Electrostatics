import numpy as np
import scipy.constants as cnst
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import datetime
import pygame
import sys
import math


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

class Line:
    def __init__(self, points: np.ndarray):
        self.points: np.ndarray = points
        self.colour: np.ndarray = (200, 200, 200)
        self.width: float = 2

    def Draw(self, surface: pygame.surface) -> None:
        for startPos, endPos in zip(self.points, self.points[1:]):
            pygame.draw.line(surface, self.colour, startPos, endPos, self.width)

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

def CreateFieldLines() -> list:
    def dfdx(t: float, f: np.ndarray) -> np.ndarray:
        fout: np.ndarray = np.array([0,0], dtype=np.float64)

        for charge in chargeList:
            derivative = f - charge.position
            distCubed = np.linalg.norm(derivative)**3
            fout += k * charge.magnitude*derivative/distCubed

        return fout / np.linalg.norm(fout)

    def stop_near_charge(t, f):
        min_distance = 5 # Set the minimum distance to stop the line
        infinity = 1000
        for charge in chargeList:
            distance = np.linalg.norm(f - charge.position)
            if distance < min_distance or np.linalg.norm(f - [500,500]) > infinity:
                return 0  # Stop the integration
        return 1  # Continue integration

    stop_near_charge.terminal = True
    stop_near_charge.direction = -1

    constLinesPerUnitCharge = 8
    constMaxT = 1000

    k = 1/(4*np.pi*cnst.epsilon_0)
    fieldLineList = []

    for charge in chargeList:
        if charge.magnitude > 0:
            for i in range(constLinesPerUnitCharge * abs(charge.magnitude)):
                initX = charge.position[0] + np.cos(i * 2 * np.pi / constLinesPerUnitCharge * abs(charge.magnitude)) * 5
                initY = charge.position[1] + np.sin(i * 2 * np.pi / constLinesPerUnitCharge * abs(charge.magnitude)) * 5
                initPos: np.ndarray = np.array([initX, initY])

                timeSpan = [0, constMaxT]
                num_points = 1000  # Choose the number of points you want
                t_eval = np.linspace(timeSpan[0], timeSpan[1], num_points)

                #start = datetime.datetime.now()
                solution = solve_ivp(dfdx, timeSpan, initPos, method='RK23', events=stop_near_charge, t_eval=t_eval)
                #print(datetime.datetime.now() - start)

                fieldLine = Line(np.stack(solution.y, axis=-1))
                fieldLineList.append(fieldLine)
    
    return fieldLineList

def CreateFieldLinesODE() -> list:
    def dfdx(t: float, f: np.ndarray) -> np.ndarray:
        fout: np.ndarray = np.array([0,0], dtype=np.float64)

        for charge in chargeList:
            derivative = f - charge.position
            distCubed = np.linalg.norm(derivative)
            fout += k * charge.magnitude*derivative/distCubed

        return fout
    
    def f_derivative(t, f):
        x, y = f  # Unpack current x and y positions
        fx, fy = 0, 0  # Initialize derivatives

        for i in range(len(chargePositions)):
            dx, dy = x - chargePositions[i][0], y - chargePositions[i][1]
            distance_cubed = (dx**2 + dy**2)**1.5

            fx += k * chargeMagnitudes[i] * dx / distance_cubed
            fy += k * chargeMagnitudes[i] * dy / distance_cubed

        return [fx, fy]
    
    constLinesPerUnitCharge = 8
    constMaxT = 5.005

    chargePositions = [charge.position for charge in chargeList]
    chargeMagnitudes = [charge.magnitude for charge in chargeList]
    k = 1 / (4 * np.pi * cnst.epsilon_0)
    fieldLineList = []

    for charge in chargeList:
        if charge.magnitude > 0:
            for i in range(constLinesPerUnitCharge):
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
        
        return fieldLineList


windowSize: list = [1000, 1000]

pygame.init()

screen: pygame.surface = pygame.display.set_mode(windowSize)
clock: pygame.time.Clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)

chargeList = CreateCharges()

while True:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        for charge in chargeList:
            charge.HandleDragging(event)


    fieldLineList = CreateFieldLines()


    for charge in chargeList:
        charge.Draw(screen)

    for fieldLine in fieldLineList:
        fieldLine.Draw(screen)

    fps = int(clock.get_fps())
    fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))  # Display at top-left corner

    clock.tick(165)
    pygame.display.flip()