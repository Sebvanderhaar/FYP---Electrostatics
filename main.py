import numpy as np
import scipy.constants as cnst
from scipy.integrate import solve_ivp
import datetime
import pygame
import sys
import math

count = 0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

class Slider:
    def __init__(self, x, y, min_val=-100, max_val=100):
        self.x = x
        self.y = y
        self.width = 300
        self.height = 10
        self.handle_x = x  # Initial handle position (at minimum value)
        self.handle_radius = 15
        self.min_val = min_val
        self.max_val = max_val
        self.value = min_val
        self.is_dragging = False

    def draw(self, surface):
        # Draw slider track (line)
        pygame.draw.rect(surface, WHITE, (self.x, self.y, self.width, self.height))
        
        # Draw handle (circle)
        pygame.draw.circle(surface, BLUE, (self.handle_x, self.y + self.height // 2), self.handle_radius)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_over_handle(event.pos):
                self.is_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_dragging = False
        if self.is_dragging:
            self.update_handle_position(pygame.mouse.get_pos())

    def is_over_handle(self, pos):
        # Use distance from the center of the handle to detect if the mouse is over the handle
        handle_center = (self.handle_x, self.y + self.height // 2)
        distance = math.sqrt((pos[0] - handle_center[0]) ** 2 + (pos[1] - handle_center[1]) ** 2)
        return distance <= self.handle_radius

    def update_handle_position(self, pos):
        # Clamp the handle within the slider bounds
        self.handle_x = max(self.x, min(pos[0], self.x + self.width))
        self.update_value()

    def update_value(self):
        # Convert handle position to a value
        percentage = (self.handle_x - self.x) / self.width
        self.value = self.min_val + (percentage * (self.max_val - self.min_val))
        if self.value == 0:
            self.value = 0.0001

    def get_value(self):
        return self.value

def DrawEquipotentialDE(surface: pygame.surface, charges: np.ndarray) -> None:
    k = 1/(4*np.pi*cnst.epsilon_0)  # Example constant
    r_n = charges[:, 1]  # Example sources at (1, 0) and (-1, 0)
    q_n = charges[:, 0]  # Example charges

    # Define the system of ODEs
    def f_derivative(t, f):
        x, y = f  # Unpack current x and y positions
        fx, fy = 0, 0  # Initialize derivatives

        for i in range(len(r_n)):
            dx, dy = x - r_n[i][0], y - r_n[i][1]
            distance_cubed = (dx**2 + dy**2)**1.5
            fx += k * q_n[i] * dx / distance_cubed
            fy += k * q_n[i] * dy / distance_cubed

        return [fx, fy]

    # Initial conditions
    f_initial = [500, 500]  # Example starting point

    # Solve the system
    t_span = [0, 1000]  # Time interval
    t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time points for evaluation

    start = datetime.datetime.now()
    solution = solve_ivp(f_derivative, t_span, f_initial, method='BDF')
    print("Time taken for solver: ", datetime.datetime.now()-start)

    # Extract the solution
    x_solution, y_solution = solution.y

    for i in range(len(x_solution) - 1):
        pygame.draw.line(surface, (255, 0, 0), [x_solution[i], y_solution[i]], [x_solution[i+1], y_solution[i+1]])


def GetAxisOfGivenLength(array: np.ndarray, length: float) -> int:
    for i in range(array.ndim):
        if array.shape[i] == length:
            return i


def CoulombForceField(r: np.ndarray, chargeQ: float, chargePos: np.ndarray) -> np.ndarray:
    axis: int = GetAxisOfGivenLength(r, 2)
    k: float = chargeQ / (4 * cnst.pi * cnst.epsilon_0)
    r1: np.ndarray = r - chargePos
    r1MagCubed: float = np.linalg.norm(r1, axis=axis)**3
    force = k * np.divide(r1, r1MagCubed[..., np.newaxis])

    return force

def CoulombPotentialField(r: np.ndarray, chargeQ: float, chargePos: np.ndarray) -> np.ndarray:
    axis: int = GetAxisOfGivenLength(r, 2)
    k: float = chargeQ / (4 * cnst.pi * cnst.epsilon_0)
    r1: np.ndarray = r - chargePos
    r1MagRecip: float = np.linalg.norm(r1, axis=axis)**-1
    potential = k*r1MagRecip

    return potential


def DrawVectorField(surface: pygame.surface, xVectors: int, yVectors: int, chargeArray: np.ndarray, vectorLength: float) -> None:
    colour: tuple = (200, 200, 200)

    windowWidth: int
    windowHeight: int

    windowWidth, windowHeight = surface.get_size()

    xPositions: np.ndarray = np.linspace(0, windowWidth, num=xVectors)
    yPositions: np.ndarray = np.linspace(0, windowHeight, num=yVectors)

    positionGrid: np.ndarray = np.stack(np.meshgrid(xPositions, yPositions), axis=-1)

    forceField: np.ndarray = np.zeros(positionGrid.shape, dtype=np.float64)

    for charge in chargeArray:
        forceField += CoulombForceField(positionGrid, charge[0], charge[1])

    for xIndex, xPos in enumerate(xPositions):
        for yIndex, yPos in enumerate(yPositions):
            force: np.ndarray = forceField[yIndex][xIndex]
            forceMag: float = np.linalg.norm(force)
            startPos: np.ndarray = [xPos, yPos]

            endPos: np.ndarray = startPos + vectorLength * force / forceMag

            pygame.draw.line(surface, colour, startPos, endPos)

def DrawEquipotentialLines(surface: pygame.surface, chargeArray: np.ndarray) -> None:
    linesPerUnitCharge = 8
    for charge in chargeArray:
        lines = np.abs(charge[0]) * linesPerUnitCharge
        for i in range(lines):
            DrawEquipotentialLine(surface, charge, i, lines, chargeArray)

def DrawEquipotentialLine(surface: pygame.surface, charge: np.ndarray, i: int, totLines: int, chargeArray: np.ndarray) -> None:
    dl: float = 0.1
    r: float = 30

    colour: tuple = (200, 200, 200)
    initOffset: np.ndarray = np.array([r * i, 0])
    startPos: np.ndarray = np.array(charge[1] + initOffset)
    ended: bool = False
    j: int = 0

    theta = np.pi/2

    rotationMatrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
    ])

    while not ended:
        force: np.ndarray = np.zeros((2), dtype=np.float64)
        for chargeForce in chargeArray:
            force += CoulombForceField(startPos, chargeForce[0], chargeForce[1])
        forceMag: float = np.linalg.norm(force)

        rotatedForce = force @ rotationMatrix

        endPos: np.ndarray = startPos + (rotatedForce * dl) / forceMag

        pygame.draw.line(surface, colour, startPos, endPos)

        startPos = endPos

        if np.linalg.norm(endPos - np.array([windowWidth, windowHeight])/2) > 400:
            ended = True
        for charge in chargeArray:
            if charge[0] < 0:
                if np.linalg.norm(charge[1] - endPos) < dl + r + 10:
                    ended = True
        if j > 50:
            ended = True
            print("ended early")
        j += 1

def DrawCharges(surface: pygame.surface, chargeArray: np.ndarray) -> None:
    colour = (200, 200, 200)
    radius = 10
    for charge in chargeArray:
        pygame.draw.circle(surface, colour, charge[1], radius)

def DrawFieldLines(surface: pygame.surface, chargeArray: np.ndarray) -> None:
    linesPerUnitCharge = 8
    for charge in chargeArray:
        lines = np.abs(charge[0]) * linesPerUnitCharge
        for i in range(lines):
            DrawFieldLine(surface, charge, i, lines, chargeArray)

def ScaleLineOnAngle(force: np.ndarray, prevForce: np.ndarray, forceMag: float, prevForceMag: float, critCharge: bool) -> float:
    angleBetween: float = abs(np.dot(force, prevForce)/(forceMag*prevForceMag))
    power = slider.value
    minLength = 1
    maxLength = 5

    a = (maxLength - minLength)/(np.e**power - 1)
    b = minLength - a
    #if critCharge:
        #(a, b, angleBetween, a*np.e**(power*angleBetween) + b)
    return a*np.e**(power*angleBetween) + b

def DrawFieldLine(surface: pygame.surface, charge: np.ndarray, i: int, totLines: int, chargeArray: np.ndarray) -> None:
    dl: float = 5
    r: float = 2

    colour: tuple = (200, 200, 200)
    initOffset: np.ndarray = np.array([
        r * np.cos(i * 2 * np.pi / totLines),
        r * np.sin(i * 2 * np.pi / totLines)
    ])
    startPos: np.ndarray = np.array(charge[1] + initOffset)
    ended: bool = False
    j: int = 0

    firstIter: bool = True
    
    critCharge: bool = False

    while not ended:
        force: np.ndarray = np.zeros((2), dtype=np.float64)
        for chargeForce in chargeArray:
            force += CoulombForceField(startPos, chargeForce[0], chargeForce[1])
        forceMag: float = np.linalg.norm(force)

        if firstIter:
            endPos: np.ndarray = startPos + (force * dl) / forceMag
            firstIter = False
        else:
            if charge[0] == 2 and i == 15 and j == 9:
                critCharge: bool = True
            else:
                critCharge: bool = False
            endPos: np.ndarray = startPos + (force * ScaleLineOnAngle(force, prevForce, forceMag, prevForceMag, critCharge) * dl) / forceMag

        if critCharge:
            pygame.draw.line(surface, (200,0,0), startPos, endPos)
        else:
            pygame.draw.line(surface, colour, startPos, endPos)

        startPos = endPos

        prevForce: np.ndarray = force
        prevForceMag: float = forceMag

        if np.linalg.norm(endPos - np.array([windowWidth, windowHeight])/2) > 400:
            ended = True
        for charge in chargeArray:
            if charge[0] < 0:
                if np.linalg.norm(charge[1] - endPos) < dl + r + 10:
                    ended = True
        if j > 100:
            ended = True
            print("ended early")
        j += 1

chargeArray: np.ndarray = np.array([[-2, [100, 500]], [1, [400, 300]], [2, [700, 500]]], dtype=object)

chargePos = np.array((400, 300))
chargeQ = -1

windowWidth, windowHeight = 800, 600

pygame.init()

screen: pygame.surface = pygame.display.set_mode((windowWidth, windowHeight))
clock = pygame.time.Clock()

dragger: int = 0
dragging: bool = False

slider = Slider(150, 200)

while True:
    startTime = datetime.datetime.now()
    for event in pygame.event.get():
        slider.handle_event(event)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for chargeIndex, charge in enumerate(chargeArray):
                    if np.linalg.norm(
                            np.array(pygame.mouse.get_pos()) - charge[1]) < 10:
                        dragging = True
                        dragger = chargeIndex
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False

    eventChecksTime = datetime.datetime.now()

    if dragging:
        chargeArray[dragger][1] = pygame.mouse.get_pos()

    screen.fill((0, 0, 0))

    slider.draw(screen)

    mouseX, mouseY = pygame.mouse.get_pos()

    #chargePos = np.array([mouseX, mouseY])

    #chargeArray[2] = [chargeQ, chargePos]

    #DrawVectorField(screen, 50, 50, chargeArray, 10)
    DrawCharges(screen, chargeArray)

    #drawChargesTime = datetime.datetime.now()

    #DrawFieldLines(screen, chargeArray)

    #DrawEquipotentialLines(screen, chargeArray)

    DrawEquipotentialDE(screen, chargeArray)

    endTime = datetime.datetime.now()

    font = pygame.font.SysFont(None, 36)
    text = font.render(f"Value: {int(slider.get_value())}", True, RED)
    screen.blit(text, (250, 300))


    
