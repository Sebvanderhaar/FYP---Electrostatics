import numpy as np
import scipy.constants as cnst
import datetime
import pygame
import sys

count = 0


def GetAxisOfGivenLength(array: np.ndarray, length: float) -> int:
    for i in range(array.ndim):
        if array.shape[i] == length:
            return i


def CoulombForceField(r: np.ndarray, chargeQ: float,
                      chargePos: np.ndarray) -> np.ndarray:
    axis: int = GetAxisOfGivenLength(r, 2)
    k: float = chargeQ / (4 * cnst.pi * cnst.epsilon_0)
    r1: np.ndarray = r - chargePos
    r1MagCubed: float = np.linalg.norm(r1, axis=axis)**3
    force = k * np.divide(r1, r1MagCubed[..., np.newaxis])

    return force


def DrawVectorField(surface: pygame.surface, xVectors: int, yVectors: int,
                    chargeArray: np.ndarray, vectorLength: float) -> None:
    colour: tuple = (200, 200, 200)

    windowWidth: int
    windowHeight: int

    windowWidth, windowHeight = surface.get_size()

    xPositions: np.ndarray = np.linspace(0, windowWidth, num=xVectors)
    yPositions: np.ndarray = np.linspace(0, windowHeight, num=yVectors)

    positionGrid: np.ndarray = np.stack(np.meshgrid(xPositions, yPositions),
                                        axis=-1)

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


def DrawCharges(surface: pygame.surface, chargeArray: np.ndarray) -> None:
    colour = (200, 200, 200)
    radius = 10
    for charge in chargeArray:
        pygame.draw.circle(surface, colour, charge[1], radius)


def DrawFieldLines(surface: pygame.surface, chargeArray: np.ndarray) -> None:
    linesPerUnitCharge = 4
    for charge in chargeArray:
        lines = np.abs(charge[0]) * linesPerUnitCharge
        for i in range(lines):
            DrawFieldLine(surface, charge, i, lines, chargeArray)


def DrawFieldLine(surface: pygame.surface, charge: np.ndarray, i: int,
                  totLines: int, chargeArray: np.ndarray) -> None:
    dl: float = 20
    r: float = 5
    colour: tuple = (200, 200, 200)
    initOffset: np.ndarray = np.array([
        r * np.cos(i * 2 * np.pi / totLines),
        r * np.sin(i * 2 * np.pi / totLines)
    ])
    startPos: np.ndarray = np.array(charge[1] + initOffset)
    ended: bool = False
    while not ended:
        force: np.ndarray = np.zeros((2), dtype=np.float64)
        for chargeForce in chargeArray:
            force += CoulombForceField(startPos, chargeForce[0],
                                       chargeForce[1])
        forceMag: float = np.linalg.norm(force)
        endPos: np.ndarray = startPos + (force * dl) / forceMag
        pygame.draw.line(surface, colour, startPos, endPos)
        startPos = endPos
        if endPos[0] > 1500 or endPos[0] < -1500 or endPos[1] > 1500 or endPos[
                1] < -1500:
            ended = True

        for charge in chargeArray:
            if charge[0] < 0:
                if np.linalg.norm(charge[1] - endPos) < r:
                    ended = True


chargeArray: np.ndarray = np.array(
    [[-2, [100, 500]], [1, [400, 300]], [2, [700, 500]]], dtype=object)

chargePos = np.array((400, 300))
chargeQ = -1

windowWidth, windowHeight = 800, 600

pygame.init()

screen: pygame.surface = pygame.display.set_mode((windowWidth, windowHeight))
clock = pygame.time.Clock()

dragger: int = 0
dragging: bool = False

while True:
    startTime = datetime.datetime.now()
    for event in pygame.event.get():
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

    mouseX, mouseY = pygame.mouse.get_pos()

    #chargePos = np.array([mouseX, mouseY])

    #chargeArray[2] = [chargeQ, chargePos]

    #DrawVectorField(screen, 50, 50, chargeArray, 10)
    DrawCharges(screen, chargeArray)

    drawChargesTime = datetime.datetime.now()

    DrawFieldLines(screen, chargeArray)

    endTime = datetime.datetime.now()

    pygame.display.flip()
