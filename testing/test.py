import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
#from openGLDrawing import *
from OpenGL.GL.shaders import compileShader, compileProgram
import time
import os

pygame.init()

sceen: pygame.display = pygame.display.set_mode((1, 1), DOUBLEBUF | OPENGL)

# Query the maximum workgroup size for each dimension
max_work_group_size = [0, 0, 0]
for i in range(3):  # 0 for x, 1 for y, 2 for z
    max_work_group_size[i] = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i)

# Print the results
print("Maximum workgroup size:")
print(f"X dimension: {max_work_group_size[0]}")
print(f"Y dimension: {max_work_group_size[1]}")
print(f"Z dimension: {max_work_group_size[2]}")

# Query the maximum number of workgroup invocations
max_invocations = glGetInteger(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
print(f"Maximum total workgroup invocations: {max_invocations}")