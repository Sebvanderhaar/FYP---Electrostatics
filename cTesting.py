import ctypes

iterationMethod = ctypes.CDLL("./iterationMethod.dll")

iterationMethod.square.argtypes = [ctypes.c_double]
iterationMethod.square.restype = ctypes.c_double

print(iterationMethod.square(9))