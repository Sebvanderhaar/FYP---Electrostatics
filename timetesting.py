from scipy.integrate import solve_ivp
import numpy as np
import datetime

# Define the ODE function, simple example: dy/dt = -y
def ode_func(t, y):
    return -y

# Initial conditions and time span
y0 = [1.0]
t_span = (0, 5)

# Solve the ODE with method='RK45' (fast for most problems) and low tolerance for speed
start = datetime.datetime.now()
solution = solve_ivp(ode_func, t_span, y0, method='RK45', rtol=1e-6, atol=1e-9)
print(datetime.datetime.now() - start)

# Output result
print("Solution times:", solution.t)
print("Solution values:", solution.y)
