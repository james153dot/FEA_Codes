import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_min, x_max = -10.0, 10.0
dx = 0.05
x = np.arange(x_min, x_max, dx)
N = len(x)  # number of points
L = x_max - x_min

def one_soliton(x, v, x0=0.0):
    # One-soliton initial condition: u(x,0) = v/2 * sech^2( (sqrt(v)/2)*(x - x0) )
    return 0.5*v*(1.0/np.cosh((np.sqrt(v)/2.0)*(x - x0)))**2

# Create the initial condition by superposing two solitons with different x0
# v=14 at x0=-3 and v=8 at x0=3
u = one_soliton(x, 14, -3) + one_soliton(x, 8, 3)

def compute_derivatives(u):
    # Periodic indexing
    up2 = u[(np.arange(N)+2) % N]
    up1 = u[(np.arange(N)+1) % N]
    um1 = u[(np.arange(N)-1) % N]
    um2 = u[(np.arange(N)-2) % N]
    
    # u_x
    ux = (up1 - um1) / (2.0*dx)
    
    # u_xxx
    uxxx = (-up2 + 2*up1 - 2*um1 + um2)/(2.0*dx**3)
    return ux, uxxx

# RHS of KdV: du/dt = -6*u*u_x - u_xxx
def f(u):
    ux, uxxx = compute_derivatives(u)
    return -6.0*u*ux - uxxx

# Time-stepping parameters (from previous stability analysis)
dt = 3.0e-5
t_final = 10.0
n_steps = int(t_final/dt)

# Integrate in time using RK4
for n in range(n_steps):
    k1 = dt * f(u)
    k2 = dt * f(u + 0.5*k1)
    k3 = dt * f(u + 0.5*k2)
    k4 = dt * f(u + k3)
    u = u + (k1 + 2*k2 + 2*k3 + k4)/6.0

# Plot the solution at t = 10
plt.figure(figsize=(8,4))
plt.plot(x, u, label='t=10')
plt.title('Two-soliton initial condition: v=14 (x0=-3), v=8 (x0=3)')
plt.xlabel('x')
plt.ylabel('u(x,10)')
plt.grid(True)
plt.legend()
plt.show()
