import numpy as np

# Parameters
x_min, x_max = -5.0, 5.0
dx = 0.05
x = np.arange(x_min, x_max, dx)
N = len(x)

rho_max = 1.0
umax = 1.0

# Initial condition: uniform rho=0.5
rho = 0.5*np.ones(N)

# Traffic light parameters
T1 = 1.0  # red duration
T2 = 1.0  # green duration
T = T1 + T2
light_position = -dx/2
# Find the interface index for the traffic light
j_light = int((light_position - x_min)/dx - 0.5)

print("Traffic light interface index:", j_light)
print("Traffic light physical position:", light_position)

# Time step: chosen to satisfy a CFL-like condition
dt = 0.4*dx
steps_per_period = int(T/dt)
if steps_per_period < 1:
    raise ValueError("Time step too large, no steps in one period.")

# Flux function and Godunov flux
def f(rho):
    return rho*(1.0 - rho)

def godunov_flux(rhol, rhor):
    # For rho_i <= rho_{i+1}: F = min(f(rhol), f(rhor))
    # For rho_i > rho_{i+1}: F = max(f(rhol), f(rhor))
    if rhol <= rhor:
        return min(f(rhol), f(rhor))
    else:
        # Interval [rhor, rhol] might contain the max at rho=0.5
        # Godunov picks the maximum if interval spans the max, else endpoint max.
        # This formula is simplified by checking endpoints and if max is in interval:
        left_bound = min(rhol, rhor)
        right_bound = max(rhol, rhor)
        # Check if [left_bound, right_bound] includes 0.5
        if left_bound <= 0.5 <= right_bound:
            # maximum at 0.5 is 0.25
            return 0.25
        else:
            # max at endpoints
            return max(f(rhol), f(rhor))

# Boundary conditions
rho_left = 0.5

def apply_bc(rho):
    # left boundary fixed
    rho[0] = rho_left
    # right boundary extrapolate
    rho[-1] = rho[-2]
    return rho

def compute_fluxes(rho, t):
    F = np.zeros(N+1)
    for i in range(N+1):
        if i == 0:
            # left boundary interface
            rhol = rho_left
            rhor = rho[i] if i < N else rho_left
        elif i == N:
            # right boundary interface
            rhol = rho[i-1]
            rhor = rho[i-1] # same cell (no outflow)
        else:
            rhol = rho[i-1]
            rhor = rho[i]

        # Traffic light condition
        if i-1 == j_light:
            # Determine phase of traffic light
            phase = (t % T)
            if phase < T1:
                # Red light: zero flux
                F[i] = 0.0
                continue
            # Green phase: normal flux calculation

        F[i] = godunov_flux(rhol, rhor)
    return F

def step(rho, t):
    apply_bc(rho)
    F = compute_fluxes(rho, t)
    rho_new = rho.copy()
    for i in range(N):
        rho_new[i] = rho[i] - (dt/dx)*(F[i+1]-F[i])
    return rho_new

# Choose a measurement index a few cells downstream of the traffic light
# If traffic light interface is j_light, measure at j_light+10 cells to the right (if possible)
measure_index = j_light + 10
if measure_index >= N:
    measure_index = N//2  # fallback to middle

print("Measuring flow at cell index:", measure_index, "x=", x[measure_index])

max_cycles = 2000
tolerance = 1e-5
old_qdot = None
t = 0.0

for cycle in range(max_cycles):
    # Run for one full period
    q_sum = 0.0
    for n in range(steps_per_period):
        q_sum += f(rho[measure_index])
        rho = step(rho, t)
        t += dt

    qdot = q_sum / steps_per_period
    print(f"After cycle {cycle+1}, qdot = {qdot:.8f}")

    if old_qdot is not None and abs(qdot - old_qdot) < tolerance:
        print(f"Converged after {cycle+1} cycles.")
        print(f"Steady periodic average flow qdot = {qdot:.8f}")
        break
    old_qdot = qdot
else:
    print("Did not converge within max_cycles cycles.")
    print(f"Last qdot = {qdot:.8f}")
