import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_min, x_max = -5.0, 5.0
dx = 0.05
x = np.arange(x_min, x_max, dx)
N = len(x)

rho_max = 1.0
umax = 1.0
rho_left = 0.5

# Light timing
T1 = 1.0  # red time
T2 = 1.0  # green time
T = T1 + T2

# Two traffic lights:
# Light 1 at x = -dx/2
# Light 2 at x = 0.1 - dx/2
light1_pos = -dx/2
light2_pos = 0.1 - dx/2

# Find interface indices for the two lights
j_light1 = int((light1_pos - x_min)/dx - 0.5)
j_light2 = int((light2_pos - x_min)/dx - 0.5)

# Time step
dt = 0.4*dx
steps_per_period = int(T/dt)
if steps_per_period < 1:
    raise ValueError("Time step too large or T too small.")

def f(rho):
    return rho*(1.0 - rho)

def godunov_flux(rhol, rhor):
    if rhol <= rhor:
        return min(f(rhol), f(rhor))
    else:
        # Check if the interval contains rho=0.5
        left_bound = min(rhol, rhor)
        right_bound = max(rhol, rhor)
        if left_bound <= 0.5 <= right_bound:
            return 0.25
        else:
            return max(f(rhol), f(rhor))

def apply_bc(rho):
    rho[0] = rho_left
    rho[-1] = rho[-2]
    return rho

def compute_fluxes(rho, t, tau):
    F = np.zeros(N+1)
    for i in range(N+1):
        if i == 0:
            rhol = rho_left
            rhor = rho[i] if i < N else rho_left
        elif i == N:
            rhol = rho[i-1]
            rhor = rho[i-1]
        else:
            rhol = rho[i-1]
            rhor = rho[i]

        # Traffic light 1
        if i-1 == j_light1:
            # Phase of light 1
            phase = (t % T)
            if phase < T1:
                # Red
                F[i] = 0.0
                continue
            # Else green, normal flux

        # Traffic light 2
        if i-1 == j_light2:
            # Phase of light 2 shifted by tau
            phase = ((t - tau) % T)
            if phase < T1:
                # Red
                F[i] = 0.0
                continue
            # Else green, normal flux

        F[i] = godunov_flux(rhol, rhor)
    return F

def step(rho, t, tau):
    apply_bc(rho)
    F = compute_fluxes(rho, t, tau)
    rho_new = rho.copy()
    for i in range(N):
        rho_new[i] = rho[i] - (dt/dx)*(F[i+1]-F[i])
    return rho_new

def run_simulation_for_tau(tau, max_cycles=500, tol=1e-5, measure_index=None):
    # Initialize rho
    rho = 0.5*np.ones(N)
    t = 0.0

    if measure_index is None:
        # measure somewhere in the interior, say the middle
        measure_index = N//2

    old_qdot = None
    for cycle in range(max_cycles):
        # Run one full period
        q_sum = 0.0
        for n in range(steps_per_period):
            q_sum += f(rho[measure_index])
            rho = step(rho, t, tau)
            t += dt

        qdot = q_sum / steps_per_period
        # Check for convergence
        if old_qdot is not None and abs(qdot - old_qdot) < tol:
            return qdot
        old_qdot = qdot

    return qdot

# Compute capacity for tau = k*T/20, k=0,...,19
taus = [(k*T/20.0) for k in range(20)]
capacities = []

for tau in taus:
    qdot = run_simulation_for_tau(tau, max_cycles=1000, tol=1e-5)
    capacities.append(qdot)
    print(f"tau={tau:.4f}, qdot={qdot:.6f}")

# Plot results
plt.figure(figsize=(8,4))
plt.plot(taus, capacities, marker='o')
plt.xlabel(r'$\tau$')
plt.ylabel('Capacity (Average Flow)')
plt.title('Capacity vs. Delay $\tau$')
plt.grid(True)
plt.show()

# Find optimal tau
optimal_index = np.argmax(capacities)
optimal_tau = taus[optimal_index]
optimal_qdot = capacities[optimal_index]
print(f"Optimal delay tau = {optimal_tau:.4f} with capacity qdot = {optimal_qdot:.6f}")
