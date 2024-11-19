import numpy as np
import matplotlib.pyplot as plt

def fdcoeffF(k, xbar, x):
    n = len(x) - 1
    if k > n:
        raise ValueError('*** len(x) must be larger than k')

    m = k 
    c1 = 1.
    c4 = x[0] - xbar
    C = np.zeros((n + 1, m + 1))
    C[0, 0] = 1.
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.
        c5 = c4
        c4 = x[i] - xbar
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2 * c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    C[i, s] = c1 * (s * C[i - 1, s - 1] - c5 * C[i - 1, s]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for s in range(mn, 0, -1):
                C[j, s] = (c4 * C[j, s] - s * C[j, s - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2

    c = C[:, -1] 
    return c

# Constants for the problem
U = 5.0
T_blade = 1.2
T_inlet = 0.6
x_start, x_end = 0, 1
h = 5.0
n_points = 10
x = np.linspace(x_start, x_end, n_points)
dx = x[1] - x[0]

# First-order backward difference method
T_cool_first_order = np.zeros(n_points)
T_cool_first_order[0] = T_inlet

for i in range(1, n_points):
    dTdx = 2 * h * (T_blade - T_cool_first_order[i-1])
    T_cool_first_order[i] = T_cool_first_order[i-1] + dTdx * dx / U

# Second-order backward difference method
T_cool_second_order = np.zeros(n_points)
T_cool_second_order[0] = T_inlet

for i in range(2, n_points):
    dTdx_2 = 2 * h * (T_blade - T_cool_second_order[i-1])
    T_cool_second_order[i] = T_cool_second_order[i-1] + dTdx_2 * dx / U

plt.figure(figsize=(8, 6))
plt.plot(x, T_cool_first_order, label="First-Order Backward Difference")
plt.plot(x, T_cool_second_order, label="Second-Order Backward Difference")
plt.xlabel("x")
plt.ylabel("Tt_cool")
plt.title("Temperature Distribution Along the Cooling Channel")
plt.legend()
plt.grid(True)
plt.show()

def Tt_cool_exact(x):
    return 1.2 - 0.6 * np.exp(-2 * x)

def calculate_L2_error(T_num, T_exact, dx):
    return np.sqrt(np.sum((T_num - T_exact)**2) * dx)


max_intervals = 10
errors_first_order = []
errors_second_order = []
dx_values = []
max_iterations = 7 


for _ in range(max_iterations):
    x = np.linspace(x_start, x_end, max_intervals)
    dx = x[1] - x[0]
    dx_values.append(dx)

    T_cool_first_order = np.zeros(max_intervals)
    T_cool_first_order[0] = T_inlet
    for i in range(1, max_intervals):
        dTdx = 2 * h * (T_blade - T_cool_first_order[i-1])
        T_cool_first_order[i] = T_cool_first_order[i-1] + dTdx * dx / U

    T_cool_second_order = np.zeros(max_intervals)
    T_cool_second_order[0] = T_inlet
    for i in range(2, max_intervals):
        dTdx_2 = 2 * h * (T_blade - T_cool_second_order[i-1])
        T_cool_second_order[i] = T_cool_second_order[i-1] + dTdx_2 * dx / U

    T_exact = Tt_cool_exact(x)
    
    error_first_order = calculate_L2_error(T_cool_first_order, T_exact, dx)
    error_second_order = calculate_L2_error(T_cool_second_order, T_exact, dx)

    errors_first_order.append(error_first_order)
    errors_second_order.append(error_second_order)

    # Double the number of intervals
    max_intervals *= 2

# Log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(dx_values, errors_first_order, label="First-Order Backward Difference", marker='o')
plt.loglog(dx_values, errors_second_order, label="Second-Order Backward Difference", marker='s')
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$||e||_2$')
plt.title('Log-Log Plot of Error vs. $\Delta x$')
plt.legend()
plt.grid(True)
plt.show()

