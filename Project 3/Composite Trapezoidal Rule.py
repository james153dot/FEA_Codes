import numpy as np
import matplotlib.pyplot as plt
import math

# Number of intervals
n_values = [1, 5, 10, 20, 40, 50, 80, 100]

# Composite trapezoidal rule function
def composite_trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = [a + i * h for i in range(1, n)]
    integral = h * (0.5 * f(a) + sum(f(x_i) for x_i in x) + 0.5 * f(b))
    return integral

# Integral (i): I = ∫₂¹⁰ 3x dx
def f1(x):
    return 3 * x

exact_value1 = 144  # Exact value

# Integral (ii): I = ∫₀^π sin(x) dx
def f2(x):
    return math.sin(x)

exact_value2 = 2  # Exact value

# Integral (iii): I = ∫₀¹ e^{2 cos(2πx)} dx
def f3(x):
    return math.exp(2 * math.cos(2 * math.pi * x))

exact_value3 = 2.279585302336067  # Exact value

# Integral (iv): I = ∫₀^{2π} |cos(x)| dx
def f4(x):
    return abs(math.cos(x))

exact_value4 = 4  # Exact value

# Function to compute errors and plot
def compute_and_plot_errors(f, a, b, exact_value, integral_label):
    errors = []
    for n in n_values:
        approx = composite_trapezoidal_rule(f, a, b, n)
        error = abs(exact_value - approx)
        errors.append(error)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(n_values, errors, 'o-', label='Error')
    plt.xlabel('Number of intervals (n)')
    plt.ylabel('Absolute Error')
    plt.title(f'Error vs. n for {integral_label}')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
    
    # Print errors for reference
    print(f"Errors for {integral_label}:")
    for n, error in zip(n_values, errors):
        print(f"n = {n}, Error = {error}")
    print("\n")

compute_and_plot_errors(f1, 2, 10, exact_value1, 'Integral (i): ∫₂¹⁰ 3x dx')
compute_and_plot_errors(f2, 0, math.pi, exact_value2, 'Integral (ii): ∫₀^π sin(x) dx')
compute_and_plot_errors(f3, 0, 1, exact_value3, 'Integral (iii): ∫₀¹ e^{2 cos(2πx)} dx')
compute_and_plot_errors(f4, 0, 2 * math.pi, exact_value4, 'Integral (iv): ∫₀^{2π} |cos(x)| dx')
