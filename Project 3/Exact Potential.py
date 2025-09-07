import numpy as np
import matplotlib.pyplot as plt

# Function to compute the normal derivative on the boundary
def compute_boundary_condition(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    numerator = -12 * sin_theta
    denominator = 17 + 8 * np.cos(2 * theta)
    return numerator / denominator

# Function to compute the potential at test points
def compute_potential(x, y, theta, sigma):
    N = len(theta)
    delta_theta = 2 * np.pi / N
    potential = np.zeros_like(x)
    for j in range(N):
        x_j = np.cos(theta[j])
        y_j = np.sin(theta[j])
        r = np.sqrt((x - x_j)**2 + (y - y_j)**2)
        ln_r = np.log(r)
        potential += sigma[j] * ln_r
    potential *= -delta_theta / (2 * np.pi)
    return potential

# Exact potential function
def exact_potential(x, y):
    term1 = np.log(x**2 + (y - 0.5)**2)
    term2 = np.log(x**2 + (y + 0.5)**2)
    return term1 - term2

# Main computation
def main():
    N_values = [10, 20, 40, 80, 160, 320]
    delta_values = [1/4, 1/16, 1/256]
    errors = {delta: [] for delta in delta_values}

    for N in N_values:
        # Discretize theta
        delta_theta = 2 * np.pi / N
        theta = -np.pi + (np.arange(N) + 0.5) * delta_theta

        # Compute boundary condition
        b = compute_boundary_condition(theta)

        # Form matrix A
        A = np.pi * np.eye(N) + (np.pi / N) * np.ones((N, N))

        # Solve for sigma
        sigma = np.linalg.solve(A, -b)

        # Number of test points
        N_test = N

        # For each circle radius
        for delta in delta_values:
            r = 1 + delta
            phi = 2 * np.pi * np.arange(N_test) / N_test
            x_test = r * np.cos(phi)
            y_test = r * np.sin(phi)

            # Compute numerical potential
            u_num = compute_potential(x_test, y_test, theta, sigma)

            # Compute exact potential
            u_exact = exact_potential(x_test, y_test)

            # Compute maximum error
            error = np.abs(u_num - u_exact)
            max_error = np.max(error)

            errors[delta].append(max_error)

    # Plotting the errors
    plt.figure(figsize=(10, 6))
    for delta in delta_values:
        plt.loglog(N_values, errors[delta], 'o-', label=f'Radius = {1 + delta}')
    plt.xlabel('Number of discretization points N')
    plt.ylabel('Maximum Error')
    plt.title('Error Decay with N for Different Radii')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.show()

    # Print errors
    for delta in delta_values:
        print(f'Errors for radius = {1 + delta}:')
        for N, error in zip(N_values, errors[delta]):
            print(f'N = {N}, Error = {error:.2e}')
        print('\n')

if __name__ == "__main__":
    main()
