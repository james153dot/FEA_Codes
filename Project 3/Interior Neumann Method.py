import numpy as np
import matplotlib.pyplot as plt

# Function to compute the normal derivative on the boundary
def compute_boundary_condition(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    D1 = 5 + 4 * sin_theta
    D2 = 5 - 4 * sin_theta
    term1 = (1 + 2 * sin_theta) / D1
    term2 = (1 - 2 * sin_theta) / D2
    return term1 - term2

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
    term1 = np.log(np.sqrt(x**2 + (y + 2)**2))
    term2 = np.log(np.sqrt(x**2 + (y - 2)**2))
    return term1 - term2

# Main computation
def main():
    N_values = [10, 20, 40, 80, 160, 320]
    errors = []

    for N in N_values:
        # Discretize theta
        delta_theta = 2 * np.pi / N
        theta = -np.pi + (np.arange(N) + 0.5) * delta_theta

        # Compute boundary condition
        b = compute_boundary_condition(theta)

        # Form matrix A
        A = np.pi * np.eye(N) - (np.pi / N) * np.ones((N, N))

        # Add additional equation for u(0.5, 0) = 0
        x0, y0 = 0.5, 0.0
        G = np.zeros(N)
        for j in range(N):
            x_j = np.cos(theta[j])
            y_j = np.sin(theta[j])
            r = np.sqrt((x0 - x_j)**2 + (y0 - y_j)**2)
            G[j] = -np.log(r) / (2 * np.pi)
        # Augment the system
        A_aug = np.vstack([A, G * delta_theta])
        b_aug = np.append(-b, 0)

        # Solve the augmented system in least squares sense
        sigma, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)

        # Number of test points
        N_test = 100
        r_test = 0.75  # Test radius
        phi = 2 * np.pi * np.arange(N_test) / N_test
        x_test = r_test * np.cos(phi)
        y_test = r_test * np.sin(phi)

        # Compute numerical potential
        u_num = compute_potential(x_test, y_test, theta, sigma)

        # Compute exact potential
        u_exact = exact_potential(x_test, y_test)

        # Compute maximum error
        error = np.abs(u_num - u_exact)
        max_error = np.max(error)
        errors.append(max_error)

        print(f"N = {N}, Max Error = {max_error:.2e}")

    # Plotting the errors
    plt.figure(figsize=(8, 6))
    plt.loglog(N_values, errors, 'o-', label='Max Error')
    plt.xlabel('Number of Nyström Points N')
    plt.ylabel('Maximum Error')
    plt.title('Convergence of Nyström Method for Interior Neumann Problem')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
