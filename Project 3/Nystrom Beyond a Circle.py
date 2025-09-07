import numpy as np
import matplotlib.pyplot as plt

# Number of discretization points
N_values = [10, 20, 40, 80, 160, 320]

# Function f(theta)
def f(theta):
    return 1 / (3 + 2 * np.cos(theta) + np.cos(2 * theta))

# Parameterization of the ellipse
def x_theta(theta):
    return 2 * np.cos(theta)

def y_theta(theta):
    return np.sin(theta)

# Derivatives of x and y with respect to theta
def dx_dtheta(theta):
    return -2 * np.sin(theta)

def dy_dtheta(theta):
    return np.cos(theta)

# Compute the normal vector at theta
def normal_vector(theta):
    dx = dx_dtheta(theta)
    dy = dy_dtheta(theta)
    # Normal vector components (dy, -dx)
    n_x = dy
    n_y = -dx
    # Normalize
    norm = np.sqrt(n_x**2 + n_y**2)
    n_x /= norm
    n_y /= norm
    return n_x, n_y

# Compute the arc length differential dGamma
def dGamma(theta):
    dx = dx_dtheta(theta)
    dy = dy_dtheta(theta)
    return np.sqrt(dx**2 + dy**2)

# Compute the potential at a point (x, y)
def compute_potential(x, y, x_nodes, y_nodes, sigma, dGamma_nodes):
    N = len(sigma)
    potential = np.zeros_like(x)
    for j in range(N):
        r = np.sqrt((x - x_nodes[j])**2 + (y - y_nodes[j])**2)
        ln_r = np.log(r)
        potential += sigma[j] * ln_r * dGamma_nodes[j]
    potential *= -1 / (2 * np.pi)
    return potential

# Main computation
def main():
    # Reference solution
    N_ref = 1000
    theta_ref = np.linspace(0, 2 * np.pi, N_ref, endpoint=False)
    x_ref = x_theta(theta_ref)
    y_ref = y_theta(theta_ref)
    n_x_ref, n_y_ref = normal_vector(theta_ref)
    dGamma_ref = dGamma(theta_ref)
    f_ref = f(theta_ref)
    # Form the matrix and solve for sigma_ref
    A_ref = np.zeros((N_ref, N_ref))
    b_ref = f_ref
    for i in range(N_ref):
        for j in range(N_ref):
            if i != j:
                r_ij_x = x_ref[i] - x_ref[j]
                r_ij_y = y_ref[i] - y_ref[j]
                r_ij_sq = r_ij_x**2 + r_ij_y**2
                K_ij = (r_ij_x * n_x_ref[i] + r_ij_y * n_y_ref[i]) / r_ij_sq
                A_ref[i, j] = K_ij * dGamma_ref[j]
        # Diagonal entries (principal value)
        A_ref[i, i] = -np.pi
    sigma_ref = np.linalg.solve(A_ref, -b_ref)

    # Compute potential at test points using reference solution
    N_test = 100
    phi_test = np.linspace(0, 2 * np.pi, N_test, endpoint=False)
    r_test = 3  # Radius outside the ellipse
    x_test = r_test * np.cos(phi_test)
    y_test = r_test * np.sin(phi_test)
    u_ref = compute_potential(x_test, y_test, x_ref, y_ref, sigma_ref, dGamma_ref)

    # Initialize error list
    errors = []

    for N in N_values:
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x_nodes = x_theta(theta)
        y_nodes = y_theta(theta)
        n_x, n_y = normal_vector(theta)
        dGamma_nodes = dGamma(theta)
        f_nodes = f(theta)
        # Form the matrix and solve for sigma
        A = np.zeros((N, N))
        b = f_nodes
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij_x = x_nodes[i] - x_nodes[j]
                    r_ij_y = y_nodes[i] - y_nodes[j]
                    r_ij_sq = r_ij_x**2 + r_ij_y**2
                    K_ij = (r_ij_x * n_x[i] + r_ij_y * n_y[i]) / r_ij_sq
                    A[i, j] = K_ij * dGamma_nodes[j]
            # Diagonal entries (principal value)
            A[i, i] = -np.pi
        sigma = np.linalg.solve(A, -b)
        # Compute potential at test points
        u_num = compute_potential(x_test, y_test, x_nodes, y_nodes, sigma, dGamma_nodes)
        # Compute error
        error = np.abs(u_num - u_ref)
        max_error = np.max(error)
        errors.append(max_error)
        print(f"N = {N}, Max Error = {max_error:.2e}")

    # Plotting the errors
    plt.figure(figsize=(8, 6))
    plt.loglog(N_values, errors, 'o-', label='Max Error')
    plt.xlabel('Number of Nyström Points N')
    plt.ylabel('Maximum Error')
    plt.title('Convergence of Nyström Method on the Ellipse')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
