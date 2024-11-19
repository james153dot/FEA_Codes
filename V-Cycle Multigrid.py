import numpy as np
import matplotlib.pyplot as plt

def relaxation(phi, f, omega, h, iterations):
    N = phi.shape[0]
    for _ in range(iterations):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                phi_old = phi[i, j]
                phi_new = 0.25 * (phi[i+1, j] + phi[i-1, j] +
                                  phi[i, j+1] + phi[i, j-1] + h**2 * f[i, j])
                phi[i, j] = phi_old + omega * (phi_new - phi_old)
    return phi

def restrict(fine):
    N_coarse = (fine.shape[0] - 1) // 2
    coarse = np.zeros((N_coarse + 1, N_coarse + 1))
    for I in range(1, N_coarse):
        for J in range(1, N_coarse):
            i = 2 * I
            j = 2 * J
            coarse[I, J] = (1/16) * (
                4 * fine[i, j] +
                2 * (fine[i+1, j] + fine[i-1, j] + fine[i, j+1] + fine[i, j-1]) +
                (fine[i+1, j+1] + fine[i+1, j-1] + fine[i-1, j+1] + fine[i-1, j-1])
            )
    return coarse

def prolong(coarse):
    N_fine = 2 * (coarse.shape[0] - 1)
    fine = np.zeros((N_fine + 1, N_fine + 1))
    for I in range(coarse.shape[0]):
        for J in range(coarse.shape[1]):
            i = 2 * I
            j = 2 * J
            fine[i, j] = coarse[I, J]
    # Interpolate in i-direction
    for i in range(1, N_fine, 2):
        for j in range(0, N_fine + 1, 2):
            fine[i, j] = 0.5 * (fine[i - 1, j] + fine[i + 1, j])
    # Interpolate in j-direction
    for i in range(0, N_fine + 1):
        for j in range(1, N_fine, 2):
            fine[i, j] = 0.5 * (fine[i, j - 1] + fine[i, j + 1])
    # Interpolate in both directions
    for i in range(1, N_fine, 2):
        for j in range(1, N_fine, 2):
            fine[i, j] = 0.25 * (fine[i - 1, j - 1] + fine[i - 1, j + 1] +
                                 fine[i + 1, j - 1] + fine[i + 1, j + 1])
    return fine

# Residual computation
def compute_residual(phi, f, h):
    N = phi.shape[0]
    residual = np.zeros_like(phi)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            laplacian = (phi[i+1, j] + phi[i-1, j] +
                         phi[i, j+1] + phi[i, j-1] - 4 * phi[i, j]) / h**2
            residual[i, j] = f[i, j] + laplacian
    return residual

# Multilevel V-cycle
def v_cycle(levels, phi_levels, f_levels, omega, h_levels, nu1, nu2, level=0):
    phi_levels[level] = relaxation(phi_levels[level], f_levels[level], omega, h_levels[level], nu1)
    residual = compute_residual(phi_levels[level], f_levels[level], h_levels[level])
    if level == len(levels) - 1:
        phi_levels[level] = relaxation(phi_levels[level], f_levels[level], omega, h_levels[level], 10)
    else:
        residual_coarse = restrict(residual)
        # Initialize error on coarse grid
        e_coarse = np.zeros_like(residual_coarse)
        phi_levels[level + 1] = e_coarse
        f_levels[level + 1] = residual_coarse
        v_cycle(levels, phi_levels, f_levels, omega, h_levels, nu1, nu2, level + 1)
        e_fine = prolong(phi_levels[level + 1])
        phi_levels[level] += e_fine
    phi_levels[level] = relaxation(phi_levels[level], f_levels[level], omega, h_levels[level], nu2)
    return phi_levels[level]

# Multigrid solver
def multigrid_solver(N, num_levels, omega, tolerance, max_cycles):
    levels = [N // (2 ** i) for i in range(num_levels)]
    h_levels = [1.0 / (n - 1) for n in levels]
    phi_levels = []
    f_levels = []
    for n in levels:
        phi_levels.append(np.zeros((n, n)))
        f_levels.append(np.zeros((n, n)))
    f_levels[0][(levels[0]//4):(3*levels[0]//4), (levels[0]//4):(3*levels[0]//4)] = 1
    residual_norms = []
    for cycle in range(max_cycles):
        phi_levels[0] = v_cycle(levels, phi_levels, f_levels, omega, h_levels, nu1=2, nu2=2)
        residual = compute_residual(phi_levels[0], f_levels[0], h_levels[0])
        residual_norm = np.linalg.norm(residual)
        residual_norms.append(residual_norm)
        if residual_norm < tolerance:
            print(f'Multigrid solver converged in {cycle+1} cycles.')
            break
    else:
        print('Multigrid solver did not converge within the maximum number of cycles.')
    return phi_levels[0], residual_norms

if __name__ == "__main__":
    N = 129
    omega = 0.8
    tolerance = 1e-6
    max_cycles = 100

    num_levels_list = [2, 3, 4]
    residuals_dict = {}

    for num_levels in num_levels_list:
        print(f'\nRunning multigrid solver with {num_levels} grid levels:')
        phi, residuals = multigrid_solver(N, num_levels, omega, tolerance, max_cycles)
        residuals_dict[num_levels] = residuals
        plt.plot(np.log10(residuals), label=f'{num_levels} levels')

    plt.xlabel('Cycle')
    plt.ylabel('Log10 Residual Norm')
    plt.title('Multigrid Convergence for Different Grid Levels')
    plt.legend()
    plt.grid(True)
    plt.show()
