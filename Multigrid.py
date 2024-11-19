import numpy as np
import matplotlib.pyplot as plt

def set_source_blocks(f, B, source_block_numbers):
    N = f.shape[0]
    block_size = (N - 2) // B
    block_indices = {
        1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (0, 3),
        5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3),
        9: (2, 0),10: (2, 1),11: (2, 2),12: (2, 3),
        13: (3, 0),14: (3, 1),15: (3, 2),16: (3, 3)
    }
    for block_num in source_block_numbers:
        row_block, col_block = block_indices[block_num]
        i_start = 1 + row_block * block_size
        i_end = i_start + block_size
        j_start = 1 + col_block * block_size
        j_end = j_start + block_size
        f[i_start:i_end+1, j_start:j_end+1] = 1  

def compute_residual(phi, f, h):
    N = phi.shape[0]
    residual = np.zeros_like(phi)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            laplacian = (phi[i+1, j] + phi[i-1, j] +
                         phi[i, j+1] + phi[i, j-1] - 4 * phi[i, j]) / h**2
            residual[i, j] = f[i, j] + laplacian
    return residual

def jacobi_iteration(phi, f, omega, h, iterations):
    N = phi.shape[0]
    for _ in range(iterations):
        phi_new = phi.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                phi_computed = 0.25 * (phi[i+1, j] + phi[i-1, j] +
                                       phi[i, j+1] + phi[i, j-1] + h**2 * f[i, j])
                phi_new[i, j] = phi[i, j] + omega * (phi_computed - phi[i, j])
        phi = phi_new
    return phi

def g_s_iteration(phi, f, omega, h, iterations):
    N = phi.shape[0]
    for _ in range(iterations):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                phi_computed = 0.25 * (phi[i+1, j] + phi[i-1, j] +
                                       phi[i, j+1] + phi[i, j-1] + h**2 * f[i, j])
                phi[i, j] = phi[i, j] + omega * (phi_computed - phi[i, j])
    return phi

def restrict(residual_fine):
    N_fine = residual_fine.shape[0] - 1
    N_coarse = N_fine // 2
    residual_coarse = np.zeros((N_coarse + 1, N_coarse + 1))
    for I in range(1, N_coarse):
        for J in range(1, N_coarse):
            i = 2 * I
            j = 2 * J
            residual_coarse[I, J] = (1/16) * (
                4 * residual_fine[i, j] +
                2 * (residual_fine[i+1, j] + residual_fine[i-1, j] +
                     residual_fine[i, j+1] + residual_fine[i, j-1]) +
                (residual_fine[i+1, j+1] + residual_fine[i-1, j+1] +
                 residual_fine[i+1, j-1] + residual_fine[i-1, j-1])
            )
    return residual_coarse

def prolong(error_coarse):
    N_coarse = error_coarse.shape[0] - 1
    N_fine = 2 * N_coarse
    error_fine = np.zeros((N_fine + 1, N_fine + 1))
    for I in range(N_coarse + 1):
        for J in range(N_coarse + 1):
            i = 2 * I
            j = 2 * J
            error_fine[i, j] = error_coarse[I, J]
    # Interpolate in i-direction
    for i in range(1, N_fine, 2):
        for j in range(0, N_fine + 1, 2):
            error_fine[i, j] = 0.5 * (error_fine[i - 1, j] + error_fine[i + 1, j])
    # Interpolate in j-direction
    for i in range(0, N_fine + 1):
        for j in range(1, N_fine, 2):
            error_fine[i, j] = 0.5 * (error_fine[i, j - 1] + error_fine[i, j + 1])
    # Interpolate in both directions
    for i in range(1, N_fine, 2):
        for j in range(1, N_fine, 2):
            error_fine[i, j] = 0.25 * (error_fine[i - 1, j - 1] + error_fine[i - 1, j + 1] +
                                       error_fine[i + 1, j - 1] + error_fine[i + 1, j + 1])
    return error_fine

# Multigrid V-cycle
def multigrid_v_cycle(phi, f, omega, h, nu1, nu2, relaxation_method):
    N = phi.shape[0] - 1
    phi = relaxation_method(phi, f, omega, h, nu1)
    residual_fine = compute_residual(phi, f, h)
    residual_coarse = restrict(residual_fine)
    # Initialize error on coarse grid
    N_coarse = (N) // 2
    e_coarse = np.zeros((N_coarse + 1, N_coarse + 1))
    h_coarse = 2 * h
    f_coarse = residual_coarse
    for _ in range(5):
        e_coarse = relaxation_method(e_coarse, f_coarse, omega, h_coarse, 1)
    e_fine = prolong(e_coarse)
    phi += e_fine
    phi = relaxation_method(phi, f, omega, h, nu2)
    return phi

# Multigrid solver
def multigrid_solver(phi_initial, f, omega, h, tolerance, max_cycles, relaxation_method):
    phi = phi_initial.copy()
    residual_norms = []
    for cycle in range(max_cycles):
        phi = multigrid_v_cycle(phi, f, omega, h, nu1=2, nu2=2, relaxation_method=relaxation_method)
        residual = compute_residual(phi, f, h)
        error_norm = np.linalg.norm(residual)
        residual_norms.append(error_norm)
        if error_norm < tolerance:
            print(f'Multigrid solver converged in {cycle+1} cycles with ω={omega}.')
            break
    else:
        print(f'Multigrid solver did not converge within the maximum number of cycles for ω={omega}.')
    return phi, residual_norms

# Jacobi method (single-grid)
def jacobi_method(phi, f, omega, h, tolerance, max_iterations):
    N = phi.shape[0]
    phi_new = phi.copy()
    residual_norms = []
    for iteration in range(max_iterations):
        phi_old = phi_new.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                phi_computed = 0.25 * (phi_old[i+1, j] + phi_old[i-1, j] +
                                       phi_old[i, j+1] + phi_old[i, j-1] + h**2 * f[i, j])
                phi_new[i, j] = phi_old[i, j] + omega * (phi_computed - phi_old[i, j])
        residual = compute_residual(phi_new, f, h)
        error_norm = np.linalg.norm(residual)
        residual_norms.append(error_norm)
        if error_norm < tolerance:
            print(f'Jacobi method converged in {iteration+1} iterations with ω={omega}.')
            break
    else:
        print(f'Jacobi method did not converge within the maximum number of iterations for ω={omega}.')
    return phi_new, residual_norms

# G-S method (single-grid)
def g_s_method(phi, f, omega, h, tolerance, max_iterations):
    N = phi.shape[0]
    residual_norms = []
    for iteration in range(max_iterations):
        phi_old = phi.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                phi_computed = 0.25 * (phi[i+1, j] + phi[i-1, j] +
                                       phi[i, j+1] + phi[i, j-1] + h**2 * f[i, j])
                phi[i, j] = phi[i, j] + omega * (phi_computed - phi[i, j])
        residual = compute_residual(phi, f, h)
        error_norm = np.linalg.norm(residual)
        residual_norms.append(error_norm)
        if error_norm < tolerance:
            print(f'G-S method converged in {iteration+1} iterations with ω={omega}.')
            break
    else:
        print(f'G-S method did not converge within the maximum number of iterations for ω={omega}.')
    return phi, residual_norms

if __name__ == "__main__":
    # Parameters
    N = 65
    h = 1.0 / (N - 1)
    tolerance = 1e-6
    max_iterations = 5000
    max_cycles = 100
    B = 4 
    source_block_numbers = [1, 7, 14, 16] 

    f = np.zeros((N, N))
    set_source_blocks(f, B, source_block_numbers)

    phi_initial = np.zeros((N, N))

    omega_values = [0.5, 0.8]

    for omega in omega_values:
        print(f'\nTesting with ω = {omega}')
        # Multigrid with Jacobi
        phi = phi_initial.copy()
        phi_mg_jacobi, residuals_mg_jacobi = multigrid_solver(
            phi, f, omega, h, tolerance, max_cycles, relaxation_method=jacobi_iteration
        )

        # Multigrid with G-S
        phi = phi_initial.copy()
        phi_mg_gs, residuals_mg_gs = multigrid_solver(
            phi, f, omega, h, tolerance, max_cycles, relaxation_method=g_s_iteration
        )

        # Single-grid Jacobi
        phi = phi_initial.copy()
        phi_jacobi, residuals_jacobi = jacobi_method(
            phi, f, omega, h, tolerance, max_iterations
        )

        # Single-grid G-S
        phi = phi_initial.copy()
        phi_gs, residuals_gs = g_s_method(
            phi, f, omega, h, tolerance, max_iterations
        )

        plt.figure(figsize=(10, 6))
        plt.plot(np.log10(residuals_mg_jacobi), label='Multigrid Jacobi')
        plt.plot(np.log10(residuals_mg_gs), label='Multigrid G-S')
        plt.plot(np.log10(residuals_jacobi), label='Single-grid Jacobi')
        plt.plot(np.log10(residuals_gs), label='Single-grid G-S')
        plt.xlabel('Iteration/Cycle')
        plt.ylabel('Log10 Residual Norm')
        plt.title(f'Convergence Comparison for ω = {omega}')
        plt.legend()
        plt.grid(True)
        plt.show()
