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

def gauss_seidel_iteration(phi, f, omega, h, iterations):
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
    # Copy coarse grid points
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

# Modified Multigrid V-cycle with νc parameter
def multigrid_v_cycle(phi, f, omega, h, nu1, nu2, nuc, relaxation_method):
    N = phi.shape[0] - 1
    phi = relaxation_method(phi, f, omega, h, nu1)
    residual_fine = compute_residual(phi, f, h)
    residual_coarse = restrict(residual_fine)
    # Initialize error on coarse grid
    N_coarse = (N) // 2
    e_coarse = np.zeros((N_coarse + 1, N_coarse + 1))
    h_coarse = 2 * h
    f_coarse = residual_coarse
    for _ in range(nuc):
        e_coarse = relaxation_method(e_coarse, f_coarse, omega, h_coarse, 1)
    e_fine = prolong(e_coarse)
    phi += e_fine
    phi = relaxation_method(phi, f, omega, h, nu2)
    return phi

# Modified Multigrid solver with νc parameter
def multigrid_solver(phi_initial, f, omega, h, tolerance, max_cycles, relaxation_method, nuc):
    phi = phi_initial.copy()
    residual_norms = []
    for cycle in range(max_cycles):
        phi = multigrid_v_cycle(phi, f, omega, h, nu1=1, nu2=1, nuc=nuc, relaxation_method=relaxation_method)
        residual = compute_residual(phi, f, h)
        error_norm = np.linalg.norm(residual)
        residual_norms.append(error_norm)
        if error_norm < tolerance:
            print(f'Multigrid solver converged in {cycle+1} cycles with ω={omega}, νc={nuc}.')
            break
    else:
        print(f'Multigrid solver did not converge within the maximum number of cycles for ω={omega}, νc={nuc}.')
    return phi, residual_norms

if __name__ == "__main__":
    # Parameters
    N = 65  
    h = 1.0 / (N - 1)
    tolerance = 1e-6
    max_cycles = 100
    B = 4 
    source_block_numbers = [1, 7, 14, 16] 

    f = np.zeros((N, N))
    set_source_blocks(f, B, source_block_numbers)

    phi_initial = np.zeros((N, N))

    omega = 0.8

    nuc_values = [2, 4, 20]

    residuals_dict = {}
    for nuc in nuc_values:
        print(f'\nTesting with νc = {nuc}')
        phi = phi_initial.copy()
        phi_mg_gs, residuals_mg_gs = multigrid_solver(
            phi, f, omega, h, tolerance, max_cycles, relaxation_method=gauss_seidel_iteration, nuc=nuc
        )
        residuals_dict[nuc] = residuals_mg_gs

    plt.figure(figsize=(10, 6))
    for nuc in nuc_values:
        plt.plot(np.log10(residuals_dict[nuc]), label=f'νc = {nuc}')
    plt.xlabel('Cycle')
    plt.ylabel('Log10 Residual Norm')
    plt.title(f'Multigrid Convergence with ν1=ν2=1, ω={omega}')
    plt.legend()
    plt.grid(True)
    plt.show()
