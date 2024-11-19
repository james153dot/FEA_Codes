import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 25  
h = 1.0 / (N - 1)
tolerance = 1e-6
max_iterations = 10000
B = 4  
source_block_numbers = [1, 7, 14, 16]  

def set_source_blocks(f, B, source_block_numbers):
    """
    Set f = 1 in the specified source blocks.

    Parameters:
        f: 2D numpy array representing the source term.
        B: Number of interior blocks in one dimension (4 for a 4x4 block division).
        source_block_numbers: List of interior block numbers (1 to 16).
    """
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
            laplacian = (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - 4 * phi[i, j]) / h**2
            residual[i, j] = f[i, j] - (-laplacian)
    return residual

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

def G_S_method(phi, f, omega, h, tolerance, max_iterations):
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
        # Divergence
        if iteration > 1 and residual_norms[-1] > residual_norms[-2]:
            print(f'G-S method diverging at iteration {iteration} with ω={omega}.')
            break
    else:
        print(f'G-S method did not converge within the maximum number of iterations for ω={omega}.')
    return phi, residual_norms


if __name__ == "__main__":
    f = np.zeros((N, N))
    set_source_blocks(f, B, source_block_numbers)

    phi_initial = np.zeros((N, N))

    omega_values_jacobi = [0.8, 1.0, 1.2]
    omega_values_gs = [1.0, 1.2, 1.4, 1.6, 1.7, 1.8]

    # Jacobi method
    plt.figure()
    for omega in omega_values_jacobi:
        phi = phi_initial.copy()
        phi_jacobi, residuals_jacobi = jacobi_method(phi, f, omega, h, tolerance, max_iterations)
        plt.plot(np.log10(residuals_jacobi), label=f'Jacobi ω={omega}')
    plt.xlabel('Iteration')
    plt.ylabel('Log10 Residual Norm')
    plt.title('Jacobi Method Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

    # G-S method
    plt.figure()
    for omega in omega_values_gs:
        phi = phi_initial.copy()
        phi_gs, residuals_gs = G_S_method(phi, f, omega, h, tolerance, max_iterations)
        plt.plot(np.log10(residuals_gs), label=f'GS ω={omega}')
    plt.xlabel('Iteration')
    plt.ylabel('Log10 Residual Norm')
    plt.title('G-S Method Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

