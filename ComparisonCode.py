import numpy as np
import pandas as pd

# Parameters
N = 25  
h = 1.0 / (N - 1)
tolerance = 1e-6
max_iterations = 10000
B = 4 
source_block_numbers = [1, 7, 14, 16]  

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
            laplacian = (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - 4 * phi[i, j]) / h**2
            residual[i, j] = f[i, j] - (-laplacian)
    return residual

# G-S method with relaxation
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

def extract_interior_boundary_potentials(phi):
    N = phi.shape[0]
    pts = np.arange(1, N - 1)
    left = phi[1:-1, 1]
    right = phi[1:-1, N - 2]
    bottom = phi[1, 1:-1]
    top = phi[N - 2, 1:-1]
    data = {
        'Pt': pts,
        'Left': left.round(4),
        'Right': right.round(4),
        'Bottom': bottom.round(4),
        'Top': top.round(4)
    }
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    f = np.zeros((N, N))
    set_source_blocks(f, B, source_block_numbers)

    phi_initial = np.zeros((N, N))

    # Run G-S method with optimal omega
    omega_opt = 1.6
    phi_gs, residuals_gs = g_s_method(phi_initial.copy(), f, omega_opt, h, tolerance, max_iterations)

    boundary_potentials_df = extract_interior_boundary_potentials(phi_gs)
    print(boundary_potentials_df.to_string(index=False))
