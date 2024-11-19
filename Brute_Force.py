import numpy as np
import itertools

N = 25
h = 1.0 / (N - 1)
tolerance = 1e-6
max_cycles = 50
omega = 0.8

interior_blocks = [i for i in range(1, 17)]

# Given flux data
flux_data = {
    'Left': np.array([0.0000, 0.0064, 0.0127, 0.0191, 0.0253, 0.0314, 0.0372, 0.0426, 0.0473, 0.0513,
                      0.0542, 0.0559, 0.0564, 0.0557, 0.0538, 0.0507, 0.0467, 0.0419, 0.0365, 0.0307,
                      0.0247, 0.0186, 0.0124, 0.0062, 0.0000]),
    'Right': np.array([0.0000, 0.0071, 0.0143, 0.0216, 0.0290, 0.0365, 0.0441, 0.0517, 0.0593, 0.0668,
                       0.0738, 0.0800, 0.0844, 0.0864, 0.0854, 0.0811, 0.0740, 0.0650, 0.0550, 0.0449,
                       0.0351, 0.0258, 0.0169, 0.0084, 0.0000]),
    'Bottom': np.array([0.0000, 0.0064, 0.0128, 0.0191, 0.0254, 0.0316, 0.0376, 0.0432, 0.0481, 0.0524,
                        0.0556, 0.0578, 0.0587, 0.0584, 0.0569, 0.0541, 0.0503, 0.0456, 0.0402, 0.0342,
                        0.0278, 0.0210, 0.0141, 0.0071, 0.0000]),
    'Top': np.array([0.0000, 0.0062, 0.0124, 0.0186, 0.0246, 0.0306, 0.0363, 0.0415, 0.0463, 0.0502,
                     0.0533, 0.0555, 0.0566, 0.0567, 0.0559, 0.0541, 0.0514, 0.0477, 0.0430, 0.0374,
                     0.0310, 0.0239, 0.0162, 0.0082, 0.0000])
}

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
        f[i_start:i_end+1, j_start:j_end+1] = 1  # Including boundaries

def compute_residual(phi, f, h):
    N = phi.shape[0]
    residual = np.zeros_like(phi)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            laplacian = (phi[i+1, j] + phi[i-1, j] +
                         phi[i, j+1] + phi[i, j-1] - 4 * phi[i, j]) / h**2
            residual[i, j] = f[i, j] + laplacian
    return residual

def G_S_iteration(phi, f, omega, h, iterations):
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


def compute_normal_flux(phi, h):
    N = phi.shape[0]
    flux = {'Left': [], 'Right': [], 'Bottom': [], 'Top': []}
    # Left boundary (x=0)
    for j in range(N):
        flux_left = - (phi[1, j] - phi[0, j]) / h
        flux['Left'].append(flux_left)
    # Right boundary (x=1)
    for j in range(N):
        flux_right = - (phi[N-1, j] - phi[N-2, j]) / h
        flux['Right'].append(flux_right)
    # Bottom boundary (y=0)
    for i in range(N):
        flux_bottom = - (phi[i, 1] - phi[i, 0]) / h
        flux['Bottom'].append(flux_bottom)
    # Top boundary (y=1)
    for i in range(N):
        flux_top = - (phi[i, N-1] - phi[i, N-2]) / h
        flux['Top'].append(flux_top)
    return flux

def calculate_flux_error(computed_flux, given_flux):
    error = 0.0
    for key in given_flux.keys():
        error += np.sum((np.array(computed_flux[key]) - given_flux[key])**2)
    return error

def find_source_blocks():
    min_error = float('inf')
    best_combination = None
    combinations = itertools.combinations(interior_blocks, 4)
    total_combinations = 0
    for combo in combinations:
        total_combinations += 1
    print(f"Total combinations to evaluate: {total_combinations}")
    
    combinations = itertools.combinations(interior_blocks, 4)
    count = 0
    for source_blocks in combinations:
        count += 1
        f = np.zeros((N, N))
        set_source_blocks(f, B=4, source_block_numbers=source_blocks)
        phi_initial = np.zeros((N, N))
        
        phi, residuals = multigrid_solver(phi_initial, f, omega, h, tolerance, max_cycles,
                                          relaxation_method=G_S_iteration, nuc=5)
        # Compute normal flux
        computed_flux = compute_normal_flux(phi, h)
        # Calculate error
        error = calculate_flux_error(computed_flux, flux_data)
        # Update best combination
        if error < min_error:
            min_error = error
            best_combination = source_blocks
            print(f"New best combination: {best_combination} with error: {min_error}")
        if count % 100 == 0:
            print(f"Evaluated {count} combinations...")
    print(f"Best combination found: {best_combination} with minimal error: {min_error}")
    return best_combination

# Run the search
best_source_blocks = find_source_blocks()
