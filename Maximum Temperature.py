import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

def Heat(N_xi, N_eta, T_hot, T_cool_inlet, h, geom, U):

    L  = geom[0]
    L0 = geom[1]
    L1 = geom[2]
    L2 = geom[3]

    xi, eta = np.meshgrid(np.linspace(0,1,N_xi), np.linspace(0,1,N_eta), indexing='ij')

    x = L * xi
    y = L0 + L1 * eta + (L2 - L1) * xi * eta

    TotalNodes = N_xi * N_eta + N_xi
    Node = np.arange(0, N_xi * N_eta).reshape((N_xi,N_eta), order='F')

    A = lil_matrix((TotalNodes, TotalNodes))
    RHS = np.zeros(TotalNodes)

    for i in range(1, N_xi - 1):
        for j in range(1, N_eta - 1):
            CN = Node[i, j] 

            # Compute grid spacings
            dx_e = x[i+1, j] - x[i, j]
            dx_w = x[i, j] - x[i-1, j]
            dy_n = y[i, j+1] - y[i, j]
            dy_s = y[i, j] - y[i, j-1]

            # Coefficients
            ae = 2.0 / (dx_e * (dx_e + dx_w))
            aw = 2.0 / (dx_w * (dx_e + dx_w))
            an = 2.0 / (dy_n * (dy_n + dy_s))
            as_ = 2.0 / (dy_s * (dy_n + dy_s))
            ap = ae + aw + an + as_

            # Stamp entries in global matrix
            A[CN, Node[i+1, j]] = -ae
            A[CN, Node[i-1, j]] = -aw
            A[CN, Node[i, j+1]] = -an
            A[CN, Node[i, j-1]] = -as_
            A[CN, CN] = ap
            RHS[CN] = 0.0

    # Left Boundary
    for j in range(N_eta):
        CN = Node[0, j]
        dx_e = x[1, j] - x[0, j]
        A[CN, Node[0, j]] = -1.0 / dx_e
        A[CN, Node[1, j]] = 1.0 / dx_e
        RHS[CN] = 0.0

    # Bottom Boundary
    for i in range(N_xi):
        CN = Node[i, 0]
        FN = N_xi * N_eta + i  
        dy_n = y[i, 1] - y[i, 0]
        A[CN, CN] = (1.0 / dy_n) + h
        A[CN, Node[i, 1]] = -1.0 / dy_n
        A[CN, FN] = -h 
        RHS[CN] = 0.0

    # Top Boundary
    for i in range(N_xi):
        CN = Node[i, N_eta - 1]
        dy_s = y[i, N_eta - 1] - y[i, N_eta - 2]
        A[CN, CN] = (1.0 / dy_s) + h
        A[CN, Node[i, N_eta - 2]] = -1.0 / dy_s
        RHS[CN] = h * T_hot

    # Right Boundary
    for j in range(1, N_eta - 1):
        CN = Node[N_xi - 1, j]
        dx_w = x[N_xi - 1, j] - x[N_xi - 2, j]
        A[CN, CN] = (1.0 / dx_w) + h
        A[CN, Node[N_xi - 2, j]] = -1.0 / dx_w
        RHS[CN] = h * T_hot

    # Inlet Condition
    FN_0 = N_xi * N_eta
    A[FN_0, FN_0] = 1.0
    RHS[FN_0] = T_cool_inlet

    for i in range(1, N_xi):
        FN_i = N_xi * N_eta + i
        FN_im1 = N_xi * N_eta + i - 1
        CN = Node[i, 0] 

        dx = x[i, 0] - x[i - 1, 0]

        A[FN_i, FN_i] = (U / dx) + 2 * h
        A[FN_i, FN_im1] = -U / dx
        A[FN_i, CN] = -2 * h 
        RHS[FN_i] = 0.0

    T_total = spsolve(A.tocsr(), RHS)

    T_solid_flat = T_total[:N_xi * N_eta]
    T_cool = T_total[N_xi * N_eta:]
    T_solid = T_solid_flat.reshape((N_xi, N_eta), order='F')

    return x, y, T_solid, T_cool

#==================================================================================
# Driver Code
#================================================================================== 
if __name__ == "__main__":

    L = 1.0
    L1 = 0.25
    L0 = 0.025
    L2 = 0.05

    T_hot = 1.4
    T_cool_inlet = 0.6
    h = 5.0

    N_xi = 81
    N_eta = 41

    U_values = np.arange(2.5, 20.1, 2.5)
    max_temperatures = []

    for U in U_values:
        x, y, T_solid, T_cool = Heat(N_xi, N_eta, T_hot, T_cool_inlet, h, [L, L0, L1, L2], U)

        # Find maximum temperature in the blade
        max_T = np.max(T_solid)
        max_temperatures.append(max_T)

        print(f"U = {U:.1f}, Maximum temperature in the blade: {max_T:.4f}")

    # Plot maximum temperature vs U
    plt.figure(figsize=(8, 6))
    plt.plot(U_values, max_temperatures, marker='o')
    plt.xlabel('Cooling Flow Velocity U')
    plt.ylabel('Maximum Temperature in Blade')
    plt.title('Maximum Blade Temperature vs Cooling Flow Velocity')
    plt.grid(True)
    plt.show()
