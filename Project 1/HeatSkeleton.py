import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

def Heat(N_xi, N_eta, T_hot, T_cool, h, geom):

    L  = geom[0]
    L0 = geom[1]
    L1 = geom[2]
    L2 = geom[3]

    xi, eta = np.meshgrid(np.linspace(0,1,N_xi), np.linspace(0,1,N_eta), indexing='ij')

    x = L * xi
    y = L0 + L1 * eta + (L2 - L1) * xi * eta

    TotalNodes = N_xi * N_eta
    Node = np.arange(0,TotalNodes).reshape((N_xi,N_eta), order='F')

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
        dy_n = y[i, 1] - y[i, 0]
        A[CN, Node[i, 0]] = (1.0 / dy_n) + h
        A[CN, Node[i, 1]] = -1.0 / dy_n
        RHS[CN] = h * T_cool[i]

    # Top Boundary
    for i in range(N_xi):
        CN = Node[i, N_eta - 1]
        dy_s = y[i, N_eta - 1] - y[i, N_eta - 2]
        A[CN, Node[i, N_eta - 1]] = (1.0 / dy_s) + h
        A[CN, Node[i, N_eta - 2]] = -1.0 / dy_s
        RHS[CN] = h * T_hot

    # Right Boundary
    for j in range(1, N_eta - 1):
        CN = Node[N_xi - 1, j]
        dx_w = x[N_xi - 1, j] - x[N_xi - 2, j]
        A[CN, Node[N_xi - 1, j]] = (1.0 / dx_w) + h
        A[CN, Node[N_xi - 2, j]] = -1.0 / dx_w
        RHS[CN] = h * T_hot

    T_flat = spsolve(A.tocsr(), RHS)
    T = T_flat.reshape((N_xi, N_eta), order='F')

    return x, y, T

#==================================================================================
# Driver for Problem 1 - Project 1
#================================================================================== 
if __name__ == "__main__":

    L = 1.0
    L1 = 0.25
    L0 = 0.025
    L2 = 0.05

    T_hot = 1.4
    T_cool_inlet = 0.6
    h = 5.0

    N_xi = 51
    N_eta = 31

    U = 5.0

    T_cool = T_cool_inlet * np.ones(N_xi)

    x, y, T = Heat(N_xi, N_eta, T_hot, T_cool, h, [L, L0, L1, L2])

    # Grid
    plt.figure()
    for i in range(N_xi):
        plt.plot(x[i, :], y[i, :], 'k')
    for j in range(N_eta):
        plt.plot(x[:, j], y[:, j], 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grid')
    plt.axis('equal')
    plt.show()

    # Temperature Contours
    plt.figure()
    plt.contourf(x, y, T, levels=50, cmap='hot')
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Temperature Distribution in Turbine Blade')
    plt.axis('equal')
    plt.show()
