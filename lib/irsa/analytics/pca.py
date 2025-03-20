from mlgrad.pca import find_loc_and_pc, find_robust_loc_and_pc
import numpy as np
import matplotlib.pyplot as plt
import mlgrad.inventory as inventory

def dist(S, X, c=0):
    return np.sqrt([((S@x)@x) for x in (X-c)])

def pca_compare_2d(X1, X2, label1, label2):
    c1, A1, L1 = find_loc_and_pc(X1, 2)
    U1 = (X1 - c1) @ A1.T
    S1 = np.linalg.inv(U1.T @ U1)
    S1 /= np.sqrt(np.linalg.det(S1))

    U2 = (X2 - c1) @ A1.T
    c2, A2, L2 = find_loc_and_pc(U2, 2)
    UU2 = (U2 - c2) @ A2.T
    S2 = np.linalg.inv(UU2.T @ UU2)
    S2 /= np.sqrt(np.linalg.det(S2))

    x_min = min(U1[:,0].min(), U2[:,0].min())
    x_max = max(U1[:,0].max(), U2[:,0].max())
    y_min = min(U1[:,1].min(), U2[:,1].min())
    y_max = max(U1[:,1].max(), U2[:,1].max())

    dy = y_max - y_min
    dx = x_max - x_min
    cc = dy / dx

    XX, YY = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100))    

    XY = np.c_[XX.ravel(), YY.ravel()]
    ZZ1 = dist(S1, XY)
    ZZ1 = ZZ1.reshape(XX.shape)
    ZZ2 = dist(S2, XY - c2)
    ZZ2 = ZZ2.reshape(XX.shape)

    ZZ = ZZ1 - ZZ2

    levels = np.arange(-5, 5, 0.5)

    # D1 = dist(S1, U1)
    # D2 = dist(S2, U2)
    # Z1 = abs(inventory.zscore(D1))
    # Z2 = abs(inventory.zscore(D2))

    if cc < 1:
        plt.figure(figsize=(12, 12*cc))
    else:
        plt.figure(figsize=(12/cc, 12))

    plt.contourf(XX, YY, ZZ, levels=levels, alpha=0.5)
    
    plt.scatter(U1[:,0], U1[:,1], c="r", s=36, edgecolors="k", linewidth=0.5, label=label1)
    plt.scatter(U2[:,0], U2[:,1], c="b", s=36, edgecolors="k", linewidth=0.5, label=label2)
    plt.scatter([0], [0], s=144, c="r", edgecolors="k", linewidth=1.0)
    plt.scatter([c2[0]], [c2[1]], s=144, c="b", edgecolors="k", linewidth=1.0)
    
    plt.contour(XX, YY, ZZ1, levels=[1.0], colors="r", alpha=0.5)
    plt.contour(XX, YY, ZZ2, levels=[1.0], colors="b", alpha=0.5)
    
    ct1 = plt.contour(XX, YY, ZZ, levels=levels, alpha=0.5)
    plt.clabel(ct1, colors='k')
    plt.contour(XX, YY, ZZ, levels=[0.0], colors='k', linewidths=1.5)
    
    # ct2 = plt.contour(XX, YY, ZZ2, colors="b", alpha=0.5)
    # plt.clabel(ct2)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def robust_pca_compare_2d(X1, X2, label1, label2):
    from mlgrad.af import averaging_function
    
    wma = averaging_function("WZ", kwds={"alpha":3.0})
    
    c1, A1, L1 = find_robust_loc_and_pc(X1, wma, 2)
    U1 = (X1 - c1) @ A1.T
    S1 = np.linalg.inv(U1.T @ U1)
    S1 /= np.sqrt(np.linalg.det(S1))

    U2 = (X2 - c1) @ A1.T
    c2, A2, L2 = find_robust_loc_and_pc(U2, wma, 2)
    UU2 = (U2 - c2) @ A2.T
    S2 = np.linalg.inv(UU2.T @ UU2)
    S2 /= np.sqrt(np.linalg.det(S2))

    x_min = min(U1[:,0].min(), U2[:,0].min())
    x_max = max(U1[:,0].max(), U2[:,0].max())
    y_min = min(U1[:,1].min(), U2[:,1].min())
    y_max = max(U1[:,1].max(), U2[:,1].max())

    dy = y_max - y_min
    dx = x_max - x_min
    cc = dy / dx    

    XX, YY = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100))    

    XY = np.c_[XX.ravel(), YY.ravel()]
    ZZ1 = dist(S1, XY)
    ZZ1 = ZZ1.reshape(XX.shape)
    ZZ2 = dist(S2, XY - c2)
    ZZ2 = ZZ2.reshape(XX.shape)

    # D1 = dist(S1, U1)
    # D2 = dist(S2, U2)
    # Z1 = abs(inventory.zscore(D1))
    # Z2 = abs(inventory.zscore(D2))

    ZZ = ZZ1 - ZZ2 
    levels = np.arange(-5, 5, 0.5)

    if cc < 1:
        plt.figure(figsize=(12, 12*cc))
    else:
        plt.figure(figsize=(12/cc, 12))

    plt.contourf(XX, YY, ZZ, levels=levels, alpha=0.5)

    plt.scatter(U1[:,0], U1[:,1], c="r", s=36, edgecolors="k", linewidth=0.5, label=label1)
    plt.scatter(U2[:,0], U2[:,1], c="b", s=36, edgecolors="k", linewidth=0.5, label=label2)
    plt.scatter([0], [0], s=144, c="r", edgecolors="k", linewidth=1.0)
    plt.scatter([c2[0]], [c2[1]], s=144, c="b", edgecolors="k", linewidth=1.0)

    plt.contour(XX, YY, ZZ1, levels=[1.0], colors="r", alpha=0.5)
    plt.contour(XX, YY, ZZ2, levels=[1.0], colors="b", alpha=0.5)

    ct1 = plt.contour(XX, YY, ZZ, levels=levels, alpha=0.5)
    plt.clabel(ct1, colors='k')
    plt.contour(XX, YY, ZZ, levels=[0.0], colors='k', linewidths=1.5)
    # ct2 = plt.contour(XX, YY, ZZ2, colors="b", alpha=0.5)
    # plt.clabel(ct2)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()
