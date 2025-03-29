from mlgrad.pca import find_loc_and_pc, find_loc_and_pc_ss, find_robust_loc_and_pc
import numpy as np
import matplotlib.pyplot as plt
import mlgrad.inventory as inventory
from scipy.stats import chi2

def dist(S, X, c=0):
    return np.sqrt([((S@x)@x) for x in (X-c)])

def dist2(S, X, c=0):
    return np.array([((S@x)@x) for x in (X-c)])

def scale_min(x):
    if x < 0:
        return 1.1*x
    else:
        return 0.9*x

def scale_max(x):
    if x < 0:
        return 0.9*x
    else:
        return 1.1*x

def find_loc_and_pc_2d(X, *, n=2, alpha=0.975):
    c, A, _ = find_loc_and_pc_ss(X, n)
    U = (X - c) @ A.T
    S = np.linalg.inv(U.T @ U)
    S /= np.sqrt(np.linalg.det(S))

    D = dist2(S, U)
    DD = np.fromiter(chi2.cdf(D, 2), "d")
    print("outliers:", len(DD[DD > alpha]))
    d_alpha = np.sqrt(max(DD[DD <= alpha]))
    X2 = np.ascontiguousarray(X[DD <= alpha])

    c2, A2, _ = find_loc_and_pc_ss(X2, n)
    return c2, A2, d_alpha

def project_on_pc(X1, X2, alpha=0.99):
    c1, A1 = find_loc_and_pc_2d(X1, alpha=alpha)
    U1 = (X1 - c1) @ A1.T
    U2 = (X2 - c1) @ A1.T
    return U1, U2

def pca_compare_symmetrical_2d(X1, X2, label1, label2, alpha=0.9999):

    c1, A1, d1_alpha = find_loc_and_pc_2d(X1, alpha=alpha)
    U1 = (X1 - c1) @ A1.T
    S1 = np.linalg.inv(U1.T @ U1)
    S1 /= np.sqrt(np.linalg.det(S1))

    c2, A2, d2_alpha = find_loc_and_pc_2d(X2, alpha=alpha)
    U2 = (X2 - c2) @ A2.T
    S2 = np.linalg.inv(U2.T @ U2)
    S2 /= np.sqrt(np.linalg.det(S2))

    x_min = scale_min(min(U1[:,0].min(), U2[:,0].min()))
    x_max = scale_max(max(U1[:,0].max(), U2[:,0].max()))
    y_min = scale_min(min(U1[:,1].min(), U2[:,1].min()))
    y_max = scale_max(max(U1[:,1].max(), U2[:,1].max()))

    dy = y_max - y_min
    dx = x_max - x_min
    # cc = dy / dx

    UX, UY = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100))    

    UXY = np.c_[UX.ravel(), UY.ravel()]
    ZZ1 = dist(S1, UXY)
    ZZ1 = ZZ1.reshape(UX.shape)
    ZZ2 = dist(S2, UXY)
    ZZ2 = ZZ2.reshape(UX.shape)

    ZZ = ZZ1 - ZZ2

    Z11 = dist(S1, U1)
    Z12 = dist(S1, U2)
    Z21 = dist(S2, U1)
    Z22 = dist(S2, U2)

    k1 = 0
    k2 = 0
    
    for z1, z2 in zip(Z11, Z21):
        if z1 >= z2:
            k1 += 1
    for z1, z2 in zip(Z12, Z22):
        if z2 >= z1:
            k2 += 1
    print(f"1: err={k1/len(U1):.2f}")
    print(f"2: err={k2/len(U2):.2f}")

    levels = np.arange(-5, 5, 0.5)

    plt.figure(figsize=(12, 3.5))

    plt.contourf(UX, UY, ZZ, levels=levels, alpha=0.5)
    
    plt.scatter(U1[:,0], U1[:,1], c="r", s=36, edgecolors="k", linewidth=0.5, label=label1)
    plt.scatter(U2[:,0], U2[:,1], c="b", s=36, edgecolors="k", linewidth=0.5, label=label2)
    plt.scatter([0], [0], s=144, c="r", edgecolors="k", linewidth=1.0)
    plt.scatter([c2[0]], [c2[1]], s=144, c="b", edgecolors="k", linewidth=1.0)
    
    plt.contour(UX, UY, ZZ1, levels=[d1_alpha], colors="r", alpha=0.5)
    plt.contour(UX, UY, ZZ2, levels=[d2_alpha], colors="b", alpha=0.5)
    
    ct1 = plt.contour(UX, UY, ZZ, levels=levels, linewidths=0.5, alpha=0.5)
    plt.clabel(ct1, colors='k')
    plt.contour(UX, UY, ZZ, levels=[0.0], colors='k', linewidths=1.5)
    
    # ct2 = plt.contour(UX, UY, ZZ2, colors="b", alpha=0.5)
    # plt.clabel(ct2)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()


def pca_compare_2d(X1, X2, label1, label2, alpha=0.9999):

    c1, A1, d1_alpha = find_loc_and_pc_2d(X1, alpha=alpha)
    U1 = (X1 - c1) @ A1.T
    S1 = np.linalg.inv(U1.T @ U1)
    S1 /= np.sqrt(np.linalg.det(S1))

    U2 = (X2 - c1) @ A1.T
    c2, A2, d2_alpha = find_loc_and_pc_2d(U2, alpha=alpha)
    UU2 = (U2 - c2) @ A2.T
    S2 = np.linalg.inv(UU2.T @ UU2)
    S2 /= np.sqrt(np.linalg.det(S2))

    x_min = scale_min(min(U1[:,0].min(), U2[:,0].min()))
    x_max = scale_max(max(U1[:,0].max(), U2[:,0].max()))
    y_min = scale_min(min(U1[:,1].min(), U2[:,1].min()))
    y_max = scale_max(max(U1[:,1].max(), U2[:,1].max()))

    dy = y_max - y_min
    dx = x_max - x_min
    # cc = dy / dx

    UX, UY = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100))    

    UXY = np.c_[UX.ravel(), UY.ravel()]
    ZZ1 = dist(S1, UXY)
    ZZ1 = ZZ1.reshape(UX.shape)
    ZZ2 = dist(S2, UXY - c2)
    ZZ2 = ZZ2.reshape(UX.shape)

    ZZ = ZZ1 - ZZ2

    Z11 = dist(S1, U1)
    Z12 = dist(S1, U2)
    Z21 = dist(S2, U1 - c2)
    Z22 = dist(S2, U2 - c2)

    k1 = 0
    k2 = 0
    
    for z1, z2 in zip(Z11, Z21):
        if z1 >= z2:
            k1 += 1
    for z1, z2 in zip(Z12, Z22):
        if z2 >= z1:
            k2 += 1
    print(f"1: err={k1/len(U1):.2f}")
    print(f"2: err={k2/len(U2):.2f}")

    levels = np.arange(-5, 5, 0.5)

    plt.figure(figsize=(12, 3.5))

    plt.contourf(UX, UY, ZZ, levels=levels, alpha=0.5)
    
    plt.scatter(U1[:,0], U1[:,1], c="r", s=36, edgecolors="k", linewidth=0.5, label=label1)
    plt.scatter(U2[:,0], U2[:,1], c="b", s=36, edgecolors="k", linewidth=0.5, label=label2)
    plt.scatter([0], [0], s=144, c="r", edgecolors="k", linewidth=1.0)
    plt.scatter([c2[0]], [c2[1]], s=144, c="b", edgecolors="k", linewidth=1.0)
    
    plt.contour(UX, UY, ZZ1, levels=[d1_alpha], colors="r", alpha=0.5)
    plt.contour(UX, UY, ZZ2, levels=[d2_alpha], colors="b", alpha=0.5)
    
    ct1 = plt.contour(UX, UY, ZZ, levels=levels, linewidths=0.5, alpha=0.5)
    plt.clabel(ct1, colors='k')
    plt.contour(UX, UY, ZZ, levels=[0.0], colors='k', linewidths=1.5)
    
    # ct2 = plt.contour(UX, UY, ZZ2, colors="b", alpha=0.5)
    # plt.clabel(ct2)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def robust_pca_compare_2d(X1, X2, label1, label2, *, kind="WMZ"):
    from mlgrad.af import averaging_function
    
    if kind == "WZ":
        wma = averaging_function("WZ", kwds={"alpha":3.0})
    elif kind == "WMZ":
        wma = averaging_function("WMZ", kwds={"alpha":3.0})
    
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

    plt.contour(XX, YY, ZZ1, levels=[1.0], colors="r", linewidths=0.5, alpha=0.5)
    plt.contour(XX, YY, ZZ2, levels=[1.0], colors="b", linewidths=0.5, alpha=0.5)

    ct1 = plt.contour(XX, YY, ZZ, levels=levels, alpha=0.5)
    plt.clabel(ct1, colors='k')
    plt.contour(XX, YY, ZZ, levels=[0.0], colors='k', linewidths=1.5)
    # ct2 = plt.contour(XX, YY, ZZ2, colors="b", alpha=0.5)
    # plt.clabel(ct2)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()
