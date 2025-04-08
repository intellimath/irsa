from mlgrad.pca import find_loc_and_pc, find_loc_and_pc_ss, find_robust_loc_and_pc
import mlgrad.pca as pca
from mlgrad.af import averaging_function
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

def find_robust_loc_and_pc_2d(X, *, n=2, alpha=0.975, normalize_S=False):
    wma = averaging_function("WM", kwds={"alpha":alpha})
    c, A, _ = find_robust_loc_and_pc(X, wma, n)
    U = (X - c) @ A.T
    S = np.linalg.inv(U.T @ U)
    if normalize_S:
        S /= np.sqrt(np.linalg.det(S))

    D2 = dist2(S, U)
    d2_alpha = chi2.cdf(D2, alpha)
    # print(d2_alpha)
    print("outliers:", (d2_alpha > alpha).sum())
    X2 = np.ascontiguousarray(X[d2_alpha <= alpha])

    c2, A2, _ = find_loc_and_pc(X2, n)
    U2 = (X2 - c2) @ A2.T
    S2 = np.linalg.inv(U2.T @ U2)
    if normalize_S:
        S2 /= np.sqrt(np.linalg.det(S2))
    D = dist(S2, U2)    
    
    return c, A, max(D)

def find_loc_and_pc_2d(X, *, n=2, alpha=0.975, normalize_S=False):
    # c, A, _ = find_loc_and_pc_ss(X, n)
    c, A, _ = find_loc_and_pc(X, n)
    U = (X - c) @ A.T
    S = np.linalg.inv(U.T @ U)
    if normalize_S:
        S /= np.sqrt(np.linalg.det(S))

    D2 = dist2(S, U)
    d2_alpha = chi2.cdf(D2, alpha)
    # print(d2_alpha)
    print("outliers:", (d2_alpha > alpha).sum())
    X2 = np.ascontiguousarray(X[d2_alpha <= alpha])

    c2, A2, _ = find_loc_and_pc(X2, n)
    U2 = (X2 - c2) @ A2.T
    S2 = np.linalg.inv(U2.T @ U2)
    if normalize_S:
        S2 /= np.sqrt(np.linalg.det(S2))
    D = dist(S2, U2)

    # D2 = dist(S2, U2)
    # d_alpha = np.quantile(D2, alpha)

    return c2, A2, max(D)

def project_on_pc(X1, X2, alpha=0.99):
    c1, A1 = find_loc_and_pc_2d(X1, alpha=alpha)
    U1 = (X1 - c1) @ A1.T
    U2 = (X2 - c1) @ A1.T
    return U1, U2

def pca_compare_symmetrical_2d(X1, X2, label1, label2, alpha=0.9999, normalize_S=False):

    c1, A1, d1_alpha = find_loc_and_pc_2d(X1, alpha=alpha, normalize_S=normalize_S)
    U1 = (X1 - c1) @ A1.T
    S1 = np.linalg.inv(U1.T @ U1)
    S1 /= np.sqrt(np.linalg.det(S1))

    c2, A2, d2_alpha = find_loc_and_pc_2d(X2, alpha=alpha, normalize_S=normalize_S)
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


def pca_compare_2d(X1, X2, label1, label2, alpha=0.975, normalize_S=False):

    c1, A1, d1_alpha = find_loc_and_pc_2d(X1, alpha=alpha, normalize_S=normalize_S)
    U1 = (X1 - c1) @ A1.T
    S1 = np.linalg.inv(U1.T @ U1)
    if normalize_S:
        S1 /= np.sqrt(np.linalg.det(S1))

    U2 = (X2 - c1) @ A1.T
    c2, A2, d2_alpha = find_loc_and_pc_2d(U2, alpha=alpha)
    UU2 = (U2 - c2) @ A2.T
    S2 = np.linalg.inv(UU2.T @ UU2)
    if normalize_S:
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
    print(f"{label1}: err={k1/len(U1):.2f}")
    print(f"{label2}: err={k2/len(U2):.2f}")

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
    
def pca_robust_compare_2d(X1, X2, label1, label2, alpha=0.975, normalize_S=False):

    c1, A1, d1_alpha = find_robust_loc_and_pc_2d(X1, alpha=alpha, normalize_S=normalize_S)
    U1 = (X1 - c1) @ A1.T
    S1 = np.linalg.inv(U1.T @ U1)
    if normalize_S:
        S1 /= np.sqrt(np.linalg.det(S1))

    U2 = (X2 - c1) @ A1.T
    c2, A2, d2_alpha = find_robust_loc_and_pc_2d(U2, alpha=alpha, normalize_S=normalize_S)
    UU2 = (U2 - c2) @ A2.T
    S2 = np.linalg.inv(UU2.T @ UU2)
    if normalize_S:
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
    print(f"{label1}: err={k1/len(U1):.2f}")
    print(f"{label2}: err={k2/len(U2):.2f}")

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
    

