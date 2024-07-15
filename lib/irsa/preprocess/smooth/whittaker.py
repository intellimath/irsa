# from mlgrad.af import averaging_function
# from mlgrad.funcs import Quantile_Sqrt
from mlgrad.funcs2 import SquareDiff1, SquareDiff2
from mlgrad.funcs import Square
from mlgrad.averager import ArrayAdaM2

import math

import numpy as np

class WhittakerSmoother:
    #
    def __init__(self, func=None, func1=None, func2=None, h=0.01, n_iter=2000, 
                 tol=1.0e-8, tau1=4.0, tau2=10.0, collect_qvals=False):
        if func is None:
            self.func = Square()
        else:
            self.func = func
        if func1 is None: 
            self.func1 = SquareDiff1()
        else:
            self.func1 = func1
        if func2 is None: 
            self.func2 = SquareDiff2()
        else:
            self.func2 = func2
        self.n_iter = n_iter
        self.tol = tol
        self.h = h
        self.tau2 = tau2
        self.tau1 = tau1
        self.Z = None
        self.collect_qvals = collect_qvals
        self.qvals = None
    #
    def fit(self, X, weights=None):
        h = self.h
        tau1 = self.tau1
        tau2 = self.tau2
        tol = self.tol
        func = self.func
        func1 = self.func1
        func2 = self.func2

        if weights is None:
            W = np.ones_like(X)
        else:
            W = np.asarray(weights)

        averager = ArrayAdaM2()
        averager.init(len(X))
        
        if self.Z is None:
            Z = X.copy()
        else:
            Z = self.Z
        Z_min = Z.copy()

        ZX = Z - X
        qval = func.evaluate_weighted_sum(ZX, W) + tau2 * func2.evaluate(Z) + tau1 * func1.evaluate(Z)
    
        qval_min = qval
        qval_min_prev = 10*qval_min

        if self.collect_qvals:
            qvals = [qval]

        for K in range(self.n_iter):
            qval_prev = qval

            grad = func.derivative_weighted_array(ZX, W) + tau2 * func2.gradient(Z) + tau1 * func1.gradient(Z)

            averager.update(grad, h)

            Z -= averager.array_average

            ZX = Z - X
            qval = func.evaluate_weighted_sum(ZX, W) + tau2 * func2.evaluate(Z) + tau1 * func1.evaluate(Z)

            if self.collect_qvals:
                qvals.append(qval)

            if qval < qval_min:
                qval_min_prev = qval_min
                qval_min = qval
                Z_min = Z.copy()

            if abs(qval - qval_prev) / (1.0 + abs(qval_min)) < tol:
                break

            if abs(qval_min - qval_min_prev) / (1.0 + abs(qval_min)) < tol:
                break

        self.Z = Z_min
        self.K = K+1
        if self.collect_qvals:
            self.qvals = qvals

def whittaker(X, func=None, func1=None, func2=None, weights=None, h=0.01, tau2=10.0, tau1=4.0, n_iter=1000, collect_qvals=False):
    alg = WhittakerSmoother(func=func, func1=func1, func2=func2, h=h, tau2=tau2, tau1=tau1, n_iter=n_iter)
    alg.fit(X, weights=weights)
    if collect_qvals:
        return alg.Z, alg.qvals
    else:
        return alg.Z
    