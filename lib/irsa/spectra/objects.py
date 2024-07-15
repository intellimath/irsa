#
# objects.py
#

import numpy as np
import rampy 
from irsa.preprocess import smooth, utils

class ExperimentSpectrasSeries:
    #
    def __init__(self, x, y, attrs):
        self.x = x
        self.y = y
        self.attrs = attrs
    #
    def crop(self, start_index, end_index=None):
        Xs = self.x
        Ys = self.y
        for k in range(len(Ys)):
            if end_index is None:
                Xs[k] = Xs[k][start_index:]
                Ys[k] = Ys[k][:,start_index:]
            else:
                Xs[k] = Xs[k][start_index:end_index]
                Ys[k] = Ys[k][:,start_index:end_index]
        
    def allign_bottom(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] -= y.min()
    #
    def robust_averaging(self, **kwargs):
        from irsa.preprocess.utils import robust_mean2
    
        if len(self.y[0].shape) == 1:
            raise TypeError("Усреднять можно только в сериях")
        tau = kwargs.get("tau", 3.5)
        Ys = self.y.copy()
        for k in range(len(Ys)):
            ys = robust_mean2(Ys[k], tau=tau)
            ys -= ys.min()
            Ys[k] = ys

        return ExperimentSpectras(self.x, Ys, self.attrs)
    #

class ExperimentSpectras:
    #
    def __init__(self, x, y, attrs):
        self.x = x
        self.y = y
        self.attrs = attrs
    #
    def crop(self, start_index, end_index=None):
        Xs = self.x
        Ys = self.y
        for k in range(len(Ys)):
            if end_index is None:
                Xs[k] = Xs[k][start_index:]
                Ys[k] = Ys[k][start_index:]
            else:
                Xs[k] = Xs[k][start_index:end_index]
                Ys[k] = Ys[k][start_index:end_index]
    #
    def allign_bottom(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            Ys[k] -= ys.min()
    #
    def normalize_area_under_curve(self):
        Ys = self.y
        Xs = self.x
        for k in range(len(Ys)):
            ys = Ys[k]
            xs = Xs[k]
            ys /= np.trapz(ys, xs)
            ys *= 1000
            Ys[k] = ys
    #
    def remove_outlier_spectras(self, **kwargs):
        tau = kwargs.get("tau", 3.5)
        Xs, Ys = self.x, self.y
        Ys = np.array(Ys)
        
        utils.mark_outliers2(Ys, tau=tau)

        Is = []
        for i, ys in enumerate(Ys):
            if np.any(np.isnan(ys)):
                continue
            Is.append(i)

        if Is:
            Is = np.array(Is)
            Ys = Ys[Is]
        
        self.y = Ys
    #
    def replace_outlier_spectras(self, **kwargs):
        tau = kwargs.get("tau", 3.5)
        Xs, Ys = self.x, self.y
        Ys = np.array(Ys)
        
        utils.replace_outliers2(Ys, tau=tau)
        
        self.y = Ys
    #
    def smooth(self, **kwargs):
        Xs = self.x
        Ys = self.y
        tau = kwargs.get('tau', 1.0)
        method = kwargs.get('method', 'runpy')
        if method == "runpy":
            for k in range(len(Ys)):
                ys = Ys[k]
                xs = Xs[k]
                ys[:] = rampy.smooth(xs, ys, Lambda=tau)
        elif method == "irsa":
            for k in range(len(Ys)):
                ys = Ys[k]
                xs = Xs[k]
                ys[:] = smooth.whittaker(ys, tau2=tau, tau1=0, h=0.001)
    #
    def subtract_baseline(self, kind="aspls", pad=0, **kwargs):
        import pybaselines
        
        Xs = self.x
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            xs = Xs[k]
            if pad:
                ys1 = np.pad(ys, pad, mode="edge")
                xs1 = np.pad(xs, pad, mode="linear_ramp")
            else:
                xs1 = xs
                ys1 = ys

            if kind == "aspls":
                bs1, _ = pybaselines.whittaker.aspls(ys1, x_data=xs1, **kwargs)
            elif kind == "arpls":
                bs1, _ = pybaselines.whittaker.arpls(ys1, x_data=xs1, **kwargs)
            elif kind == "mor":
                bs1, _ = pybaselines.morphological.mor(ys1, x_data=xs1, **kwargs)
            ys1[:] -= bs1

            if pad:
                ys[:] = ys1[pad:-pad]
                xs[:] = xs1[pad:-pad]
    
            np.putmask(ys, ys < 0, 0)
    