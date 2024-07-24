#
# objects.py
#

import numpy as np
import rampy 
from irsa.preprocess import utils
import mlgrad.smooth as smooth

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
    def smooth(self, tau=1.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = smooth.whittaker(y, tau2=tau, tau1=0, h=0.001, tol=1.0e6)
    #
    def remove_overflow_spectras(self, y_max=2000.0, y_max_count=10):
        Xs, Ys = self.x, self.y
        Is = []

        for k in range(len(Ys)):
            ys = Ys[k]
        
            ids = []
            for i, y in enumerate(ys):
                count = (y > y_max).sum()
                if count > y_max_count:
                    continue
                ids.append(i)
            
            if ids:
                if len(ids) > 5:
                    ids = np.array(ids)
                    ys = ys[ids]
                    Ys[k] = ys
                    Is.append(k)

        if Is:
            Ys = [Ys[k] for k in Is]
            Xs = [Xs[k] for k in Is]
            self.x, self.y = Xs, Ys
                
    #
    def remove_by_zscore_spectras(self, tau=3.5, max_count=40):
        Xs, Ys = self.x, self.y
        Is = []

        for k in range(len(Ys)):
            ys = Ys[k]

            zs = utils.modified_zscore(ys)
            ids = []
            for i,z in enumerate(zs):
                if (abs(z) > tau).sum() < max_count:
                    ids.append(i)

            if ids:
                if len(ids) > 5:
                    ids = np.array(ids)
                    ys = ys[ids]
                    Ys[k] = ys
                    Is.append(k)

        if Is:
            Ys = [Ys[k] for k in Is]
            Xs = [Xs[k] for k in Is]
            self.x, self.y = Xs, Ys
                
    #
    def remove_outlier_spectras(self, delta=0.10, tau=3.5, max_count=30):
        from irsa.preprocess.utils import robust_mean2

        median = np.median

        Xs, Ys = self.x, self.y
        ids = []
        for k in range(len(Ys)):
            ys = Ys[k]
            xs = Xs[k]
            ym = robust_mean2(ys, tau=tau)
            dy = abs(ys - ym)
            dd = dy / (ym+0.001)
            if median(dd) <= delta:
                ids.append(k)
            # else:
            #     print(ds)
        Ys = [Ys[k] for k in ids]
        Xs = [Xs[k] for k in ids]
        self.x, self.y = Xs, Ys
    #    
    def robust_averaging(self, tau=3.5):
        from irsa.preprocess.utils import robust_mean2
    
        if len(self.y[0].shape) == 1:
            raise TypeError("Усреднять можно только в сериях")

        Ys = self.y.copy()
        for k in range(len(Ys)):
            ys = robust_mean2(Ys[k], tau=tau)
            ys -= ys.min()
            Ys[k] = ys

        Ys = np.ascontiguousarray(Ys)
        Xs = np.ascontiguousarray(self.x)

        return ExperimentSpectras(Xs, Ys, self.attrs)
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
    def remove_overflow_spectras(self, y_max=1500, y_max_count=100):
        Xs, Ys = self.x, self.y

        Is = []
        for i, ys in enumerate(Ys):
            if (ys >= 1500).sum() < y_max_count:
                continue
            Is.append(i)
        
        if Is:
            Is = np.array(Is)
            Ys = Ys[Is]
            Xs = Ys[Is]

        self.x, self.y = Xs, Ys
    #
    def remove_by_zscore_spectras(self, tau=3.5, max_count=40):
        Xs, Ys = self.x, self.y
        Is = []

        Zs = utils.modified_zscore(Ys)
        ids = []
        for i,zs in enumerate(Zs):
            if (abs(zs) > tau).sum() < max_count:
                ids.append(i)

        if ids:
            if len(ids) > 5:
                ids = np.array(ids)
                Ys = Ys[ids]
                Xs = Xs[ids]

        self.x, self.y = Xs, Ys
                
    #
    def remove_outlier_spectras(self, tau=3.5):
        Xs, Ys = self.x, self.y

        Is = []
        for i, ys in enumerate(Ys):
            if np.any(np.isnan(ys)):
                continue
            Is.append(i)
        
        utils.mark_outliers2(Ys, tau=tau)

        Is = []
        for i, ys in enumerate(Ys):
            if np.any(np.isnan(ys)):
                continue
            Is.append(i)

        if Is:
            Is = np.array(Is)
            Ys = Ys[Is]
            Xs = Ys[Is]

        self.x, self.y = Xs, Ys
    #
    def replace_outlier_spectras(self, tau=3.5):
        Xs, Ys = self.x, self.y
        Ys = np.array(Ys)
        
        utils.replace_outliers2(Ys, tau=tau)
        
        self.y = Ys
    #
    def smooth(self, method="irsa", tau=1.0, **kwargs):
        Xs = self.x
        Ys = self.y
        if method == "runpy":
            for k in range(len(Ys)):
                ys = Ys[k]
                xs = Xs[k]
                ys[:] = rampy.smooth(xs, ys, Lambda=tau)
        elif method == "irsa":
            for k in range(len(Ys)):
                ys = Ys[k]
                xs = Xs[k]
                func = kwargs.get("func", None)
                ys[:] = smooth.whittaker(ys, func=func, tau2=tau, tau1=0, h=0.01)
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
    