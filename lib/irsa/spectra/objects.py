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
        attr_names = (
            "вид_бактерий", "штамм_бактерий", "резистентность", 
            "отсечки_по_молекулярной_массе", "начальная_концентрация_клеток_в_пробе", 
            "номер_эксперимента_в_цикле", "номер_повтора", "дата", "комментарий"
        )
        key = "_".join(
            attrs[k] for k in attr_names)
        self.key = key
    #
    def plot_spectras(self, ax=None):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
        
        f_slider = ipywidgets.FloatSlider(value=3.5, min=1.0, max=10.0)
        f_slider.layout.width="50%"        

        @ipywidgets.interact(i=i_slider, f=f_slider, zscore=False)
        def _plot_spectras(i, f, zscore):
            # i_slider.value=i
            plt.figure(figsize=(12,4))
            plt.title(self.key)
            i = i_slider.value
            xs = self.x[i]
            Ys = self.y[i]
            for ys in Ys:
                plt.plot(xs, ys, linewidth=0.75)
                
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()
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
    #
    def use_range(self, start_index=0, end_index=None):
        Ys = self.y
        for k in range(len(Ys)):
            if end_index is None:
                Ys[k] = Ys[k][start_index:,:]
            else:
                Ys[k] = Ys[k][start_index:end_index,:]
        self.y = Ys
    #
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
                y[:] = smooth.whittaker_smooth(y, tau=tau)[0]
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
                # if len(ids) > 5:
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
                # if len(ids) > 5:
                ids = np.array(ids)
                ys = ys[ids]
                Ys[k] = ys
                Is.append(k)

        if Is:
            Ys = [Ys[k] for k in Is]
            Xs = [Xs[k] for k in Is]
            self.x, self.y = Xs, Ys
                
    #
    def remove_outlier_spectras(self, delta=0.10, tau=3.0, max_count=30):
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
    def replace_small_values(self, delta):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            np.putmask(ys, abs(ys)<delta, 0)
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

        Zs = utils.modified_zscore2(Ys)
        ids = []
        for i,zs in enumerate(Zs):
            count = (abs(zs) > tau).astype('i').sum()
            if count < max_count:
                ids.append(i)
            # else:
            #     print(count)

        if ids:
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
    def smooth(self, method="runpy", tau=1.0, **kwargs):
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
                # func = kwargs.get("func", None)
                # func2 = kwargs.get("func2", None)
                ys[:] = smooth.whittaker_smooth(ys, tau=tau)
                np.putmask(ys, ys < 0, 0)

    #
    def get_baselines(self, kind="aspls", **kwargs):
        import pybaselines
        
        Xs = self.x
        Ys = self.y
        Bs = []
        for k in range(len(Ys)):
            ys = Ys[k]
            xs = Xs[k]

            if kind == "aspls":
                lam = kwargs.get("lam", 1.0e5)
                diff_order = kwargs.get("diff_order", 2)
                bs, _ = pybaselines.whittaker.aspls(ys, x_data=xs, 
                                                     lam=lam, diff_order=diff_order,
                                                     **kwargs)
            elif kind == "arpls":
                lam = kwargs.get("lam", 1.0e5)
                diff_order = kwargs.get("diff_order", 2)
                bs, _ = pybaselines.whittaker.arpls(ys, x_data=xs,
                                                     lam=lam, diff_order=diff_order, 
                                                     **kwargs)
            elif kind == "mor":
                bs, _ = pybaselines.morphological.mor(ys, x_data=xs, **kwargs)

        Bs.append(bs)

        return Bs
        
    def subtract_baseline(self, kind="aspls", **kwargs):
        import pybaselines
        
        Xs = self.x
        Ys = self.y
        Bs = []
        for k in range(len(Ys)):
            ys = Ys[k]
            xs = Xs[k]

            if kind == "aspls":
                lam = kwargs.pop("lam", 1.0e5)
                diff_order = kwargs.pop("diff_order", 2)
                bs, _ = pybaselines.whittaker.aspls(ys, x_data=xs, 
                                                     lam=lam, diff_order=diff_order,
                                                     **kwargs)
            elif kind == "arpls":
                lam = kwargs.pop("lam", 1.0e5)
                diff_order = kwargs.pop("diff_order", 2)
                bs, _ = pybaselines.whittaker.arpls(ys, x_data=xs,
                                                     lam=lam, diff_order=diff_order, 
                                                     **kwargs)
            elif kind == "mor":
                bs, _ = pybaselines.morphological.mor(ys, x_data=xs, **kwargs)

            ys[:] -= bs
                
            np.putmask(ys, ys < 0, 0)
    