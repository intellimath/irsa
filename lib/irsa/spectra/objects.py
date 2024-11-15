#
# objects.py
#

import numpy as np
import rampy 
from irsa.preprocess import utils
import mlgrad.funcs as funcs
import mlgrad.smooth as smooth
import mlgrad.inventory as inventory
import mlgrad.array_transform as array_transform
import scipy.special as special

class ExperimentSpectrasSeries:
    #
    def __init__(self, x, y, attrs):
        self.x = x
        self.y = y
        self.attrs = attrs
        attr_names = (
            "вид_бактерий", "штамм_бактерий", "резистентность", 
            "отсечки_по_молекулярной_массе", "начальная_концентрация_клеток_в_пробе", 
            "номер_эксперимента_в_цикле", 
            "номер_повтора", "дата", "комментарий"
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
        
        xrange_slider = ipywidgets.IntRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="50%"

        @ipywidgets.interact(i=i_slider, f=f_slider, xrange=xrange_slider)
        def _plot_spectras(i, f, xrange):
            plt.figure(figsize=(12,4))
            plt.title(self.key)
            xs = self.x[i]
            Ys = self.y[i]
            for ys in Ys:
                plt.plot(xs, ys, linewidth=0.75)

            plt.plot(xs, inventory.robust_mean_2d_t(Ys, tau=f), linewidth=1.0, color='k')
                
            plt.minorticks_on()
            plt.tight_layout()
            plt.xlim(*xrange)
            plt.show()
    #
    def crop(self, start_index, end_index=None):
        Xs = self.x
        Ys = self.y
        for k in range(len(Ys)):
            if end_index is None:
                Xs[k] = np.ascontiguousarray(Xs[k][start_index:])
                Ys[k] = np.ascontiguousarray(Ys[k][:,start_index:])
            else:
                Xs[k] = np.ascontiguousarray(Xs[k][start_index:end_index])
                Ys[k] = np.ascontiguousarray(Ys[k][:,start_index:end_index])
    #
    def use_range(self, start_index=0, end_index=None):
        Ys = self.y
        for k in range(len(Ys)):
            if end_index is None:
                Ys[k] = np.ascontiguousarray(Ys[k][start_index:,:])
            else:
                Ys[k] = np.ascontiguousarray(Ys[k][start_index:end_index,:])
        self.y = Ys
    #
    def allign_bottom(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] -= y.min()
    #
    def scale(self, delta=3.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                z = abs(utils.modified_zscore(y))
                mu = y[z > delta].mean()
                y[:] = y / mu
    #
    def smooth(self, tau=1.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = smooth.whittaker_smooth(y, tau=tau, d=2, solver="fast")[0]
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
        # from irsa.preprocess.utils import robust_mean2
    
        if len(self.y[0].shape) == 1:
            raise TypeError("Усреднять можно только в сериях")

        Ys = self.y.copy()
        dYs = []
        for k in range(len(Ys)):
            # ys_k = np.ascontiguousarray(Ys[k])
            ys_k = Ys[k]
            ys = inventory.robust_mean_2d_t(ys_k, tau=tau)
            dys = inventory.robust_mean_2d_t(abs(ys_k - ys), tau=tau)
            dYs.append(dys)
            # ys -= ys.min()
            Ys[k] = ys

        Ys = np.ascontiguousarray(Ys)
        Xs = np.ascontiguousarray(self.x)

        o = ExperimentSpectras(Xs, Ys, self.attrs)
        o.key = self.key
        o.dy = np.array(dYs)
        return o
    #

def array_rel_max(E):
    abs_E = abs(E)
    max_E = max(abs_E)
    min_E = min(abs_E)
    rel_E =  (abs_E - min_E) / (max_E - min_E)
    return rel_E

def array_expit_sym(E):
    return special.expit(-E)
def array_expit(E):
    return special.expit(E)
def array_sigmoid_pos(E):
    return 2*special.expit(-E) - 1

def array_sqrtit(E):
    return (1 - E / np.sqrt(1 + E*E)) / 2
def array_sqrtit_sym(E):
    return (1 + E / np.sqrt(1 + E*E)) / 2

def sqrt2(E):
    # E /= np.median(np.abs(E))
    return (1 + E / np.sqrt(1 + E*E)) / 2
def gauss(E):
    return 1-np.exp(-E*E/2)
    

class ExperimentSpectras:
    #
    def __init__(self, x, y, attrs):
        self.x = x
        self.y = y
        self.attrs = attrs
    #
    def crop(self, start_index, end_index=None):
        if end_index is None:
            self.x = self.x[:,start_index:]
            self.y = self.y[:,start_index:]
        else:
            self.x = self.x[:,start_index:end_index]
            self.y = self.y[:,start_index:end_index]
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
            ys[:] = ys - ys.min()
    #
    def apply_func(self, y_func, x_func=None, a=1, b=0):
        Ys = self.y
        Xs = self.x
        for k in range(len(Ys)):
            ys = Ys[k]
            ys[:] = y_func(a*ys + b)
        if x_func is not None:
            for k in range(len(Xs)):
                xs = Xs[k]
                xs[:] = x_func(a*xs + b)
    #
    # def normalize_area_under_curve(self):
    #     Ys = self.y
    #     Xs = self.x
    #     for k in range(len(Ys)):
    #         ys = Ys[k]
    #         xs = Xs[k]
    #         ys = ys / np.trapezoid(ys)
    #         # ys /= np.mean(ys)
    #         ys *= len
    # #
    def scale(self, delta=3.5, scale=100):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            mu = utils.robust_mean(ys)
            ys[:] = (ys / mu) * scale
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
                ys[:] = smooth.whittaker_smooth(ys, tau=tau, solver="fast")
        np.putmask(ys, ys < 0, 0)
    #
    def select_baselines(self, kind="irsa", tau=1000.0, bs_scale=1.5, solver="fast", 
                         func=None, func2=None, d=2, **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.x)
        self.tau_values = N * [tau]

        max_tau = max(self.tau_values)
        max_tau *= 4
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        tau_slider = ipywidgets.FloatSlider(value=self.tau_values[0], min=1.0, max=max_tau, step=1.0)
        tau_slider.layout.width="80%"   
        
        def tau_on_value_change(change):
            i = i_slider.value
            self.tau_values[i] = tau_slider.value

        def i_on_value_change(change):
            i = i_slider.value
            tau_slider.value = self.tau_values[i]

        def xrange_on_value_change(change):
            xmin, xmax = xrange_slider.value
            plt.xlim(xmin, xrange)
            i = i_slider.value
            xs = self.x[i]
            ys = self.y[i]
            ys = ys[xs >= xmin & xs <= xmax]
            plt.ylim(0.9*min(ys), 1.1*max(ys))
        
        tau_slider.on_trait_change(tau_on_value_change, name="value")
        i_slider.on_trait_change(i_on_value_change, name="value")
                
        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="80%"

        # def _func1(E):
        #     return special.expit(-E)

        if func is None:
            func = funcs.Hinge2()

        # def _func2(E,Z):
        #     E = abs(E)
        #     return E / E.max()

        if func2 is None:
            func2 = funcs.Square()

        @ipywidgets.interact(i=i_slider, tau=tau_slider, xrange=xrange_slider)
        def _plot_spectras(i, tau, xrange):
            plt.figure(figsize=(13,5))
            plt.title(self.key)
            for xs, ys in zip(self.x, self.y):
                plt.plot(xs, ys, linewidth=0.25, alpha=0.25)

            xs_i = self.x[i]
            ys_i = self.y[i]
            plt.plot(xs_i, ys_i, linewidth=1.0, color='k')
            # plt.fill_between(xs_i, ys_i - self.dy[i], ys_i + self.dy[i], color='LightBlue', alpha=0.5)
            
            # ys_i_smooth = ys_i
            ys_i_smooth = smooth.whittaker_smooth(ys_i, tau=1.0, solver=solver)
            plt.plot(xs_i, ys_i_smooth, linewidth=1.0, color='DarkBlue')

            bs, dd = smooth.whittaker_smooth_weight_func2(
                ys_i_smooth, 
                func=func,
                func2=func2, 
                tau=self.tau_values[i], d=d, solver=solver)
    
            plt.plot(xs_i, bs, linewidth=1.0, color='m')

            x_min, x_max = xrange
            plt.xlim(x_min, x_max)

            bs_xrange = bs[(x_min <= xs_i) & (xs_i <= x_max)]
            ys_xrange = ys_i[(x_min <= xs_i) & (xs_i <= x_max)]
            bs_max, bs_min = np.max(bs_xrange), np.min(bs_xrange)
            plt.ylim(0.95*np.min(ys_xrange), bs_scale*(bs_max))

                        
            plt.minorticks_on()
            plt.tight_layout()
            plt.grid(1)
            plt.show()
            
            # plt.figure(figsize=(10,3))
            # plt.plot(np.log10(dd['qvals']))
            # plt.show()
    #
    def get_baselines(self, kind="irsa", **kwargs):
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

            elif kind == "irsa":
                lam = kwargs.pop("lam", 1.0e3)
            bs, _ = smooth.whittaker_smooth_weight_func(ys, 
                        weight_func=array_transform.array_sqrtit, 
                        weight_func2=array_transform.array_rel_max, 
                        tau=lam, d=2, solver="fast")
        
        Bs.append(bs)

        return Bs
        
    def subtract_baseline(self, kind="irsa", **kwargs):
        import pybaselines

        def rel_error(E,Z):
            abs_E = abs(E)
            return abs_E / max(abs_E)
        def sign2(E):
            return expit(-E / np.median(abs(E)) / 3)
        def sign(E):
            e = 1
            return (1 - E / np.sqrt(e*e + E*E))/2
        
        Xs = self.x
        Ys = self.y
        # Bs = []
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

            elif kind == "irsa":
                lam = kwargs.pop("lam", 1.0e3)
                bs, _ = smooth.whittaker_smooth_weight_func(ys, 
                            weight_func=array_transform.array_sqrtit, 
                            weight_func2=array_transform.array_rel_max, 
                            tau=lam, d=2, solver="fast")
            
            ys[:] = ys - bs
                
            np.putmask(ys, ys < 0, 0)
    #
    def plot_spectras(self, ax=None, baseline=True):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
        
        # f_slider = ipywidgets.FloatSlider(value=3.5, min=1.0, max=10.0)
        # f_slider.layout.width="50%"   
        
        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="50%"

        @ipywidgets.interact(i=i_slider, xrange=xrange_slider)
        def _plot_spectras(i, xrange):
            plt.figure(figsize=(12,4))
            plt.title(self.key)
            for xs, ys in zip(self.x, self.y):
                plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            xs_i = self.x[i]
            ys_i = self.y[i]
            plt.plot(xs_i, ys_i, linewidth=1.5, color='k')
            
            ys_i_smooth = ys_i
            # ys_i_smooth = smooth.whittaker_smooth(ys_i, tau=1.0, solver="fast")

            def rel_error(E):
                abs_E = abs(E)
                return abs_E / max(abs_E)
            def sign2(E):
                return expit(-E / np.median(abs(E)) / 3)
            def sign(E):
                e = 1
                return (1 - E / np.sqrt(e*e + E*E))/2

            if baseline:
                bs, _ = smooth.whittaker_smooth_weight_func(
                    ys_i_smooth, 
                    weight_func=array_transform.array_sqrtit, 
                    weight_func2=array_transform.array_rel_max, 
                    tau=1000.0, d=2, solver="fast")
    
                plt.plot(xs_i, bs, linewidth=1.5, color='m')

            x_min, x_max = xrange
            ys_range = ys_i[(x_min <= xs_i) & (xs_i <= x_max)]
            ymin, y_max = 0.95*np.min(ys_range), 1.05*np.max(ys_range)
            plt.ylim(ymin, y_max)
            
            plt.minorticks_on()
            plt.tight_layout()
            plt.xlim(x_min, x_max)
            plt.show()
    #
    