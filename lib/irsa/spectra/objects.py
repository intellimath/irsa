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

from mlgrad.af import averaging_function

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

            ys_m = inventory.robust_mean_2d_t(Ys, tau=f)
            std = inventory.robust_mean_2d_t(abs(Ys - ys_m), tau=f)

            plt.fill_between(xs, ys_m-2*std, ys+2*std, alpha=0.5)
            plt.plot(xs, ys_m, linewidth=1.5, color='k')
                
            plt.minorticks_on()
            plt.tight_layout()
            plt.xlim(*xrange)
            plt.show()
            
            # plt.figure(figsize=(12,4))
            # plt.title(self.key)
            # xs = self.x[i]
            # Ys = self.y[i]
            # # ys_m = inventory.robust_mean_2d_t(Ys, tau=f)
            # Zs = utils.modified_zscore2(abs(Ys - ys_m))
            # for zs in Zs:
            #     plt.plot(xs, abs(zs), linewidth=0.75)
            # plt.xlim(*xrange)
            # plt.show()
    #
    def crop(self, start_index, end_index=None):
        Xs = self.x
        Ys = self.y 
        N = len(Ys)
        for k in range(N):
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
    def scale_zs(self, delta=3.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                z = abs(array_transform.array_modified_zscore(y))
                mu = y[z <= delta].mean()
                y[:] = y / mu
    #
    def to_modified_zscore(self, delta=3.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                z = array_transform.array_modified_zscore(y)
                y[:] = z
    #
    def scale_min(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = y / y.min()
    #
    def smooth(self, tau=1.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = smooth.whittaker_smooth_func2(y, tau=tau, d=2, solver="fast")[0]
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
            Ys[k] = ys

        Ys = np.ascontiguousarray(Ys)
        Xs = np.ascontiguousarray(self.x)

        o = ExperimentSpectras(Xs, Ys, self.attrs)
        o.key = self.key
        o.stderr = np.array(dYs)
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
        self.bs = np.zeros_like(y)
        self.params = {}
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
    def scale(self, tau=3.5, scale=100):
        N = len(self.y)
        for k in range(N):
            ys = self.y[k]
            err = self.stderr[k]
            mu = utils.robust_mean(ys, tau=tau)
            ys[:] = (ys / mu) * scale
            err[:] = (err / mu) * scale
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
    def smooth(self, method="irsa2", tau=1.0, **kwargs):
        Xs = self.x
        Ys = self.y
        solver = kwargs.get("solver", "fast")
        if method == "runpy":
            for k in range(len(Ys)):
                ys = Ys[k]
                xs = Xs[k]
                ys[:] = rampy.smooth(xs, ys, Lambda=tau)
        elif method == "irsa":
            for k in range(len(Ys)):
                ys = Ys[k]
                xs = Xs[k]
                ys[:] = smooth.whittaker_smooth(ys, tau=tau, solver=solver)
        elif method == "irsa2":
            func = kwargs.get("func", funcs.Square())
            func2 = kwargs.get("func2", None)
            for k in range(len(Ys)):
                ys = Ys[k]
                xs = Xs[k]
                ys[:], _ = smooth.whittaker_smooth_weight_func(
                            ys, tau=tau, 
                            func=func, 
                            func2=func2, 
                            solver=solver)
        # np.putmask(ys, ys < 0, 0)
    #
    def select_baselines(self, kind="irsa", tau2=1000.0, tau1=0.0, tau_z=0, tau_smooth=1,
                         bs_scale=3.0, solver="fast", 
                         func = None,
                         func2 = None,                         
                         func1=None, d=2, func2_mode="d", **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.y)
        if hasattr(self, "bs"):
            Bs = self.bs
        else:        
            self.bs = Bs = np.zeros((N, len(self.y[0])))
        
        params = self.params
        Ls = params.get("lam", None)
        
        self.tau2_values = (N * [tau2])  if Ls is None else Ls
        self.tau1_values = N * [tau1]

        tau2_max = max(self.tau2_values)
        tau2_max *= 20

        # tau1_max = max(self.tau1_values)
        # tau1_max *= 6
        
        # if tau1 < 1.0:
        #     tau1_min = tau1 / 10.0
        #     tau1_step = tau1_min
        # else:
        #     tau1_min = 1.0
        #     tau1_step = 1.0

        if tau2 < 1.0:
            tau2_min = tau2 / 10.0
            tau2_step = tau2_min
        else:
            tau2_min = 1.0
            tau2_step = 1.0
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        tau2_slider = ipywidgets.FloatSlider(value=self.tau2_values[0], min=tau2_min, max=tau2_max, step=tau2_step)
        tau2_slider.layout.width="80%"   

        # tau1_slider = ipywidgets.FloatSlider(value=self.tau1_values[0], min=tau1_min, max=tau1_max, step=tau1_step)
        # tau1_slider.layout.width="80%"   
        
        def tau2_on_value_change(change):
            i = i_slider.value
            self.tau2_values[i] = tau2_slider.value

        # def tau1_on_value_change(change):
        #     i = i_slider.value
        #     self.tau1_values[i] = tau1_slider.value
            
        def i_on_value_change(change):
            i = i_slider.value
            tau2_slider.value = self.tau2_values[i]
            # tau1_slider.value = self.tau1_values[i]

        def xrange_on_value_change(change):
            xmin, xmax = xrange_slider.value
            plt.xlim(xmin, xrange)
            i = i_slider.value
            xs = self.x[i]
            ys = self.y[i]
            ys = ys[xs >= xmin & xs <= xmax]
            plt.ylim(0.9*min(ys), 1.1*max(ys))
        
        tau2_slider.on_trait_change(tau2_on_value_change, name="value")
        # tau1_slider.on_trait_change(tau1_on_value_change, name="value")
        i_slider.on_trait_change(i_on_value_change, name="value")
                
        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="80%"
        
        # def _func1(E):
        #     return special.expit(-E)

        # if func is None:
        #     func = funcs.SoftHinge_Sqrt(0.001)

        # if func1 is None:
        #     func1 = funcs.RELU()

        # if func2 is None:
        #     func2 = funcs.SoftHinge_Sqrt(0.001)

        # for i in range(len(self.x)):
        #     xs_i, ys_i = self.x[i], self.y[i]
        #     ys_i_smooth = smooth.whittaker_smooth(ys_i, tau1=1.0, tau2=1.0, solver=solver)
        #     bs, _ = smooth.whittaker_smooth_weight_func(
        #         ys_i_smooth, 
        #         func=func,
        #         func1=func1,
        #         func2=func2, 
        #         tau1=self.tau1_values[i], 
        #         tau2=self.tau2_values[i], 
        #         d=d, solver=solver, func2_mode=func2_mode)
        #     # print(bs)
        #     self.bs[i,:] = bs
                    
        # @ipywidgets.interact(i=i_slider, tau2=tau2_slider, tau1=tau1_slider, xrange=xrange_slider)
        # def _plot_spectras(i, tau2, tau1, xrange):
        @ipywidgets.interact(i=i_slider, tau2=tau2_slider, xrange=xrange_slider)
        def _plot_spectras(i, tau2, xrange):
            plt.figure(figsize=(13,4))
            plt.title(self.key)
            for xs, ys in zip(self.x, self.y):
                plt.plot(xs, ys, linewidth=0.25, alpha=0.4)

            xs_i = self.x[i]
            ys_i = self.y[i]
            plt.plot(xs_i, ys_i, linewidth=1.5, color='Grey')
            # plt.fill_between(xs_i, ys_i - self.dy[i], ys_i + self.dy[i], color='LightBlue', alpha=0.5)

            err_i = self.stderr[i]
            plt.fill_between(xs_i, ys_i-err_i, ys_i+err_i, alpha=0.5)

            std_err = err_i.mean()
            
            def smooth_func2(x, scale=kwargs["ww"]):
                v = x/scale
                return 1/(1+abs(v))
            
            # ys_i_smooth = ys_i
            ys_i_smooth = smooth.whittaker_smooth_weight_func(
                ys_i, func2=None, tau2=tau_smooth, solver=solver)[0]            
            plt.plot(xs_i, ys_i_smooth, linewidth=1.5, color='DarkBlue')

            bs, dd = smooth.whittaker_smooth_weight_func(
                ys_i_smooth, 
                func=func,
                func2=func2,
                # tau1=self.tau1_values[i], 
                tau2=self.tau2_values[i], 
                tau_z=tau_z,
                d=d, solver=solver, func2_mode=func2_mode)
            # print(bs)
            self.bs[i,:] = bs
            # self.y[i,:] = ys_i_smooth
    
            plt.plot(xs_i, bs, linewidth=1.0, color='m')

            x_min, x_max = xrange
            plt.xlim(0.95*x_min, 1.05*x_max)

            bs_xrange = bs[(x_min <= xs_i) & (xs_i <= x_max)]
            ys_xrange = ys_i[(x_min <= xs_i) & (xs_i <= x_max)]
            bs_max, bs_min = np.max(bs_xrange), np.min(bs_xrange)
            # plt.ylim(0.95*np.min(ys_xrange), bs_scale*(bs_max))
            plt.ylim(0, bs_scale*(bs_max))

                        
            plt.minorticks_on()
            plt.tight_layout()
            plt.grid(1)
            plt.show()

            # plt.figure(figsize=(13,5))
            # plt.title(self.key)
            # for xs, ys, bs in zip(self.x, self.y, self.bs):
            #     plt.plot(xs, ys-bs, linewidth=0.25, alpha=0.25)
            # plt.plot(self.x[i], self.y[i]-self.bs[i], linewidth=1.0, color='DarkBlue')
            # plt.xlim(0.95*x_min, 1.05*x_max)
            # plt.ylim(0, np.max(ys))
            # plt.show()
             
            plt.figure(figsize=(10,3))
            plt.plot(np.log10(dd['qvals']))
            plt.show()
    #
    def get_baselines(self, kind="irsa", **kwargs):
        import pybaselines
        
        Xs = self.x
        Ys = self.y

        N = len(Ys)
        
        if hasattr(self, "params"):
            params = self.params
        else:
            self.params = params = {}

        if hasattr(self, "bs"):
            Bs = self.bs
        else:        
            self.bs = Bs = np.zeros((N, len(Ys[0])))

        Ls = params.get("lam", None)
        if Ls is None:
            Ls = N * [lam]
            params["lam"] = Ls
            
        for k in range(len(Ys)):
            ys = Ys[k]
            xs = Xs[k]
            lam = Ls[k]

            if kind == "aspls":
                diff_order = kwargs.get("diff_order", 2)
                bs, _ = pybaselines.whittaker.aspls(ys, x_data=xs, 
                                                     lam=lam, diff_order=diff_order,
                                                     **kwargs)
            elif kind == "arpls":
                diff_order = kwargs.get("diff_order", 2)
                bs, _ = pybaselines.whittaker.arpls(ys, x_data=xs,
                                                     lam=lam, diff_order=diff_order, 
                                                     **kwargs)
            elif kind == "mor":
                bs, _ = pybaselines.morphological.mor(ys, x_data=xs, **kwargs)

            elif kind == "irsa":
                func = kwargs["func"]
                func2 = kwargs["func2"]
                bs, _ = smooth.whittaker_smooth_weight_func2(
                            ys, 
                            func=func, 
                            func2=func2, 
                            tau2=lam, d=2, solver="fast")
        
            Bs[k,:] = bs

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
        Bs = self.bs
        for k in range(len(Ys)):
            ys = Ys[k]
            xs = Xs[k]
            bs = Bs[k]
            ys[:] = ys - bs
        Bs.fill(0)
                
            # np.putmask(ys, ys < 0, 0)
    #
    def plot_spectras(self, tau=1.0, ss=100, ax=None, baseline=False, **kwargs):
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
            err_i = self.stderr[i]
            plt.fill_between(xs_i, ys_i-2*err_i, ys_i+2*err_i, alpha=0.5)
            plt.plot(xs_i, ys_i, linewidth=1.5, color='Grey', label="original")

            std_err = err_i.mean()

            def func2(x, scale=std_err):
                v = x/scale
                return 1/np.sqrt(1+v*v)
            
            # ys_i_smooth = ys_i
            ys_i_smooth = smooth.whittaker_smooth_weight_func2(
                ys_i, func2=None, tau2=tau, solver="scipy")[0]
            plt.plot(xs_i, ys_i_smooth, linewidth=1.5, color='DarkBlue', label="smoothed")

            def rel_error(E):
                abs_E = abs(E)
                return abs_E / max(abs_E)
            def sign2(E):
                return expit(-E / np.median(abs(E)) / 3)
            def sign(E):
                e = 1
                return (1 - E / np.sqrt(e*e + E*E))/2

            if baseline:
                bs, _ = smooth.whittaker_smooth_weight_func2(
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
            plt.legend()
            plt.show()
    #
    