#
# objects.py
#

import numpy as np
import rampy

import matplotlib.pyplot as plt
import ipywidgets

from irsa.preprocess import utils
import irsa.analytics as analytics

import mlgrad.funcs as funcs
import mlgrad.pca as pca
from mlgrad.af import averaging_function
import mlgrad.smooth as smooth
import mlgrad.inventory as inventory
import mlgrad.array_transform as array_transform
import scipy.special as special
from IPython.display import display

from mlgrad.af import averaging_function

def dist(S, X, c=0):
    return np.sqrt([((S@x)@x) for x in (X-c)])

def dist2(S, X, c=0):
    return np.array([((S@x)@x) for x in (X-c)])

def scale_min(x, alpha=0.01):
    if x < 0:
        return (1+alpha)*x
    else:
        return (1-alpha)*x

def scale_max(x, alpha=0.01):
    if x < 0:
        return (1-alpha)*x
    else:
        return (1+alpha)*x

def smooth_spectra(xs, ys, tau2=1.0, windows=None, beta=100):
    ys_smooth = smooth.whittaker_smooth(ys, tau2=tau2, d=2) 

    ee = ys - ys_smooth
    ee /= ee.std()

    W2 = np.exp(-0.5 * ee * ee)[1:-1]
    if windows is not None:
        for xa, xb in windows:
            W2[(xs[1:-1] >= xa) & (xs[1:-1] <= xb)] *= beta
    
    ys_smooth = smooth.whittaker_smooth(ys, tau2=tau2, W2=W2, d=2)
    return ys_smooth

_attr_names = (
    "вид_бактерий", "штамм_бактерий", "резистентность", 
    "отсечки_по_молекулярной_массе", "начальная_концентрация_клеток_в_пробе", 
    'номер_цикла', 'номер_эксперимента_в_цикле', 
    'номер_повтора', 'тип_измерения_спектров',
    'начальная_концентрация_клеток_в_пробе', 
    'капля', 'вода', 'отмывка_фильтров', "дата", 
    "комментарий"
)

class SpectraSeries:
    #
    def __init__(self, x, y, attrs):
        self.x = x
        self.y = y
        self.attrs = attrs
        key = "_".join(
            (attrs[k] if attrs[k] != 'no_date' else '_') for k in _attr_names if k in attrs)
        self.key = key
        self.excludes = []
        self.tau2s = None
        self.bs = None
        self.windows = []
    #
    def __getitem__(self, i):
        sp = Spectra(self.x[i], self.y[i], self.attrs)
        sp.key = self.key
        sp.windows = self.windows
        return sp
    #
    def __iter__(self):
        for i in range(self.x):
            yield self[i]
    #
    def select_for_exclusion(self, sigma_mu=0.10, alpha=3.5, clear=False):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.y)
        if self.excludes and clear:
            self.excludes = []
        if not self.excludes:
            for i in range(N):
                Ys = self.y[i]
                mu = inventory.robust_mean_2d_t(Ys, tau=alpha)
                std = np.sqrt(inventory.robust_mean_2d_t((Ys - mu)**2, tau=alpha))
                if (mu == 0).sum() == 0:
                    ss = std / mu
                    if ss.mean() >= sigma_mu:
                        self.excludes.append(i)
        
        i_slider = ipywidgets.IntSlider(value=0, min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
        
        b_exclude = ipywidgets.Checkbox(value=(i_slider.value in self.excludes))
        
        def i_on_value_change(change):
            i = i_slider.value
            b_exclude.value = (i in self.excludes)

        def b_on_value_change(change):
            i = i_slider.value
            if b_exclude.value:
                if i not in self.excludes:
                    self.excludes.append(i)
            else:
                if i in self.excludes:
                    self.excludes.remove(i)
        
        i_slider.on_trait_change(i_on_value_change, name="value")
        b_exclude.on_trait_change(b_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, exclude=b_exclude, continuous_update=False)
        def _plot_select_for_exclusion(i, exclude):
            fig = plt.figure("_select_for_exclusion", figsize=(10,4))
            fig.clear()
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(f"{self.key} ({len(self.y[i])} spectra)", fontdict={"size":10}, loc="left")
            xs = self.x[i]
            Ys = self.y[i]
            for ys in Ys:
                plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            mu = inventory.robust_mean_2d_t(Ys, tau=alpha)
            sigma = np.sqrt(inventory.robust_mean_2d_t((Ys - mu)**2, tau=alpha))
            if (sigma == 0).sum() == 0:
                ss = sigma / mu
    
                plt.fill_between(xs, mu-2*sigma, mu+2*sigma, alpha=0.5, 
                                 label=fr"$\sigma/\mu={ss.mean():.3f}\pm{ss.std():.3f}$")
            
            plt.plot(xs, mu, linewidth=1.0, color='DarkRed', label="robust mean")

            plt.minorticks_on()
            plt.tight_layout()
            plt.legend()
            plt.show()
    #
    def exclude_selected(self):
        if not self.excludes:
            return

        N = len(self.y)

        self.x_excluded = [self.x[i] for i in range(N) if i in self.excludes]
        self.x_excluded = [self.y[i] for i in range(N) if i in self.excludes]
        self.i_excluded = self.excludes

        self.x = [self.x[i] for i in range(N) if i not in self.excludes]
        self.y = [self.y[i] for i in range(N) if i not in self.excludes]

        self.excludes = []
    #
    def select_baselines(self, tau2=1.0e5, tau2_smooth=10, func=None, func2=None, beta=100, func2_mode="e"):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.y)

        # if self.tau2s is None:
        #     self.tau2s = np.full(N, tau2, "d")
        
        i_series_slider = ipywidgets.IntSlider(value=0, min=0, max=N-1)
        i_series_slider.layout.width="50%"

        @ipywidgets.interact(i_series=i_series_slider, continuous_update=False)
        def _plot_select_baselines(i_series):
            nonlocal tau2, tau2_smooth, func, func2, beta
            spcol = self[i_series]

            Ys = self.y[i_series]

            # if func is None:
            #     mu = inventory.robust_mean_2d_t(Ys, 3.5)
            #     sigma2 = inventory.robust_mean_2d_t((Ys - mu)**2, 3.5)
            #     func = funcs.Step(np.sqrt(sigma2.mean()))
            
            spcol.select_baselines(tau2=tau2, tau2_smooth=tau2_smooth, func=func, func2=func2, beta=beta, func_mode='e')            
    #
    def replace_spectra_with_corrected(self):
        for i in range(len(self.y)):
            sp = self[i]
            sp.replace_spectra_with_corrected()
    #
    def plot_spectra(self, tau2_smooth=1.0):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_series_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_series_slider.layout.width="50%"

        @ipywidgets.interact(i_series=i_series_slider, continuous_update=False)
        def _plot_spectra(i_series):
            sp = self[i_series]
            sp.plot_spectra(tau2_smooth=tau2_smooth)
    #
    def plot_pca_spectra(self, n_component=None):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
        
        xrange_slider = ipywidgets.IntRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="90%"
        
        # i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, xrange=xrange_slider, continuous_update=False)
        def _plot_pca_spectra(i, xrange):
            # plt.close("_plot_spectra_series")
            fig = plt.figure("_plot_spectra_series", figsize=(12,5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()

            Ys = self.y[i]
            Ys2 = Ys / 100 

            nonlocal n_component
            if n_component is None:
                n_component = Ys.shape[0]
            c, As, Ls = pca.find_loc_and_pc(Ys2, n_component)
            print(Ls/Ls.sum())
            print(As)
            plt.subplot(1,2,1)
            plt.title(f"{self.key} ({len(self.y[i])} spectra)")
            plt.minorticks_on()
            plt.grid(1)
            plt.plot(Ls/Ls.sum(), marker='o')
            plt.subplot(1,2,2)
            plt.scatter((Ys2 - c) @ As[0], (Ys2 - c) @ As[1])
            plt.minorticks_on()

            # xa, xb = xrange
            # ix_range = np.argwhere((xs >= xa) & (xs <= xb)).ravel()
            # i0, i1 = min(ix_range), max(ix_range)
            # plt.legend()
            plt.tight_layout()
            plt.show()            
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
    def plot_zscore_hist(self):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
                
        # def i_on_value_change(change):
        #     i = i_slider.value
        
        # i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_zscore_hist_items(i=i_slider):
            xs = self.x[i]
            Ys = self.y[i]

            j_slider = ipywidgets.IntSlider(min=0, max=len(xs)-1, value=0)
            j_slider.layout.width="50%"
                    
            # def j_on_value_change(change):
            #     j = j_slider.value
            
            # j_slider.on_trait_change(i_on_value_change, name="value")

            plt.close("_zscore_hist_i")
            fig = plt.figure("_zscore_hist_i", figsize=(12,3))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(f"{self.key} ({len(self.x[i])} spectra)")
            for ys in Ys:
                plt.plot(xs, ys, linewidth=0.75, alpha=0.5)

            # plt.plot(xs, ys_m, linewidth=1.5, color='k', label="current")
            plt.xlabel("j")
            plt.show()
            
            @ipywidgets.interact(j=j_slider, continuous_update=False)
            def _plot_zscore_hist(j):
                plt.close("_zscore_hist_j")
                fig = plt.figure("_zscore_hist_j", figsize=(10,3))
                fig.canvas.header_visible = False
                fig.canvas.footer_visible = False
                fig.canvas.toolbar_position = 'right'
                plt.title(f"Modified z-score histogram")

                ys_j = Ys[:,j]
                plt.hist(inventory.modified_zscore(ys_j), 
                         rwidth=0.95, 
                         bins=range(-10,10))
                ymin, ymax = plt.ylim()
                plt.vlines([-3.5, 3.5], 0, ymax, colors="r")
                
                plt.xticks(list(range(-10,10)))
                plt.minorticks_on()
                # plt.grid(1)
                plt.show()            
    #
    def plot_zscore(self, kind="m-zscore"):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
                
        # def i_on_value_change(change):
        #     i = i_slider.value
        
        # i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_zscore_items(i=i_slider):
            xs = self.x[i]
            Ys = self.y[i]

            plt.close("_zscore_i")
            fig = plt.figure("_zscore_i", figsize=(10,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(f"Modified z-score: {self.key} ({len(self.x[i])} spectra)")

            mu = inventory.robust_mean_2d_t(Ys, 3.5)
            plt.plot(xs, 5+3*mu/max(mu), color="Gray", alpha=0.5)
            
            Zs = np.empty_like(Ys)
            for j in range(len(xs)):
                ys_j = Ys[:,j]
                if kind == "zscore":
                    # mu = np.mean(ys_j)
                    # ss = np.std(ys_j)
                    Zs[:,j] = inventory.zscore(Ys[:,j])
                    plt.hlines([-3.0,3.0], min(xs), max(xs), colors='r', linestyles="--")
                elif kind == "m-zscore":
                    # mu = np.median(ys_j)
                    # ss = np.median(abs(ys_j - mu))
                    Zs[:,j] = inventory.modified_zscore(Ys[:,j])
                    plt.hlines([-3.5,3.5], min(xs), max(xs), colors='r', linestyles="--")
                else:
                    raise TypeError("invalid kind of zscore")
            for zs in Zs:
                plt.scatter(xs, zs, s=2, c='k', alpha=0.5)

            # plt.plot(xs, inventory.robust_mean_2d_t(Zs, 3.5), color="g", linewidth=1.5)

            plt.minorticks_on()
            plt.show()      
    #
    def plot_sigma_mu(self):
        import matplotlib.pyplot as plt
        import ipywidgets
        import mlgrad.smooth as smooth

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
                
        # def i_on_value_change(change):
        #     i = i_slider.value
        
        # i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_sigma_mu(i=i_slider):
            xs = self.x[i]
            Ys = self.y[i]

            plt.close("_sigma_mu_i")
            fig = plt.figure("_sigma_mu_i", figsize=(10,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(rf"$100\cdot\sigma/\mu$: {self.key} ({len(self.x[i])} spectra)")
            mu = inventory.robust_mean_2d_t(Ys, 3.5)
            sigma = inventory.robust_mean_2d_t(abs(Ys - mu), 3.5)
            sigma_mu = 100*sigma/mu
            plt.scatter(xs, sigma_mu, s=2, c='k', label=r"$\sigma/\mu$")
            ss = smooth.whittaker_smooth(sigma_mu, tau2=1.0e5)
            plt.plot(xs, ss, color="b", linewidth=1.5, label=r"smoothed $\sigma/\mu$")
            
            plt.plot(xs, 1+mu/max(mu), color="Gray", alpha=0.5, label="current spectrum")
            

            plt.hlines([0.5,1], min(xs), max(xs), colors='r', linewidths=0.5, linestyles="--")
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()            
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
    def align_bottom(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] -= y.min()
    #
    def scale_pca(self):
        from np.linalg import det
        for ys in self.y:
            Y = ys @ ys.T
            d = det(Y)
    #
    def scale_zs(self, delta=3.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                z = abs(inventory.modified_zscore(y))
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
    def scale_by_min(self, scale=1.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = (y / y.min()) * scale
    #
    def scale_by_max(self, scale=1.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = (y / y.max()) * scale
    #
    def smooth(self, tau=1.0, windows=None, beta=10.):
        Ys = self.y
        W2 = None
        if windows is not None:
            W2 = np.full(tau, len(self.y[0]), "d")
            for wa,wb in windows:
                W2[wa:wb+1] *= beta
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = smooth.whittaker_smooth_func2(y, W2=W2, tau=tau, d=2)[0]
    #
    def remove_overflow_spectra(self, y_max=2000.0, y_max_count=10):
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
    def remove_by_zscore_spectra(self, tau=3.5, max_count=40):
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
    def remove_outlier_spectra(self, delta=0.10, tau=3.0, max_count=30):
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
    def robust_averaging(self, tau=3.0):
        if len(self.y[0].shape) == 1:
            raise TypeError("Усреднять можно только в сериях")

        Ys = self.y.copy()
        # dYs = []
        for k in range(len(Ys)):
            ys_k = np.ascontiguousarray(Ys[k])
            # ys_k = Ys[k]
            ys = inventory.robust_mean_2d_t(ys_k, tau=tau)
            # dys = np.sqrt(inventory.robust_mean_2d_t((ys_k - ys)**2, tau=tau))
            # dYs.append(dys)
            Ys[k] = ys

        Ys = np.ascontiguousarray(Ys)
        Xs = np.ascontiguousarray(self.x)

        o = Spectra(Xs, Ys, self.attrs)
        o.key = self.key
        # o.stderr = np.ascontiguousarray(dYs)
        return o
    #

class Spectra:
    #
    def __init__(self, x, y, attrs):
        self.x = x
        self.y = y
        self.attrs = attrs
        self.bs = np.zeros_like(y)
        self.params = {}
        self.windows = []
        if len(x.shape) > 1:
            self.ensure_xs()
            self.x = self.x[0]
        self.excludes = []
        self.tau2_values = np.zeros_like(self.x)
        self.tau2s = None
        self.bs = np.zeros_like(self.y)
        self.ys_bs = np.zeros_like(self.y)
        self.ys_sm = y.copy()
    #
    def ensure_xs(self):
        iter_xs = iter(self.x)
        x0 = next(iter_xs)
        for x in iter_xs:
            if not np.all(x == x0):
                raise TypeError("some xs not equal")
    #
    def crop(self, start_index, end_index=None):
        if end_index is None:
            self.x = self.x[start_index:]
            self.y = self.y[:,start_index:]
        else:
            self.x = self.x[start_index:end_index]
            self.y = self.y[:,start_index:end_index]
    #
    def replace_small_values(self, delta, value=0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            np.putmask(ys, abs(ys)<delta, value)
    #
    def allign_bottom(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            ys[:] = ys - ys.min()
    #
    def apply_func(self, y_func, x_func=None, a=1, b=0):
        Ys = self.y
        xs = self.x
        for k in range(len(Ys)):
            ys = Ys[k]
            ys[:] = y_func(a*ys + b)
        if x_func is not None:
            for k in range(len(Xs)):
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
    def select_for_exclusion(self, clear=False):

        N = len(self.y)
        if self.excludes and clear:
            self.excludes = []

        i_slider = ipywidgets.IntSlider(value=0, min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        b_exclude = ipywidgets.Checkbox(value=(i_slider.value in self.excludes))

        button = ipywidgets.Button(description="Exclude selected")
        def button_click(_):
            self.exclude_selected()
        button.on_click(button_click)
        display(button)

        def i_on_value_change(change):
            i = i_slider.value
            b_exclude.value = (i in self.excludes)

        def b_on_value_change(change):
            i = i_slider.value
            if b_exclude.value:
                if i not in self.excludes:
                    self.excludes.append(i)
            else:
                if i in self.excludes:
                    self.excludes.remove(i)

        i_slider.on_trait_change(i_on_value_change, name="value")
        b_exclude.on_trait_change(b_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, exclude=b_exclude, continuous_update=False)
        def _plot_select_for_exclusion(i, exclude):
            fig = plt.figure("_select_for_exclusion", figsize=(10,4))
            fig.clear()
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(f"{self.key} ({len(self.y[i])} spectra)", fontdict={"size":10}, loc="left")
            xs = self.x
            Ys = self.y
            for ys in Ys:
                plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            plt.plot(xs, Ys[i], linewidth=1.0, color='DarkRed', label=f"current: {i}")

            plt.minorticks_on()
            plt.tight_layout()
            plt.legend()
            plt.show()
    #
    def exclude_selected(self):
        if not self.excludes:
            return

        N = len(self.y)

        # self.x_excluded = [self.x[i] for i in range(N) if i in self.excludes]
        self.y_excluded = [self.y[i] for i in range(N) if i in self.excludes]
        self.i_excluded = self.excludes

        # self.x = [self.x[i] for i in range(N) if i not in self.excludes]
        self.y = [self.y[i] for i in range(N) if i not in self.excludes]

        self.excludes = []
    #
    def select_windows(self, windows=None):
        import matplotlib.pyplot as plt
        import ipywidgets

        ws_select = ipywidgets.Select(options=[str(w) for w in self.windows], description="windows:")

        xrange_slider = ipywidgets.IntRangeSlider(
            value=(min(self.x), max(self.x)), 
            min=min(self.x), 
            max=max(self.x),
            description="xrange:")
        xrange_slider.layout.width="50%"

        # add_button = ipywidgets.Button(description="Add")

        # def onclick_options_button(b):
        #     self.windows.append(self.xrange_slider.value)
        #     ws.options = [str(w) for w in self.windows]

        b_include = ipywidgets.Checkbox(value=False)

        # def i_on_value_change(change):
        #     i = i_slider.value
        #     b_exclude.value = self.excludes[i]

        def b_on_value_change(change):
            flag = b_include.value
            xrange_value = xrange_slider.value
            if flag:
                self.windows.append(xrange_value)
                ws_select.options += (str(xrange_value),)
                b_include.value = False
            # else:
            #     self.windows.remove(xrange_value)

        # xrange_slider.on_trait_change(i_on_value_change, name="value")
        b_include.on_trait_change(b_on_value_change, name="value")

        @ipywidgets.interact(xrange=xrange_slider, ws=ws_select, b=b_include, continuous_update=False)
        def _plot_select_windows(xrange, ws, b):
            fig = plt.figure("_select_windows", figsize=(13,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            y_max = self.y.max()
            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.75, alpha=0.5)

            plt.vlines(xrange, 0, y_max, color='LightGreen')
    
            for xa, xb in self.windows:
                ii = np.argwhere((xs >= xa) & (xs <= xb)).ravel()
                i0, i1 = min(ii), max(ii)
                plt.fill_between(xs[i0:i1], 0, np.max(self.y[:,i0:i1], axis=0), color='LightBlue')
    #    
    def scale_by_robust_mean(self, tau=3.0, scale=1.0):
        N = len(self.y)
        for k in range(N):
            ys = self.y[k]
            # err = self.stderr[k]
            mu = inventory.robust_mean_1d(ys, tau=tau)
            ys[:] = (ys / mu) * scale
            # err[:] = (err / mu) * scale
    #    
    def scale_by_max(self, scale=1.0):
        N = len(self.y)
        for k in range(N):
            ys = self.y[k]
            # err = self.stderr[k]
            mu = ys.max()
            ys[:] = (ys / mu) * scale
            # err[:] = (err / mu) * scale
    #
    # def remove_outlier_spectra(self, tau=3.0):
    #     Xs, Ys = self.x, self.y

    #     Is = []
    #     for i, ys in enumerate(Ys):
    #         if np.any(np.isnan(ys)):
    #             continue
    #         Is.append(i)
        
    #     utils.mark_outliers2(Ys, tau=tau)

    #     Is = []
    #     for i, ys in enumerate(Ys):
    #         if np.any(np.isnan(ys)):
    #             continue
    #         Is.append(i)

    #     if Is:
    #         Is = np.array(Is)
    #         Ys = Ys[Is]
    #         Xs = Ys[Is]

    #     self.x, self.y = Xs, Ys
    #
    # def replace_outlier_spectra(self, tau=3.0):
    #     Xs, Ys = self.x, self.y
    #     Ys = np.array(Ys)
        
    #     utils.replace_outliers2(Ys, tau=tau)
        
    #     self.y = Ys
    #
    def smooth(self, method="irsa2", tau=1.0, **kwargs):
        xs = self.x
        Ys = self.y
        if method == "runpy":
            for k in range(len(Ys)):
                ys = Ys[k]
                ys[:] = rampy.smooth(xs, ys, Lambda=tau)
        elif method == "irsa":
            for k in range(len(Ys)):
                ys = Ys[k]
                ys[:] = smooth.whittaker_smooth(ys, tau=tau)
        elif method == "irsa2":
            func = kwargs.get("func", funcs.Square())
            # func2 = kwargs.get("func2", None)
            for k in range(len(Ys)):
                ys = Ys[k]

                diff2_i = array_transform.array_diff2(ys)
                mu_i = np.median(diff2_i)
                dd2_i = (np.percentile(diff2_i, 75) - np.percentile(diff2_i, 25)) / 2
                # dd2_k = np.median(abs(diff2_k - mu_k))
                
                def smooth_func2(x, mu=mu_k, scale=dd2_k):
                    v = (x - mu)/scale
                    return 1/(1 + v*v)
                
                ys[:], _ = smooth.whittaker_smooth_weight_func2(
                                ys, tau=tau,
                                func=func, 
                                func2=smooth_func2)
        # np.putmask(ys, ys < 0, 0)
    #
    def smooth_by_windows(self, tau=0.1, windows=None, beta=1000.):
        Ys = self.y
        W2 = None
        xs = self.x
        if windows is not None:
            W2 = np.full(len(self.y[0]), 1.0, "d")
            for xa,xb in windows:
                W2[(xs >= xa) & (xs <= xb)] *= beta
            for k, y in enumerate(self.y):
                y[:] = smooth.whittaker_smooth(y, W2=W2, tau2=tau, d=2)
    #
    def select_baselines(self, tau2=1000.0, tau1=0.0, tau_z=0, tau2_smooth=1.0, override_tau2=True,
                         bs_scale=3.0, alpha=0.001, eps=0.001, beta=1.0e2,
                         func=None, func2=None, func2_e=None, w_tau2=1.0,
                         d=2, **kwargs):

        N = len(self.y)

        xs = self.x

        if override_tau2:
            for i in range(N):
                self.tau2_values[i] = tau2
        else:
            mask = (self.tau2_values == 0)
            if mask.sum() > 0:
                self.tau2_values[mask] = tau2

        # self.tau1_values = N * [tau1]

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        tau2_0 = self.tau2_values[0]
        tau2_slider = ipywidgets.FloatSlider(value=tau2_0, min=tau2_0/10, max=tau2_0*10, step=tau2_0/50)
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
            tau2 = tau2_slider.value = self.tau2_values[i]
            tau2_slider.min = tau2 / 10
            tau2_slider.max = tau2 * 10
            tau2_slider.step = tau2_slider.min / 50
            # tau1_slider.value = self.tau1_values[i]

        def xrange_on_value_change(change):
            xmin, xmax = xrange_slider.value
            # plt.xlim(xmin, xrange)
            # i = i_slider.value
            # ys = self.y[i]
            # ys = ys[self.x >= xmin & self.x <= xmax]
            # plt.ylim(0.9*min(ys), 1.1*max(ys))

        tau2_slider.on_trait_change(tau2_on_value_change, name="value")
        # tau1_slider.on_trait_change(tau1_on_value_change, name="value")
        i_slider.on_trait_change(i_on_value_change, name="value")

        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x), max(self.x)), 
            min=min(self.x), 
            max=max(self.x))
        xrange_slider.layout.width="80%"

        button = ipywidgets.Button(description="Subtract baselines")
        def button_click(_):
            self.replace_spectra_with_corrected()
        button.on_click(button_click)
        display(button)

        @ipywidgets.interact(i=i_slider, tau2=tau2_slider, #tau1=tau1_slider, 
                             xrange=xrange_slider, continuous_update=False)
        def _plot_select_baselines(i, tau2, xrange):
            fig = plt.figure("_select_baselines", figsize=(13,7))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            ax1 = plt.subplot(2,1,1)
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.4, alpha=0.5)

            ys_i = self.y[i]

            W2 = np.ones(len(self.y[0]), "d")
            if self.windows is not None:
                for xa,xb in self.windows:
                    W2[(xs >= xa) & (xs <= xb)] = beta

            nonlocal func
            ys_i_smooth = smooth.whittaker_smooth(
                    ys_i, tau2=tau2_smooth,
                    # W=W,
                    W2=W2,
                    d=2)
            sigma = (ys_i - ys_i_smooth).std()

            plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkGreen', label=f"current smoothed ({i})")
            plt.plot(xs, ys_i, linewidth=1.5, color='DarkBlue', label=f"current ({i})")

            if func is None:
                func = funcs.Step(sigma)
            bs, dd = smooth.whittaker_smooth_weight_func2(
                ys_i_smooth,
                func=func,
                func2=func2,
                func2_e=func2_e,
                tau2=tau2,
                tau_z=tau_z,
                d=d)

            self.bs[i,:] = bs
            self.ys_bs[i,:] = self.y[i] - bs

            plt.plot(xs, bs, linewidth=1.0, color='m', label="baseline")

            x_min, x_max = xrange
            plt.xlim(0.99*x_min, 1.01*x_max)

            # bs_xrange = bs[(x_min <= xs) & (xs <= x_max)]
            # bs_max, bs_min = np.max(bs_xrange), np.min(bs_xrange)
            ys_xrange = ys_i[(x_min <= xs) & (xs <= x_max)]
            plt.ylim(scale_min(ys_xrange.min()), scale_max(ys_xrange.max()))
            plt.minorticks_on()
            plt.grid(1)
            plt.legend()
            plt.tight_layout()
            # plt.show()

            # fig = plt.figure("_draw_baselines", figsize=(12,3))
            # fig.canvas.header_visible = False
            # fig.canvas.footer_visible = False
            # fig.canvas.toolbar_position = 'right'            
            # fig.clear()
            # plt.title(self.key, fontdict={"size":10}, loc="left")
            ax2 = plt.subplot(2,1,2, sharex=ax1)
            for ys_bs in self.ys_bs:
                plt.plot(self.x, ys_bs, linewidth=0.4, alpha=0.5)
            plt.plot(self.x, self.ys_bs[i], linewidth=1.0, color='DarkBlue')
            plt.xlim(0.99*x_min, 1.01*x_max)
            # plt.ylim(0, np.max(ys_i))
            plt.minorticks_on()
            plt.grid(1)
            ax1.label_outer()
            ax2.label_outer()
            plt.tight_layout()
            plt.show()
    #
    def select_baseline_param(self,
                         tau2=1000.0,
                         func=None, func2=None, func2_e=None,
                         d=2,
                         override_tau2=False, 
                         tau_z=0, tau2_smooth=1.0, beta=100,
                         **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets

        xs = self.x

        if override_tau2 or not self.tau2s:
            self.tau2s = 3*[tau2]            
        
        tau2_low_slider = ipywidgets.FloatSlider(value=self.tau2s[0], 
                                                 min=self.tau2s[0]/50, 
                                                 max=self.tau2s[0]*25, 
                                                 step=self.tau2s[0]/100)
        tau2_low_slider.layout.width="80%"   
        tau2_mid_slider = ipywidgets.FloatSlider(value=self.tau2s[1], 
                                                min=self.tau2s[1]/50, 
                                                max=self.tau2s[1]*25, 
                                                step=self.tau2s[1]/100)
        tau2_mid_slider.layout.width="80%"   
        tau2_high_slider = ipywidgets.FloatSlider(value=self.tau2s[2], 
                                                 min=self.tau2s[2]/50, 
                                                 max=self.tau2s[2]*25, 
                                                 step=self.tau2s[2]/100)
        tau2_high_slider.layout.width="80%"   

        def tau2_low_on_value_change(change):
            self.tau2s[0] = tau2_low_slider.value
        def tau2_mid_on_value_change(change):
            self.tau2s[1] = tau2_mid_slider.value
        def tau2_high_on_value_change(change):
            self.tau2s[2] = tau2_high_slider.value
        
        tau2_low_slider.on_trait_change(tau2_low_on_value_change, name="value")
        tau2_high_slider.on_trait_change(tau2_high_on_value_change, name="value")
        tau2_mid_slider.on_trait_change(tau2_mid_on_value_change, name="value")

        ys_high = np.quantile(self.y, 0.83333, axis=0)
        ys_mid = np.quantile(self.y, 0.5, axis=0)
        ys_low = np.quantile(self.y, 0.16666, axis=0)
        
        self.Ys = np.vstack((ys_low, ys_mid, ys_high))
        labels = ["low", "mid", "high"]
        
        W2 = np.full(len(ys_high), 10.0, "d")
        if self.windows is not None:
            for xa,xb in self.windows:
                W2[(xs >= xa) & (xs <= xb)] = beta
                
        Ys_smooth = []
        # scale = ys_mid.mean()
        for ys in self.Ys:
            ys_smooth = smooth.whittaker_smooth(
                ys, tau2=tau2_smooth, 
                W2=W2, 
                d=2)
            Ys_smooth.append(ys_smooth)
        
                
        @ipywidgets.interact(tau2_high=tau2_high_slider, 
                             tau2_mid=tau2_mid_slider, 
                             tau2_low=tau2_low_slider, 
                             continuous_update=False)
        def _plot_select_baseline_param(tau2_high, tau2_mid, tau2_low):
            nonlocal func
            
            fig = plt.figure("select_baseline_param", figsize=(12,5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'            
            fig.canvas.toolbar_visible = True            
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            
            colors=["DarkRed", "DarkGreen", "DarkBlue"]
            colors2=["Chocolate", "Lime", "DodgerBlue"]
            for i in range(3):
                plt.plot(xs, self.Ys[i], linewidth=1.25, color=colors[i], label=labels[i])
            
            Bs = []
            for i in range(3):
                if func is None:
                    sigma = abs(self.Ys[i] - Ys_smooth[i]).mean()      
                    func = funcs.Step(sigma) 
                
                bs, _ = smooth.whittaker_smooth_weight_func2(
                    Ys_smooth[i], 
                    func=func,
                    func2=func2,
                    tau2=self.tau2s[i], 
                    tau_z=tau_z,
                    d=d, 
                    func2_mode=func2_mode)
                Bs.append(bs)
    
            for i in range(3):
                if i == 0:
                    plt.plot(xs, Bs[i], linewidth=1.25, color=colors2[i], label="baseline")                
                    # plt.plot(xs, self.Ys[i]-Bs[i], linewidth=0.75, color='Grey', label="corrected")
                else:
                    plt.plot(xs, Bs[i], linewidth=1.25, color=colors2[i])
                    # plt.plot(xs, self.Ys[i]-Bs[i], linewidth=0.75, color='Grey')
            
            self.Bs = Bs
                        
            plt.minorticks_on()
            plt.grid(1)
            plt.legend()
            plt.tight_layout()
            plt.show()
    #
#     def get_baselines(self, kind="irsa", **kwargs):
#         import pybaselines
        
#         xs = self.x
#         Ys = self.y

#         N = len(Ys)
        
#         if hasattr(self, "params"):
#             params = self.params
#         else:
#             self.params = params = {}

#         if hasattr(self, "bs"):
#             Bs = self.bs
#         else:        
#             self.bs = Bs = np.zeros((N, len(Ys[0])))

#         Ls = params.get("lam", None)
#         if Ls is None:
#             Ls = N * [lam]
#             params["lam"] = Ls
            
#         for k in range(len(Ys)):
#             ys = Ys[k]
#             lam = Ls[k]

#             if kind == "aspls":
#                 diff_order = kwargs.get("diff_order", 2)
#                 bs, _ = pybaselines.whittaker.aspls(ys, x_data=xs, 
#                                                      lam=lam, diff_order=diff_order,
#                                                      **kwargs)
#             elif kind == "arpls":
#                 diff_order = kwargs.get("diff_order", 2)
#                 bs, _ = pybaselines.whittaker.arpls(ys, x_data=xs,
#                                                      lam=lam, diff_order=diff_order, 
#                                                      **kwargs)
#             elif kind == "mor":
#                 bs, _ = pybaselines.morphological.mor(ys, x_data=xs, **kwargs)

#             elif kind == "irsa":
#                 func = kwargs["func"]
#                 func2 = kwargs["func2"]
#                 bs, _ = smooth.whittaker_smooth_weight_func2(
#                             ys, 
#                             func=func, 
#                             func2=func2, 
#                             tau2=lam, d=2)
        
#             Bs[k,:] = bs

#         return Bs
    #
    def subtract_selected_baselines(self):
        self.ys_bs = self.y - self.bs
    #
    def subtract_baselines(self, func=None, func2=None, d=2, func2_mode="d", tau2_smooth=10, beta=100, tau_z=0):
        import ipywidgets
        from IPython.display import display
        
        N = len(self.y)
        fp = ipywidgets.FloatProgress(min=0, max=N, value=0, bar_style="success", description=self.key)
        fp.style.width = 50
        display(fp)
        
        W2 = np.full(len(self.y[0]), 10.0, "d")
        if self.windows is not None:
            for xa,xb in self.windows:
                W2[(xs >= xa) & (xs <= xb)] = beta

        scale = self.Ys[1].mean()
        
        for i, ys_i in enumerate(self.y):
            
            j = abs(self.Ys - ys_i).sum(axis=1).argmin()

            # ys_i /= scale
            ys_smooth = smooth.whittaker_smooth(
                ys_i, tau2=tau2_smooth, 
                W2=W2, 
                d=2)
        
            if func is None:
                sigma = abs(ys_i - ys_smooth).mean()
                func = funcs.Step(sigma)        
            
            bs, _ = smooth.whittaker_smooth_weight_func2(
                ys_smooth, 
                func=func,
                func2=func2,
                tau2=self.tau2s[j], 
                tau_z=tau_z,
                d=d, 
                func2_mode=func2_mode)    

            # ys_i *= scale
            # bs *= scale
            self.bs[i,:] = bs
            self.ys_bs[i,:] = ys_i - bs

            fp.value = i+1
        fp.close()
    #
    def replace_spectra_with_corrected(self):
        for i in range(self.y.shape[0]):
            self.y[i,:] = self.ys_bs[i]
        self.bs.fill(0)
        self.ys_bs.fill(0)
        self.tau2_mean=0
        self.tau2_values.fill(0)
    #
    def plot_spectra(self, smoothing=True, tau2_smooth=1, beta=100, **kwargs):
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1, description="i")
        i_slider.layout.width="50%"

        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x), max(self.x)), 
            min=min(self.x), 
            max=max(self.x),
            description="xrange",)
        xrange_slider.layout.width="50%"

        @ipywidgets.interact(i=i_slider, xrange=xrange_slider, continuous_update=False)
        def _plot_spectra(i, xrange):
            fig = plt.figure("_plot_current_spectra", figsize=(12,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            ys_i = self.y[i]
            plt.plot(xs, ys_i, linewidth=1.5, color='DarkBlue', 
                     label="current")

            if smoothing:
                ys_i_smooth = smooth_spectra(xs, ys_i, tau2=tau2_smooth, windows=self.windows, beta=beta)
                plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkRed', 
                         label="smoothed", alpha=0.75)
            else:
                y_i_smooth = y_i

            x_min, x_max = xrange
            ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
            ymin, y_max = 0.9*np.min(ys_range), 1.2*np.max(ys_range)
            plt.ylim(ymin, y_max)
            
            plt.minorticks_on()
            plt.tight_layout()
            plt.xlim(x_min, x_max)
            plt.legend()
            plt.show()
    #
    def plot_corrected_spectra(self, **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.x)-1)
        i_slider.layout.width="50%"
        
        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x), max(self.x)), 
            min=min(self.x), 
            max=max(self.x))
        xrange_slider.layout.width="50%"

        @ipywidgets.interact(i=i_slider, xrange=xrange_slider, continuous_update=False)
        def _plot_corrected_spectra(i, xrange):
            fig = plt.figure("_plot_corrected_spectra", figsize=(12,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'            
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            for ys_bs in self.ys_bs:
                plt.plot(xs, ys_bs, linewidth=0.5, alpha=0.25, color='k')

            ys_bs_i = self.ys_bs[i]
            plt.plot(xs, ys_bs_i, linewidth=1.0, color='m')
            
            # x_min, x_max = xrange
            # ys_range = ys_bs_i[(x_min <= xs) & (xs <= x_max)]
            # ymin, y_max = 0.9*np.min(ys_range), 1.2*np.max(ys_range)
            # plt.ylim(ymin, y_max)
            
            plt.minorticks_on()
            plt.tight_layout()
            # plt.xlim(x_min, x_max)
            # plt.legend()
            plt.show()
    #
    def plot_spectra_with_robust_smoothing(self, tau2=1.0, alpha=1, beta=1.0e2, **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
        
        alpha_slider = ipywidgets.FloatSlider(value=alpha, min=0.0, max=1.0, step=0.01)
        alpha_slider.layout.width="50%"   
        
        tau2_slider = ipywidgets.FloatSlider(value=tau2, min=0.0, max=10*tau2, step=tau2/100)
        tau2_slider.layout.width="50%"   
        
        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x), max(self.x)), 
            min=min(self.x), 
            max=max(self.x))
        xrange_slider.layout.width="50%"

        def i_on_value_change(change):
            i = i_slider.value
            tau2_slider.value = tau2
            alpha_slider.value = alpha
        
        i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, alpha=alpha_slider, tau2=tau2_slider, 
                             xrange=xrange_slider, continuous_update=False)
        def _plot_smooth_spectra(i, alpha, tau2, xrange):
            fig = plt.figure("_plot_smooth", figsize=(12,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_visible = True            
            fig.canvas.toolbar_position = 'right'            
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            ys_i = self.y[i]
            plt.plot(xs, ys_i, linewidth=1.5, color='DarkBlue', label="original")
            
            if alpha == 1.0:
                if self.windows is not None:
                    W2 = np.full(len(self.y[0])-2, 1.0, "d")
                    for xa,xb in self.windows:
                        mask = (xs >= xa) & (xs <= xb)
                        W2[mask] = beta
                
                ys_i_smooth = smooth.whittaker_smooth(
                        ys_i, 
                        tau2=tau2,
                        W2=W2, 
                        d=2)
                
                ee = ys_i_smooth - ys_i
                ee_mean = ee.mean()
                ee_std = ee.std()
                zs = (ee - ee_mean) / ee_std
                W = np.exp(0.5 * zs*zs)
                
                ys_i_smooth = smooth.whittaker_smooth(
                        ys_i, 
                        tau2=tau2,
                        W=W,
                        W2=W2, 
                        d=2)
            else:            
                ys_i_smooth, _ = smooth.whittaker_smooth_ex(
                        ys_i, 
                        aggfunc = averaging_function("WM", kwds={"alpha":alpha}),
                        tau2=tau2,
                        # W2=W2, 
                        d=2)

            dy = ys_i - ys_i_smooth
            mu = np.mean(dy)
            sigma = np.std(dy)
        
            plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkRed', 
                     label=fr"smoothed: $\tau_2={tau2}$ $\mu={mu:.2f}$ $\sigma={sigma:.2f}$")

            x_min, x_max = xrange
            ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
            ymin, y_max = 0.9*np.min(ys_range), 1.2*np.max(ys_range)
            plt.ylim(ymin, y_max)
            
            plt.minorticks_on()
            plt.tight_layout()
            plt.xlim(x_min, x_max)
            plt.legend()
            plt.show()
    #            
    def plot_pca_spectra(self, n_component=None, kind="AM", alpha=0.95):
        import matplotlib.pyplot as plt
        import ipywidgets
        from scipy.stats import chi2
        
        # xrange_slider = ipywidgets.IntRangeSlider(
        #     value=(min(self.x), max(self.x)), 
        #     min=min(self.x), 
        #     max=max(self.x))
        # xrange_slider.layout.width="90%"
        
        alpha_slider = ipywidgets.FloatSlider(value=alpha, min=0.0, max=1.0, step=0.01)
        alpha_slider.layout.width="50%"   

        # i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(alpha=alpha_slider, continuous_update=False)
        def _plot_pca_spectra(alpha):
            # plt.close("_plot_spectra_series")
            fig = plt.figure("_plot_spectra_series", figsize=(12,5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            Ys = self.y
            Ys2 = Ys / 100 

            nonlocal n_component
            if n_component is None:
                n_component = Ys.shape[0]
            if kind == "AM":
                c, As, Ls = pca.find_loc_and_pc(Ys2, n_component)
            else:
                wma = averaging_function(kind, kwds={'alpha':alpha})
                c, As, Ls = pca.find_robust_loc_and_pc(Ys2, wma, n_component)
            # U = np.c_[(Ys2 - c) @ As[0], (Ys2 - c) @ As[1]]
            Ls /= Ls.sum()
            with np.printoptions(precision=4, suppress=True):
                print(Ls)
            # print(As)
            plt.subplot(1,2,1)
            plt.title(f"{self.key} ({len(self.y)} spectra)")
            m = 1 + (Ls.cumsum() <= alpha).sum()
            plt.plot(Ls, marker='o', label=f"{m} components for level {alpha:.3f}")
            plt.minorticks_on()
            plt.grid(1)
            plt.legend()
            plt.subplot(1,2,2)
            if kind == "AM":
                c, A, Ls, U, S, d_max = analytics.find_loc_and_pc(Ys2, n=2, alpha=alpha)
            else:
                c, A, Ls, U, S, d_max = analytics.find_robust_loc_and_pc(Ys2, n=2, kind=kind, alpha=alpha)

            plt.scatter(U[:,0], U[:,1], c='w', edgecolors='k')
            x_min = scale_min(U[:,0].min())
            x_max = scale_max(U[:,0].max())
            y_min = scale_min(U[:,1].min())
            y_max = scale_max(U[:,1].max())

            # dy = y_max - y_min
            # dx = x_max - x_min

            UX, UY = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100))

            UXY = np.c_[UX.ravel(), UY.ravel()]
            ZZ = dist(S, UXY)
            # ZZ2 = chi2.cdf(ZZ, 2)

            ZZ = ZZ.reshape(UX.shape)

            plt.scatter([c[0]], [c[1]], s=144, c="k", marker='+', linewidth=1.0)
            with np.printoptions(precision=4, suppress=True):
                print("scatter matrix:\n", S)
            print(alpha)
            levels = [0.01, 0.1]

            ct = plt.contour(UX, UY, ZZ, colors="r", linestyles=':', linewidths=0.5, alpha=0.75)
            plt.clabel(ct, fontsize=10)

            plt.minorticks_on()

            # xa, xb = xrange
            # ix_range = np.argwhere((xs >= xa) & (xs <= xb)).ravel()
            # i0, i1 = min(ix_range), max(ix_range)
            # plt.legend()
            plt.tight_layout()
            plt.show()
    #

class SpectraCollection:
    #
    def __init__(self, spectra=None):
        if spectra is None:
            self.spectra = {}
        else:
            self.spectra = spectra
        self.wg_select_key = None
        self.current_key = None

        self.active_windows = []
    #
    def __getitem__(self, key):
        return self.spectra[key]
    #
    def __setitem__(self, key, val):
        self.spectra[key] = val
    #
    def __iter__(self):
        return iter(self.spectra)
    #
    def __len__(self):
        return len(self.spectra)
    #
    def keys(self):
        return self.spectra.keys()
    #
    def select_by_attr_value(self, name, val):
        for sp in self.spectra.values():
            if sp.attrs[name] == val:
                yield sp
    #
    def select_y_by_attr_value(self, name, val):
        for sp in self.spectra.values():
            if sp.attrs[name] == val:
                yield sp.y
    #
    def select_key(self):
        if self.wg_select_key is None:
            keys = list(self.keys())
            self.wg_select_key = wg_select = ipywidgets.Dropdown(options=keys, value=keys[0], description="Spectra key:")
            wg_select.layout.width="50%"
            self.current_key = keys[0]

            def on_value_change(change):
                self.current_key = self.wg_select_key.value

            wg_select.on_trait_change(on_value_change, name="value")
        else:
            wg_select = self.wg_select_key

        return wg_select
    #
    def align_bottom(self):
        for sp in self.spectra.values():
            sp.align_bottom()
    #
    def crop(self, start_index, end_index=None):
        for sp in self.spectra.values():
            sp.crop(start_index, end_index)
    #
    def plot_spectra(self):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_current_spectra(key):
            sp = self.spectra[key]
            sp.plot_spectra()
    #
    def select_for_exclusion(self):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_current_spectra(key):
            sp = self.spectra[key]
            sp.select_for_exclusion()
    #
    def select_baselines(self, tau2=1000.0, tau2_smooth=1.0,
                         func=funcs.Expit(-10.0), func2=None, func2_e=None,
                         d=2):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _select_baselines(key):
            sp = self.spectra[key]
            sp.select_baselines(tau2=tau2, tau2_smooth=tau2_smooth, func=func, func2=func2, func2_e=func2_e, d=d)
    #
    def save(self, root, tag):
        import os
        root_tag = f"{root}/{tag}"
        if not os.path.exists(root_tag):
            os.makedirs(root_tag)
        for key, sp in self.spectra.items():
            file_path = f"{root_tag}/{key}.txt"
            if os.path.exists(file_path):
                os.remove(file_path)
            with open(file_path, "wt") as f:
                for name, val in sp.attrs.items():
                    f.write(f"#{name}: {val}\n")
                f.write(f"#data: {len(sp.x)} {len(sp.y)}\n")
                xy = np.vstack((sp.x, sp.y),)
                np.savetxt(f, xy.T, fmt="%.3e")
                f.write("\n")
    #
    def load(self, root, tag):
        import os
        root_tag = f"{root}/{tag}"
        for fname in os.listdir(root_tag):
            file_path = f"{root_tag}/{fname}"
            if not fname.endswith(".txt"):
                continue
            with open(file_path, "rt") as f:
                attrs = {}
                while 1:
                    line = f.readline()
                    line = line.strip()
                    if not line.startswith("#"):
                        raise TypeError("Comment line start with #")
                    name, val = line[1:].split(":")
                    name = name.strip()
                    val = val.strip()
                    if name == "data":
                        n_row, n_col = tuple(int(x) for x in val.split(' '))
                        break
                    attrs[name] = val
                xy = np.loadtxt(f)
                # xy = xy.reshape(n_row, n_col)
                # print(xy.shape)
                x = xy[:,0]
                y = xy[:,1:].T
                x = np.ascontiguousarray(x)
                y = np.ascontiguousarray(y)
                attrs["source"] = file_path
                key = fname[:-4]
                attrs["key"] = key
                sp = Spectra(x, y, attrs)
                sp.key = key
                self[key] = sp
