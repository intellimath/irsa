#
# objects.py
#

import numpy as np
import rampy 
from irsa.preprocess import utils
import mlgrad.funcs as funcs
from mlgrad.af import averaging_function
import mlgrad.smooth as smooth
import mlgrad.inventory as inventory
import mlgrad.array_transform as array_transform
import scipy.special as special
from IPython.display import display

from mlgrad.af import averaging_function

class ExperimentSpectraSeries:
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
        self.excludes = []
    #
    def check_spectra(self, sigma_mu=0.10, alpha=3.5):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.y)
        if not self.excludes:
            for i in range(N):
                Ys = self.y[i]
                mu = inventory.robust_mean_2d_t(Ys, tau=alpha)
                std = np.sqrt(inventory.robust_mean_2d_t((Ys - mu)**2, tau=alpha))
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
        def _plot_spectra(i, exclude):
            fig = plt.figure("_check_spectra", figsize=(10,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(f"{self.key} ({len(self.x[i])} spectra)")
            xs = self.x[i]
            Ys = self.y[i]
            for ys in Ys:
                plt.plot(xs, ys, linewidth=0.5)

            mu = inventory.robust_mean_2d_t(Ys, tau=alpha)
            sigma = np.sqrt(inventory.robust_mean_2d_t((Ys - mu)**2, tau=alpha))
            ss = sigma / mu

            plt.fill_between(xs, mu-2*sigma, mu+2*sigma, alpha=0.5, 
                             label=fr"$\sigma/\mu={ss.mean():.3f}\pm{ss.std():.3f}$")
            plt.plot(xs, mu, linewidth=1.5, color='k', label="robust mean")

            plt.minorticks_on()
            plt.tight_layout()
            plt.legend()
            plt.show()
    #
    def exclude_checked(self):
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
    def plot_spectra(self):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        j_slider = ipywidgets.IntSlider(min=0, max=len(self.y[0])-1)
        j_slider.layout.width="50%"
        
        xrange_slider = ipywidgets.IntRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="90%"

        def i_on_value_change(change):
            j_slider.value = 0
        
        i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, j=j_slider, xrange=xrange_slider, continuous_update=False)
        def _plot_spectra(i, j, xrange):
            # plt.close("_plot_spectra_series")
            fig = plt.figure("_plot_spectra_series", figsize=(10,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            plt.title(f"{self.key} ({len(self.x[i])} spectra)")
            xs = self.x[i]
            Ys = self.y[i]
            for ys in Ys:
                plt.plot(xs, ys, linewidth=0.5)

            ys_m = inventory.robust_mean_2d_t(Ys, tau=3.5)
            std = np.sqrt(inventory.robust_mean_2d_t((Ys - ys_m)**2, tau=3.5))
            ss = std/ys_m

            plt.fill_between(xs, ys_m-2*std, ys_m+2*std, alpha=0.5, 
                             label=fr"$\sigma={std.mean():.0f}\ (\sigma/\mu={ss.mean():.3f}\pm{ss.std():.3f})$")
            plt.plot(xs, ys_m, linewidth=1.5, color='k', label="mean (robust)")
            plt.plot(xs, Ys[j], linewidth=1.0, color='DarkBlue', label=f"current ({j})")

            xa, xb = xrange
            ix_range = np.argwhere((xs >= xa) & (xs <= xb)).ravel()
            i0, i1 = min(ix_range), max(ix_range)
                
            plt.minorticks_on()
            plt.tight_layout()
            plt.ylim(0.95*np.min(Ys[:,i0:i1+1]), 1.05*np.max(Ys[:,i0:i1+1]))
            plt.xlim(*xrange)
            # plt.legend(loc="upper left")
            plt.legend()
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
    def allign_bottom(self):
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

        o = ExperimentSpectra(Xs, Ys, self.attrs)
        o.key = self.key
        # o.stderr = np.ascontiguousarray(dYs)
        return o
    #

# def array_rel_max(E):
#     abs_E = abs(E)
#     max_E = max(abs_E)
#     min_E = min(abs_E)
#     rel_E =  (abs_E - min_E) / (max_E - min_E)
#     return rel_E

# def array_expit_sym(E):
#     return special.expit(-E)
# def array_expit(E):
#     return special.expit(E)
# def array_sigmoid_pos(E):
#     return 2*special.expit(-E) - 1

# def array_sqrtit(E):
#     return (1 - E / np.sqrt(1 + E*E)) / 2
# def array_sqrtit_sym(E):
#     return (1 + E / np.sqrt(1 + E*E)) / 2

# def sqrt2(E):
#     # E /= np.median(np.abs(E))
#     return (1 + E / np.sqrt(1 + E*E)) / 2
# def gauss(E):
#     return 1-np.exp(-E*E/2)
    

class ExperimentSpectra:
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
        self.tau2_values = np.zeros_like(self.x)
        self.tau2s = None
        self.bs = np.zeros_like(y)
        self.ys_bs = np.zeros_like(y)
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
    # def remove_overflow_spectra(self, y_max=1500, y_max_count=100):
    #     Xs, Ys = self.x, self.y

    #     Is = []
    #     for i, ys in enumerate(Ys):
    #         if (ys >= 1500).sum() < y_max_count:
    #             continue
    #         Is.append(i)
        
    #     if Is:
    #         Is = np.array(Is)
    #         Ys = Ys[Is]
    #         Xs = Ys[Is]

    #     self.x, self.y = Xs, Ys
    #
    # def remove_by_zscore_spectra(self, tau=3.5, max_count=40):
    #     Xs, Ys = self.x, self.y
    #     Is = []

    #     Zs = utils.modified_zscore2(Ys)
    #     ids = []
    #     for i,zs in enumerate(Zs):
    #         count = (abs(zs) > tau).astype('i').sum()
    #         if count < max_count:
    #             ids.append(i)
    #         # else:
    #         #     print(count)

    #     if ids:
    #         ids = np.array(ids)
    #         Ys = Ys[ids]
    #         Xs = Xs[ids]

    #         self.x, self.y = Xs, Ys
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
    def select_baselines(self, tau2=1000.0, tau1=0.0, tau_z=0, tau2_smooth=1.0, override_tau2=False,
                         bs_scale=3.0, alpha=0.001, eps=0.001, beta=1.0e2,
                         func=None, func1=None, func2=None, w_tau2=1.0,                         
                         d=2, func2_mode="d", **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.y)

        xs = self.x

        for i in range(N):
            self.tau2_values[i] = tau2
        
        self.tau1_values = N * [tau1]

        tau2_max = max(self.tau2_values)
        tau2_min = min(self.tau2_values)

        tau2_max *= 20
        if tau2 <= 20.0:
            tau2_min /= 20.0
            tau2_step = tau2_min / 20
        else:
            tau2_min = 1.0
            tau2_step = 1.0
        
        # tau1_max = max(self.tau1_values)
        # tau1_min = min(self.tau1_values)

        # tau1_max *= 20
        # if tau1 < 20.0:
        #     tau1_min /= 20.0
        #     tau1_step = tau1_min
        # else:
        #     tau1_min = 1.0
        #     tau1_step = 1.0
        
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
            tau2_slider.min = tau2_slider.value / 50
            tau2_slider.max = tau2_slider.value * 20
            tau2_slider.step = tau2_slider.min / 10
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
        
        @ipywidgets.interact(i=i_slider, tau2=tau2_slider, #tau1=tau1_slider, 
                             xrange=xrange_slider, continuous_update=False)
        def _plot_select_baselines(i, tau2, xrange):
            fig = plt.figure("_select_baselines", figsize=(12,7))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'            
            fig.clear()
            plt.subplot(2,1,1)
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.4, alpha=0.5)

            ys_i = self.y[i]
            plt.plot(xs, ys_i, linewidth=1.5, color='DarkBlue', label=f"current ({i})")

            W2 = np.full(len(self.y[0]), 1.0, "d")
            if self.windows is not None:
                for xa,xb in self.windows:
                    W2[(xs >= xa) & (xs <= xb)] = beta
            
            ys_i_smooth = smooth.whittaker_smooth(
                    ys_i, 
                    tau2=tau2_smooth,
                    W2=W2, 
                    d=2)
                        
            # ys_i_smooth = ys_i
            # plt.plot(xs, ys_i_smooth, linewidth=1.5, color='LightBlue', label="smoothed")


            # i_windows = []
            xs = self.x
            # for xa, xb in self.windows:
            #     ix_range = np.argwhere((xs >= xa) & (xs <= xb)).ravel()
            #     i_windows.append((ix_range[0], ix_range[-1]))

            # ys_i_smooth /= ys_i_smooth[0]

            bs, dd = smooth.whittaker_smooth_weight_func2(
                ys_i_smooth,
                func=func,
                # func1=func1,
                func2=func2,
                # tau1=self.tau1_values[i], 
                tau2=tau2_slider.value, 
                tau_z=tau_z,
                d=d, 
                # windows = i_windows, w_tau2=w_tau2,
                func2_mode=func2_mode)

            self.bs[i,:] = bs
            # self.ys_bs[i,:] = self.ys[i] - self.bs[i]
    
            plt.plot(xs, bs, linewidth=1.0, color='m', label="baseline")

            x_min, x_max = xrange
            plt.xlim(0.95*x_min, 1.05*x_max)

            bs_xrange = bs[(x_min <= xs) & (xs <= x_max)]
            ys_xrange = ys_i[(x_min <= xs) & (xs <= x_max)]
            bs_max, bs_min = np.max(bs_xrange), np.min(bs_xrange)
            # plt.ylim(0.95*np.min(ys_xrange), bs_scale*(bs_max))
            # plt.ylim(0, bs_scale*(bs_max))
            plt.minorticks_on()
            plt.grid(1)
            plt.legend()
            # plt.tight_layout()
            # plt.show()

            # fig = plt.figure("_draw_baselines", figsize=(12,3))
            # fig.canvas.header_visible = False
            # fig.canvas.footer_visible = False
            # fig.canvas.toolbar_position = 'right'            
            # fig.clear()
            # plt.title(self.key, fontdict={"size":10}, loc="left")
            plt.subplot(2,1,2)
            for ys, bs in zip(self.y, self.bs):
                plt.plot(self.x, ys-bs, linewidth=0.4, alpha=0.5)
            plt.plot(self.x, self.y[i]-self.bs[i], linewidth=1.0, color='DarkBlue')
            plt.xlim(0.95*x_min, 1.05*x_max)
            # plt.ylim(0, np.max(ys_i))
            plt.minorticks_on()
            plt.grid(1)
            
            plt.tight_layout()
            plt.show()
             
            # plt.figure(figsize=(10,3))
            # plt.plot(dd['qvals'])
            # plt.show()
    #
    def select_baseline_param(self, 
                         tau2=1000.0,
#                          bs_scale=3.0, 
                         func=None, func2=None,                         
                         d=2, func2_mode="d",
                         override_tau2=False, 
                         tau_z=0, tau2_smooth=1.0, beta=100,
                         **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets

        xs = self.x

        if override_tau2 or not self.tau2s:
            self.tau2s = 3*[tau2]            
        
        tau2_min_slider = ipywidgets.FloatSlider(value=self.tau2s[0], 
                                                 min=self.tau2s[0]/50, 
                                                 max=self.tau2s[0]*25, 
                                                 step=self.tau2s[0]/100)
        tau2_min_slider.layout.width="80%"   
        tau2_mean_slider = ipywidgets.FloatSlider(value=self.tau2s[1], 
                                                min=self.tau2s[1]/50, 
                                                max=self.tau2s[1]*25, 
                                                step=self.tau2s[1]/100)
        tau2_mean_slider.layout.width="80%"   
        tau2_max_slider = ipywidgets.FloatSlider(value=self.tau2s[2], 
                                                 min=self.tau2s[2]/50, 
                                                 max=self.tau2s[2]*25, 
                                                 step=self.tau2s[2]/100)
        tau2_max_slider.layout.width="80%"   

        def tau2_min_on_value_change(change):
            self.tau2s[0] = tau2_min_slider.value
        def tau2_mean_on_value_change(change):
            self.tau2s[1] = tau2_mean_slider.value
        def tau2_max_on_value_change(change):
            self.tau2s[2] = tau2_max_slider.value
        
        tau2_min_slider.on_trait_change(tau2_min_on_value_change, name="value")
        tau2_max_slider.on_trait_change(tau2_max_on_value_change, name="value")
        tau2_mean_slider.on_trait_change(tau2_mean_on_value_change, name="value")

        ys_max = np.max(self.y, axis=0)
        ys_mean = np.mean(self.y, axis=0)
        ys_min = np.min(self.y, axis=0)
        
        self.Ys = np.vstack((ys_min, ys_mean, ys_max))
        labels = ["min", "mean", "max"]
        
        W2 = np.full(len(ys_max), 10.0, "d")
        if self.windows is not None:
            for xa,xb in self.windows:
                W2[(xs >= xa) & (xs <= xb)] = beta
                
        Ys_smooth = []
        for ys in self.Ys:
            ys_smooth = smooth.whittaker_smooth(
                ys, tau2=tau2_smooth, 
                W2=W2, 
                d=2)
            Ys_smooth.append(ys_smooth)
        
#         if func is None:
#             sigma = (ys - ys_smooth).std()        
#             func = funcs.Step(sigma)        
                
        @ipywidgets.interact(tau2_max=tau2_max_slider, 
                             tau2_mean=tau2_mean_slider, 
                             tau2_min=tau2_min_slider, 
                             continuous_update=False)
        def _plot_select_baseline_param(tau2_max, tau2_mean, tau2_min):
            fig = plt.figure("select_baseline_param", figsize=(12,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'            
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            colors=["DarkRed", "DarkGreen", "DarkBlue"]
            for i in range(3):
                plt.plot(xs, self.Ys[i], linewidth=1.0, color=colors[i], label=labels[i])
            
            Bs = []
            for i, ys_smooth in enumerate(Ys_smooth):
                bs, _ = smooth.whittaker_smooth_weight_func2(
                    ys_smooth, 
                    func=func,
                    # func1=func1,
                    func2=func2,
                    # tau1=self.tau1_values[i], 
                    tau2=self.tau2s[i], 
                    tau_z=tau_z,
                    d=d, 
                    func2_mode=func2_mode)
                Bs.append(bs)
    
    
            for i in range(3):
                if i == 0:
                    plt.plot(xs, Bs[i], linewidth=1.0, color='m', label="baseline")                
                    plt.plot(xs, self.Ys[i]-Bs[i], linewidth=0.75, color='Grey', label="corrected")
                else:
                    plt.plot(xs, Bs[i], linewidth=1.0, color='m')
                    plt.plot(xs, self.Ys[i]-Bs[i], linewidth=0.75, color='Grey')
            
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
    def subtract_baselines(self, func=None, func2=None, d=2, func2_mode="d", tau_z=0):
        import ipywidgets
        from IPython.display import display
        
        N = len(self.y)
        fp = ipywidgets.FloatProgress(min=0, max=N, value=0, bar_style="success", description=self.key)
        fp.style.width = 50
        display(fp)
        for i, ys_i in enumerate(self.y):
            
            j = abs(self.Ys - ys_i).sum(axis=1).argmin()

            bs, _ = smooth.whittaker_smooth_weight_func2(
                ys_i, 
                func=func,
                # func1=func1,
                func2=func2,
                # tau1=self.tau1_values[i], 
                tau2=self.tau2s[j], 
                tau_z=tau_z,
                d=d, 
                func2_mode=func2_mode)    

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
    def plot_spectra(self, **kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets
        
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1, description="i")
        i_slider.layout.width="50%"
        
        # f_slider = ipywidgets.FloatSlider(value=3.5, min=1.0, max=10.0)
        # f_slider.layout.width="50%"   
        
        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x), max(self.x)), 
            min=min(self.x), 
            max=max(self.x),
            description="xrange",)
        xrange_slider.layout.width="50%"

        lines = []
        i_current = i_slider.value
        # xrange_current = xrange_slider.value

        with plt.ion():
            fig = plt.figure("_plot_spectra", figsize=(12,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
        
        def i_slider_on_value_change(change):
            nonlocal i_current
            with plt.ion():
                line_prev = lines[i_current]
                line_prev.set_alpha(0.25)
                line_prev.set_linewidth(0.5)
                i_current = i_slider.value
                line = lines[i_current]
                line.set_linewidth(1.5)
                line.set_alpha(1.0)

                xs = self.x
                x_min, x_max = xrange_slider.value
                ys_i = self.y[i_current]
                ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
                y_min, y_max = 0.85*ys_range.min(), 1.15*ys_range.max()
                
                plt.ylim(y_min, y_max)
                
                # ax.figure.canvas.draw()
                # ax.figure.canvas.flush_events()
                
                fig.canvas.draw()
                fig.canvas.flush_events()        

        def xrange_slider_on_value_change(change):
            with plt.ion():
                xs = self.x
                x_min, x_max = xrange_slider.value
                i = i_slider.value
                ys_i = self.y[i]
                ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
                ymin, y_max = 0.85*ys_range.min(), 1.15*ys_range.max()
                ax = fig.gca()
                plt.ylim(ymin, y_max)
                plt.xlim(x_min, x_max)
                
                # ax.figure.canvas.draw()
                # ax.figure.canvas.flush_events()
                
                fig.canvas.draw()
                fig.canvas.flush_events()        
                

        i_slider.on_trait_change(i_slider_on_value_change, name="value")
        xrange_slider.on_trait_change(xrange_slider_on_value_change, name="value")

        display(i_slider)
        display(xrange_slider)

        plt.title(self.key, fontdict={"size":10}, loc="left")
        xs = self.x
        for ys in self.y:
            line = plt.plot(xs, ys, linewidth=0.5, alpha=0.25)[0]
            lines.append(line)

        # err_i = self.stderr[i]
        # plt.fill_between(xs, ys_i-2*err_i, ys_i+2*err_i, alpha=0.5)
        # plt.plot(xs, ys_i, linewidth=1.5, color='DarkBlue')

        line = lines[i_current]
        line.set_linewidth(1.5)
        line.set_alpha(1.0)
        

        # std_err_i = err_i.mean()

        # diff2_i = inventory.diff2(ys_i)
        # mu_i = inventory.robust_mean_1d(diff2_i, 3.0)
        # dd2_i = inventory.robust_mean_1d(abs(diff2_i - mu_i), 3.0)
        
                # ys_i_smooth = ys_i
        # plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkRed', 
        #          # marker='s', markersize=2,
        #          label=fr"smoothed") # ($\sigma={std_err_i:.3f}$)")

        # ys_i_smooth = smooth.whittaker_smooth(ys_i, tau2=tau, W2=W2, d=2)
        
        # ys_i_smooth = ys_i
        # plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkRed', 
        #          # marker='s', markersize=2,
        #          label=fr"smoothed") # ($\sigma={std_err_i:.3f}$)")

        i = i_slider.value
        ys_i = self.y[i]
        x_min, x_max = xrange = xrange_slider.value
        ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
        ymin, y_max = 0.9*np.min(ys_range), 1.2*np.max(ys_range)
        plt.ylim(ymin, y_max)
        
        plt.minorticks_on()
        plt.tight_layout()
        plt.xlim(x_min, x_max)
        # plt.legend()
        plt.show()
        
        # @ipywidgets.interact(i=i_slider, xrange=xrange_slider, continuous_update=False)
        # def _plot_spectra(i, xrange):
        #     plt.figure(figsize=(12,4))
        #     plt.title(self.key)
        #     xs = self.x
        #     for ys in self.y:
        #         plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

        #     ys_i = self.y[i]
        #     # err_i = self.stderr[i]
        #     # plt.fill_between(xs, ys_i-2*err_i, ys_i+2*err_i, alpha=0.5)
        #     plt.plot(xs, ys_i, linewidth=1.5, color='DarkBlue', label="original")

        #     # std_err_i = err_i.mean()

        #     # diff2_i = inventory.diff2(ys_i)
        #     # mu_i = inventory.robust_mean_1d(diff2_i, 3.0)
        #     # dd2_i = inventory.robust_mean_1d(abs(diff2_i - mu_i), 3.0)
            
            
        #     # ys_i_smooth = smooth.whittaker_smooth(ys_i, tau2=tau, W2=W2, d=2)
        #     ys_i_smooth = ys_i
        #     plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkRed', 
        #              # marker='s', markersize=2,
        #              label=fr"smoothed") # ($\sigma={std_err_i:.3f}$)")

        #     x_min, x_max = xrange
        #     ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
        #     ymin, y_max = 0.9*np.min(ys_range), 1.2*np.max(ys_range)
        #     plt.ylim(ymin, y_max)
            
        #     plt.minorticks_on()
        #     plt.tight_layout()
        #     plt.xlim(x_min, x_max)
        #     plt.legend()
        #     plt.show()
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
            plt.figure(figsize=(12,4))
            plt.title(self.key)
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
            fig.canvas.toolbar_position = 'right'            
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            ys_i = self.y[i]
            # err_i = self.stderr[i]
            # plt.fill_between(xs, ys_i-2*err_i, ys_i+2*err_i, alpha=0.5)
            plt.plot(xs, ys_i, linewidth=1.5, color='DarkBlue', label="original")

            if alpha == 1.0:
                if self.windows is not None:
                    W2 = np.full(len(self.y[0]), 1.0, "d")
                    for xa,xb in self.windows:
                        W2[(xs >= xa) & (xs <= xb)] = beta
                
                ys_i_smooth = smooth.whittaker_smooth(
                        ys_i, 
                        tau2=tau2,
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
                     # marker='s', markersize=2,
                     label=fr"smoothed: $\mu={mu:.2f}$ $\sigma={sigma:.2f}$")

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

class SpectraCollection:
    #
    def __init__(self, spectra=None):
        if spectra is None:
            self.spectra = {}
        else:
            self.spectra = spectra
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
                sp = ExperimentSpectra(x, y, attrs)
                sp.key = key
                self[key] = sp
