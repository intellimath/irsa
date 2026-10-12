#
# objects.py
#

import numpy as np
import rampy

import matplotlib.pyplot as plt
import ipywidgets
from ipywidgets import fixed

from irsa.preprocess import utils
import irsa.analytics as analytics

import mlgrad.funcs as funcs
import mlgrad.pca as pca
from mlgrad.af import averaging_function
import mlgrad.smooth as smooth
import mlgrad.inventory as inventory
import scipy.special as special
from IPython.display import display
import joblib

from mlgrad.af import averaging_function

from scipy.spatial.distance import pdist, squareform
from scipy import stats

def xrange2irange(xrange, xs):
    xa, xb = xrange
    ix_range = np.argwhere((xs >= xa) & (xs <= xb)).ravel()
    ia, ib = min(ix_range), max(ix_range)
    return ia, ib

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

def logscale(ys, by="std"):
    zs = np.log(ys - ys.min() + 1)
    mu = zs.mean()
    if by == "mean":
        zs = zs / mu
    elif by == "std":
        ss = zs.std()
        zs = (zs - mu) / ss
    zs = np.exp(zs)
    return zs

def smooth_spectra(xs, ys, tau2=1.0, windows=None, beta=100, tune=True):
    W2 = None
    if tune:
        ys_smooth = smooth.whittaker_smooth(ys, tau2=tau2, d=2)

        ee = ys - ys_smooth
        ss = ee.std()
        if ss != 0:
            ee = (ee - ee.mean()) / ss

        W2 = np.exp(-0.5 * ee * ee)[1:-1]
        if windows is not None:
            for xa, xb in windows:
                W2[(xs[1:-1] >= xa) & (xs[1:-1] <= xb)] *= beta

    ys_smooth = smooth.whittaker_smooth(ys, tau2=tau2, W2=W2, d=2)
    return ys_smooth

def explained_variance_ratio(X0, *, n_component=None):
    from sklearn.decomposition import PCA, TruncatedSVD

    m = X0.shape[0]
    X1 = X0.copy()
    gap = X1[-1] - X1[0]
    trend = np.linspace(0, gap, m)
    X = X1 - trend
    Z = np.empty((m,m))
    for i in range(m):
        Z[i,:] = np.roll(X, i)
    pca = PCA(n_component)
    U = pca.fit_transform(Z)
    ratio = np.array(pca.explained_variance_ratio_)
    return ratio

def smooth_pca(X0, *, n_component=None):
    from sklearn.decomposition import PCA

    m = X0.shape[0]
    # if n_component is None:
    #     n_component = m

    X1 = X0.copy()
    gap = X1[-1] - X1[0]
    trend = np.linspace(0, gap, m)
    X = X1 - trend
    Z = np.empty((m,m))
    for i in range(m):
        Z[i,:] = np.roll(X, i)
    pca = PCA(n_component)
    U = pca.fit_transform(Z)
    # ratio = pca.explained_variance_ratio_
    # print(ratio)
    Z2 = pca.inverse_transform(U)
    for i in range(m):
        Z2[i,:] = np.roll(Z2[i], -i)
    # X1 = inventory.robust_mean_2d_t(Z2, 3.0)
    X1 = Z2.mean(axis=0)
    X = X1 + trend
    # ret = np.zeros_like(X0)
    # ret[I] = X
    return X, {"ls": pca.explained_variance_ratio_}

def denoise(X, *, n_component=None):
    from .denoise import Denoiser

    if n_component is None:
        m = X.shape[0] // 2
    else:
        m = n_component
    denoiser = Denoiser()
    # gap = X[-1] - X[0]
    return denoiser.denoise(X, m)

def denoise_svd(X, method='energy', threshold=0.9975, k=None, transpose=False):
    """
    Фильтрация набора спектров методом SVD.

    Параметры:
    -----------
    X : ndarray (m x n)
        Матрица спектров (строки — точки спектра, столбцы — спектры)
    method : str
        Метод выбора ранга: 'variance', 'energy', 'elbow', 'manual'
    threshold : float
        Пороговое значение (по умолчанию 0.995)
    k : int
        Ручной выбор ранга (если method='manual')

    Возвращает:
    -----------
    Y : ndarray (m x n)
        Спектры после фильтрации
    info : dict
        Словарь с информацией об оценках качества фильтрации
        (k, собственные значения, ошибка, SNR, энергия сигнала, энергия шума)
    """
    from scipy.linalg import svd

    if transpose:
        XX = X.T
    else:
        XX = X
    # Вычислить SVD
    U, sigma, VT = svd(XX, full_matrices=False)

    # Выбрать ранг k
    if k is not None:
        k = int(k)
    elif method == 'variance':
        cumsum_variance = sigma.cumsum() / sigma.sum()
        k = np.argmax(cumsum_variance >= threshold) + 1
    elif method == 'energy':
        sigma2 = sigma**2
        cumsum_energy = sigma2.cumsum() / sigma2.sum()
        k = np.argmax(cumsum_energy >= threshold) + 1
    elif method == 'elbow':
        gaps = sigma[:-1] - sigma[1:]
        k = np.argmax(gaps) + 1
    else:
        raise ValueError("method должен быть 'threshold', 'energy', 'elbow' или 'k' должен быть задан")

    if k >= len(sigma):
        k = len(sigma) - 1

    # Восстановить очищенные спектры
    Y = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]

    if transpose:
        Y = Y.T

    # Вычислить метрики качества
    error = np.linalg.norm(X - Y, 'fro')
    signal_energy = np.sum(sigma[:k]**2)
    noise_energy = np.sum(sigma[k:]**2)
    sigma2 = sigma**2
    # cumsum_energy = np.cumsum(sigma2) / np.sum(sigma2)
    SNR_ratio = signal_energy / (noise_energy + 1e-10)

    info = {
        'error': error,
        'signal_energy': signal_energy,
        'noise_energy': noise_energy,
        # 'cumulative_energy': cumsum_energy,
        'SNR_ratio': SNR_ratio,
        "sigma": sigma,
        'k': k,
    }
    return np.ascontiguousarray(Y), info

def despike(X, tau=3.5, mode=None, beta=0.5):
    X11 = X[1:-1]
    X02 = X[:-2]
    X20 = X[2:]
    S1, S2 = abs(X11 - X02), abs(X11 - X20)
    S = S1 * S2 / (S1 + S2 + 1.0e-10)
    # S = inventory.diff2(X)
    Z = inventory.modified_zscore(S)
    if mode == "+":
        mask1 = Z < -tau
    elif mode == "-":
        mask1 = Z > tau
    else:
        mask1 = abs(Z) > tau
    mask2 = ((X02 < X11) & (X11 > X20)) | ((X02 > X11) & (X11 < X20))
    mask12 = mask1 & mask2
    mask = (X != X)
    mask[1:-1] = mask12
    Y = np.empty_like(X)
    Y[:] = X
    for i,ma in enumerate(mask):
        if ma:
            xx1 = abs(X[i] - X[i-1])
            xx2 = abs(X[i+1] - X[i])
            xx_min, xx_max = min(xx1, xx2), max(xx1, xx2)
            if xx_min < beta * xx_max:
                mask[i] = False
                continue
            Y[i] = (X[i-1] + X[i+1]) / 2
    return Y, mask

def despiking(X, tau=3.0, mode=None, beta=0.5):
    i_tau = int(10*tau)
    i = 35
    mask = (X != X)
    while i >= i_tau:
        t = i / 10
        t = round(t, 1)
        while True:
            X1, mask0 = despike(X, t, mode=mode, beta=beta)
            mask |= mask0
            if (X1 == X).all():
                X = X1
                break
            X = X1
        i -= 1
    return X, mask

def fractal_dimension_box_counting(curve_points, max_box_size=None, min_box_size=1):
    """
    Calculate fractal dimension using the box-counting method.
    
    Parameters:
    - curve_points: array of (x, y) coordinates
    - max_box_size: largest box size (default: half the curve's extent)
    - min_box_size: smallest box size (default: 1)
    
    Returns:
    - dimension: estimated fractal dimension
    #- box_sizes: array of box sizes tested
    #- counts: number of boxes needed at each size
    """

    # Normalize curve to a grid
    curve_points = np.array(curve_points)

    # Scale to integer pixel coordinates
    min_coords = curve_points.min(axis=0)
    max_coords = curve_points.max(axis=0)
    range_coords = max_coords - min_coords

    # Map to pixel grid (1000x1000)
    grid_size = 1024
    normalized = ((curve_points - min_coords) / range_coords * (grid_size - 1)).astype(int)

    # Create binary image
    image = np.zeros((grid_size, grid_size), dtype=bool)
    image[normalized[:, 1], normalized[:, 0]] = True

    # Box sizes to test (powers of 2)
    if max_box_size is None:
        max_box_size = grid_size // 4

    box_sizes = 2 ** np.arange(np.log2(min_box_size), np.log2(max_box_size), 0.5).astype(int)
    box_sizes = np.unique(box_sizes)

    counts = []

    for box_size in box_sizes:
        # Count boxes containing the curve
        coarse = image.reshape(
            image.shape[0] // box_size, box_size,
            image.shape[1] // box_size, box_size
        ).max(axis=(1, 3))
        count = np.count_nonzero(coarse)
        counts.append(count)

    counts = np.array(counts)

    # Fit log-log relationship: log(N) = -D * log(box_size) + const
    log_sizes = np.log(box_sizes)
    log_counts = np.log(counts)

    # Linear regression
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    dimension = -coeffs[0]  # Negative slope

    return dimension #, box_sizes, counts, coeffs

def covering_greedy(points, radius):
    """
    Greedy algorithm to find minimum number of balls of given radius
    that cover all points.
    
    Complexity: O(N²) but practical for N < 10000
    """
    uncovered = set(range(len(points)))
    num_balls = 0
    
    while uncovered:
        # Pick an uncovered point
        point_idx = uncovered.pop()
        center = points[point_idx]
        
        # Find all points within radius
        distances = np.linalg.norm(points - center, axis=1)
        covered = np.where(distances <= radius)[0]
        
        # Remove covered points from uncovered set
        uncovered -= set(covered)
        num_balls += 1
    
    return num_balls

def covering_random(points, radius, max_iterations=1000):
    """
    Random covering: pick random centers and cover points.
    Alternative to greedy method.
    """
    uncovered = set(range(len(points)))
    num_balls = 0
    iterations = 0

    while uncovered and iterations < max_iterations:
        # Pick random uncovered point
        point_idx = np.random.choice(list(uncovered))
        center = points[point_idx]

        # Find all points within radius
        distances = np.linalg.norm(points - center, axis=1)
        covered = np.where(distances <= radius)[0]

        uncovered -= set(covered)
        num_balls += 1
        iterations += 1

    return num_balls


def hausdorff_dimension(points, num_scales=25, min_scale=None, max_scale=None):
    """
    Refined version using multiple covering strategies and averaging.
    More robust to noise and sampling artifacts.
    """
    
    points = np.asarray(points, dtype=float)
    
    extent = points.max(axis=0) - points.min(axis=0)
    max_extent = np.max(extent)
    
    if min_scale is None:
        min_scale = max_extent * 0.005
    if max_scale is None:
        max_scale = max_extent * 0.4
    
    radii = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    covers_greedy = []
    covers_random = []
    
    # Greedy covering
    for r in radii:
        covers_greedy.append(covering_greedy(points, r))
    
    # Random covering (average over 3 trials)
    for r in radii:
        trials = []
        for _ in range(3):
            trials.append(covering_random(points, r))
        covers_random.append(np.mean(trials))
    
    # Combine both methods
    covers = np.array([(g + rnd) / 2 for g, rnd in zip(covers_greedy, covers_random)])
    
    log_radii = np.log(radii)
    log_covers = np.log(covers)
    
    slope, intercept, r_value, _, std_err = stats.linregress(log_radii, log_covers)
    dimension = -slope
    r_squared = r_value ** 2
    
    return {
        'dimension': dimension,
        'r_squared': r_squared,
        'radii': radii,
        'covers': covers,
        'covers_greedy': np.array(covers_greedy),
        'covers_random': np.array(covers_random),
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err
    }

def multiplicative_scatter_correction(spectra, reference=None):
    """
    Multiplicative scatter correction using pure NumPy matrix operations.
    
    Parameters:
    -----------
    spectra : np.ndarray, shape (n_samples, n_features)
        Input spectral data
    reference : np.ndarray, shape (n_features,), optional
        Reference spectrum (default: mean of all samples)
    
    Returns:
    --------
    corrected_spectra : np.ndarray, shape (n_samples, n_features)
        MSC-corrected spectra
    m_factors : np.ndarray, shape (n_samples,)
        Multiplicative correction factors
    a_factors : np.ndarray, shape (n_samples,)
        Additive correction factors
    reference : np.ndarray, shape (n_features,)
        Reference spectrum used
    """
    
    spectra = np.asarray(spectra, dtype=np.float64)
    n_samples, n_features = spectra.shape
    
    # Compute reference if not provided
    if reference is None:
        reference = spectra.mean(axis=0)
    else:
        reference = np.asarray(reference, dtype=np.float64)
    
    # Center reference spectrum
    ref_mean = np.mean(reference)
    ref_centered = reference - ref_mean
    
    # Center all spectra
    spec_means = spectra.mean(axis=1)[:,None]  # Shape: (n_samples, 1)
    spectra_centered = spectra - spec_means
    
    # Calculate regression coefficients using matrix operations
    # m_i = Σ[(x_i - x̄_i) * (x_ref - x̄_ref)] / Σ[(x_ref - x̄_ref)²]
    
    numerator = spectra_centered @ ref_centered  # Shape: (n_samples,)
    denominator = ref_centered @ ref_centered    # Scalar
    
    m_factors = numerator / denominator
    
    # a_i = x̄_i - m_i * x̄_ref
    a_factors = spec_means[:,0] - m_factors * ref_mean
    
    # Apply correction: (x_i - a_i) / m_i
    corrected_spectra = (spectra - a_factors[:, None]) / m_factors[:, None]

    corrected_spectra = np.array([y for y in corrected_spectra if not np.any(np.isnan(y)) and not np.any(np.isinf(y))])
    
    return corrected_spectra #, m_factors, a_factors, reference


class SpectraSeries:
    """
    Набор спектров, состоящий из серий. Все спектры в серии измерены в одной "точке".
    """
    #
    def __init__(self, x, y, attrs):
        self.x = x
        self.y = y
        self.attrs = attrs
        self.excludes = []
        self.tau2s = None
        self.bs = None
        self.windows = []
    #
    @property
    def key(self):
        return self.attrs["key"]
    #
    def __getitem__(self, i):
        sp = Spectra(self.x[i], self.y[i], self.attrs)
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
    def exclude_by_default(self, sigma_mu=0.10, alpha=3.5, clear=False):
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

        self.exclude_selected()
    #
    def select_baselines(self, tau2=1.0e5, tau1=0, tau2_smooth=0, func=None, func2=None, func2_e=None, 
                         beta=100, func2_mode="e", xrange=None, use_sigma=False, d=2):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.y)

        i_series_slider = ipywidgets.IntSlider(value=0, min=0, max=N-1)
        i_series_slider.layout.width="50%"

        @ipywidgets.interact(i_series=i_series_slider, continuous_update=False)
        def _plot_select_baselines(i_series):
            nonlocal tau2, tau2_smooth, func, func2, beta
            spcol = self[i_series]

            Ys = self.y[i_series]

            spcol.select_baselines(tau2=tau2, tau1=tau1, tau2_smooth=tau2_smooth,
                                   func=func, func2=func2, func2_e=func2_e, beta=beta,
                                   func_mode='e', xrange=xrange, use_sigma=use_sigma, d=d)
    #
    def replace_spectra_with_corrected(self):
        for i in range(len(self.y)):
            sp = self[i]
            sp.replace_spectra_with_corrected()
    #
    def subtract_baselines(self, func=None, func2=None, func2_e=None, d=2, tau2=1.0e3, tau1=0, verbose=True, n_jobs=1):
        import ipywidgets
        from IPython.display import display

        N = len(self.y)
        if verbose:
            fp = ipywidgets.FloatProgress(min=0, max=N, value=0, bar_style="success", description=self.key)
            fp.layout.width = "auto"
            display(fp)

        for i in range(N):
            sp = self[i]
            sp.subtract_baselines(func=func, func2=func2, func2_e=func2_e, d=d, tau2=tau2, tau1=tau1, verbose=False, n_jobs=n_jobs)
            self.y[i] = sp.y
            if verbose:
                fp.value = i+1
        if verbose:
            fp.close()
    #
    def plot_spectra(self, smoothing=False, kind="wh", #display_data=False,
                     marker='s', ms=0, lw=0.25, **kwds):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_series_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_series_slider.layout.width="50%"

        @ipywidgets.interact(i_series=i_series_slider, continuous_update=False)
        def _plot_spectra(i_series):
            sp = self[i_series]
            sp.plot_spectra(smoothing=smoothing, kind=kind, #display_data=display_data, 
                            marker=marker, ms=ms, lw=lw, **kwds)
    #
    def plot_msc(self):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_series_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_series_slider.layout.width="50%"

        @ipywidgets.interact(i_series=i_series_slider, continuous_update=False)
        def _plot_msc(i_series):
            sp = self[i_series]
            sp.plot_msc()
    #
    def plot_pca_spectra(self, n_component=None):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_pca_spectra(i):
            fig = plt.figure("_plot_pca_spectra", figsize=(12,5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()

            Ys = self.y[i]

            nonlocal n_component
            if n_component is None:
                n_component = Ys.shape[0]
            c, As, Ls = pca.find_loc_and_pc(Ys, n_component)
            Us = (Ys - c) @ As.T
            plt.subplot(1,2,1)
            plt.title(f"{self.key} ({len(self.y[i])} spectra)")
            plt.minorticks_on()
            plt.grid(1)
            plt.plot(Ls/Ls.sum(), marker='o')
            plt.subplot(1,2,2)
            plt.scatter(Us[0], Us[1], c='w', edgecolors='k', s=36)
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()
    #
    def crop(self, start_x, end_x=None):
        Xs = self.x
        Ys = self.y
        N = len(Ys)
        for k in range(N):
            xs = Xs[k]
            mask = (xs >= start_x)
            if end_x is not None:
                mask &= (xs <= end_x)
            Xs[k] = np.ascontiguousarray(Xs[k][mask])
            Ys[k] = np.ascontiguousarray(Ys[k][:,mask])
    #
    def take(self, indices):
        Ys = self.y
        N = len(Ys)
        for k in range(N):
            Ys[k] = np.ascontiguousarray(Ys[k][indices])
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
    def plot_window_mean_std(self):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        def i_on_value_change(change):
            # i = i_slider.value
            j_slider.value = 0

        i_slider.on_trait_change(i_on_value_change, name="value")

        j_slider = ipywidgets.IntSlider(min=0, max=len(self.x[0])-1, value=0)
        j_slider.layout.width="50%"

        xrange_slider = ipywidgets.IntRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="90%"

        @ipywidgets.interact(i=i_slider, j=j_slider, xrange=xrange_slider, continuous_update=False)
        def _plot_window_mean_std(i, j, xrange):
            xs = self.x[i]
            Ys = self.y[i]

            # plt.close("_window_mean_std")
            plt.figure("_window_mean_std", figsize=(14,4))
            # fig.canvas.header_visible = False
            # fig.canvas.footer_visible = False
            # fig.canvas.toolbar_position = 'right'
            plt.title(f"{self.key} ({len(self.x[i])} spectra)")
            for ys in Ys:
                plt.plot(xs, ys, lw=0, alpha=0.5, marker='s', markersize=4, color='gray')

            mu_i = inventory.robust_mean_2d_t(Ys, tau=3.5)
            plt.plot(xs, mu_i, linewidth=1.5, color='darkred', label="mean spectra")
            bs = smooth.whittaker_smooth(mu_i, tau2=1.0e3,
                      func=funcs.RStep(0.01, 0.001),
                      func2_e=inventory.relative_abs_max,)
            plt.plot(xs, bs, linewidth=1.5, color='m', label="baseline")

            plt.plot(xs, Ys[i], linewidth=0.75, color='darkblue', label="current")

            plt.vlines(xs[j], 0, Ys[i].max(), color='k')

            # plt.xlabel("j")
            plt.tight_layout()
            x_min, x_max = xrange
            ii = np.argwhere((xs >= x_min) & (xs <= x_max)).ravel()
            i0, i1 = min(ii), max(ii)
            plt.xlim(0.99*x_min, 1.01*x_max)
            plt.ylim(min(ys[i0:i1+1].min() for ys in Ys), max(ys[i0:i1+1].max() for ys in Ys))
            plt.legend()
            plt.minorticks_on()
            plt.grid(lw=0.75, ls=':')
            plt.show()

            plt.figure("_hist_j", figsize=(10,3))
            # plt.hist(inventory.modified_zscore(ys_j), rwidth=0.9, bins=6)
            plt.hist(Ys[:,j], rwidth=0.9, bins=6)
            plt.minorticks_on()
            plt.grid(lw=0.75, ls=':')
            plt.tight_layout()
            plt.show()

            plt.figure("_window_mean_std_j", figsize=(12,2))
            # fg.canvas.header_visible = False
            # fg.canvas.footer_visible = False
            # fg.canvas.toolbar_position = 'right'
            plt.title(f"Means and std")

            ys_j = Ys[:,j]
            m = len(ys_j)
            _mu = []
            _std = []
            for ii in range(m-10):
                _mu.append(ys_j[ii:ii+10].mean())
                _std.append(ys_j[ii:ii+10].std())
            _mu = np.array(_mu)
            _std = np.array(_std)

            plt.plot(range(len(_mu)), _mu, marker='o', color='k')
            plt.fill_between(range(len(_mu)), _mu-_std, _mu+_std, color='lightblue')
            plt.minorticks_on()
            plt.grid(ls=':', lw=0.75)
            plt.tight_layout()
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
    def plot_logscale(self):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        # def i_on_value_change(change):
        #     i = i_slider.value

        # i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_logscale_items(i=i_slider):
            xs = self.x[i]
            Ys = self.y[i]

            plt.close("_zscore_i")
            fig = plt.figure("plot_logscale_i", figsize=(12,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(f"Logscale: {self.key} ({len(self.x[i])} spectra)")

            Zs = np.empty_like(Ys)
            for k, ys in enumerate(Ys):
                Zs[k,:] = logscale(ys)

            for zs in Zs:
                plt.plot(xs, zs, linewidth=0.25, alpha=0.25)

            plt.plot(xs, inventory.robust_mean_2d_t(Zs, 3.5), color="k", linewidth=1.5)

            plt.minorticks_on()
            plt.show()
    #
    def plot_sigma_mu(self, tau2_smooth=1.0, tau2_bl=1.0e6, func=None, n_component=None):
        import matplotlib.pyplot as plt
        import ipywidgets
        import mlgrad.smooth as smooth

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        xrange_slider = ipywidgets.IntRangeSlider(
            value=(min(self.x[0]), max(self.x[0])), 
            min=min(self.x[0]), 
            max=max(self.x[0]))
        xrange_slider.layout.width="90%"

        if func is None:
            func = funcs.Step(0.0, 0.001)

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_sigma_mu(i=i_slider, xrange=xrange_slider):
            xs = self.x[i]
            Ys = self.y[i]

            plt.close("_sigma_mu_i")
            fig = plt.figure("_sigma_mu_i", figsize=(12,5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(rf"$\sigma/\mu$: {self.key} ({len(self.x[i])} spectra)")
            mu = inventory.robust_mean_2d_t(Ys, 3.5)
            sigma = inventory.robust_mean_2d_t(abs(Ys - mu), 3.5)
            mu2 = mu/max(mu)

            bl = smooth.whittaker_smooth(
                mu2,
                func=func,
                func2_e=None, #inventory.relative_abs_max,
                tau2=tau2_bl,
                d=2)
            mu2_bl = mu2 - bl
            # plt.plot(xs, mu2_bl, color="m", alpha=0.75, label=r"baseline")

            # ss = inventory.robust_mean_1d(sigma, 3.5) / mu

            sigma_mu = sigma/mu

            plt.scatter(xs, sigma_mu, s=2, c='k', label=r"$\sigma/\mu$")
            sm = smooth.whittaker_smooth(sigma_mu, tau2=1.0e4)
            plt.plot(xs, sm, color="b", linewidth=1.5, label=r"$\sigma/\mu$ (smoothed)")

            plt.plot(xs, 0.5 + mu2_bl, color="Gray", alpha=0.5, label=r"$\mu/\mu_{\max} - \mu_{\text{bl}}$")
            mu2_bl_smoothed = smooth.whittaker_smooth(mu2_bl, tau2=tau2_smooth) #, W2=1+ss)
            plt.plot(xs, 0.5 + mu2_bl_smoothed, color="k", alpha=0.8, label=r"$\mu/\mu_{\max} - \mu_{\text{bl}}$ (smoothed)")

            # mu2_bl2 = smooth_pca(mu2_bl, n_component=n_component)
            mu2_bl2 = denoise(mu2_bl, n_component=n_component)
            plt.plot(xs, 0.75 + mu2_bl, color="Gray", marker='o', ms=1.0, alpha=0.5)
            plt.plot(xs, 0.75 + mu2_bl2, color="darkred", alpha=0.8, label=f"denoised")

            plt.hlines([0.05,0.1,0.2,0.5,], min(xs), max(xs), colors='r', linewidths=0.25, linestyles="--")
            plt.minorticks_on()
            plt.tight_layout()
            x_min, x_max = xrange
            plt.xlim(0.99*x_min, 1.01*x_max)
            plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            plt.show()
    #
    # def crop(self, start_index, end_index=None):
    #     Xs = self.x
    #     Ys = self.y
    #     N = len(Ys)
    #     for k in range(N):
    #         if end_index is None:
    #             Xs[k] = np.ascontiguousarray(Xs[k][start_index:])
    #             Ys[k] = np.ascontiguousarray(Ys[k][:,start_index:])
    #         else:
    #             Xs[k] = np.ascontiguousarray(Xs[k][start_index:end_index])
    #             Ys[k] = np.ascontiguousarray(Ys[k][:,start_index:end_index])
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
    def drop_notfinite(self):
        Ys = self.y
        for k in range(len(Ys)):
            Ys[k] = np.array([y for y in Ys[k] if np.all(np.isfinite(y)) and not np.any(np.isnan(y)) and not np.any(np.isinf(y))])
    #
    def scale_pca(self):
        from np.linalg import det
        for ys in self.y:
            Y = ys @ ys.T
            d = det(Y)
    #
    def isfinite(self):
        for k, ys in enumerate(self.y):
            m = np.sum(1 - np.isfinite(ys).astype("i"))
            if m > 0:
                print(f"Not all finite in serie {k}: {self.key}: {m}")
    #
    def multiplicative_scatter_correction(self):
        for k in range(len(self.y)):
            sp = self[k]
            sp.multiplicative_scatter_correction()
    #
    def drop_zeros_spectrum(self):
        Ys = self.y
        for k in range(len(Ys)):
            Ys[k] = np.array([y for y in Ys[k] if not np.all(y == 0)])
    #
    def derivative(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:-1] = y[:-1] - y[1:]
                y[-1] = 0
    #
    def scale_zs(self, delta=3.5):
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
    def scale_by_mean(self, power=1.0, scale=1.0):
        for k in range(len(self.y)):
            sp = self[k]
            sp.scale_by_mean(power=1.0, scale=scale)
    #
    def scale_by_total_mean(self, scale=1.0):
        for k in range(len(self.y)):
            sp = self[k]
            sp.scale_by_total_mean(scale=scale)
    #
    def scale_by_median(self, scale=1.0):
        for k in range(len(self.y)):
            sp = self[k]
            sp.scale_by_median(scale=scale)
    #
    def scale_by_total_median(self, scale=1.0):
        for k in range(len(self.y)):
            sp = self[k]
            sp.scale_by_total_median(scale=scale)
    #
    def scale_by_norm(self, scale=1.0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            for y in ys:
                y[:] = (y / np.sqrt(y @ y).mean()) * scale
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
                y[:] = smooth.whittaker_smooth(y, W2=W2, tau=tau, d=2)[0]
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
    def despike(self, tau=2.0, mode=None, beta=0.5):
        for ys in self.y:
            for y in ys:
                y[:] = despiking(y, tau, mode=mode, beta=beta)[0]
    #
    def denoise_svd(self, method='energy', threshold=0.9975, k=None, transpose=False):
        m = len(self.y)
        for i in range(m):
            self.y[i], _ = denoise_svd(self.y[i], method=method, threshold=threshold, k=k, transpose=transpose)
    #
    def robust_averaging(self, tau=3.5):
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
        # o.key = self.key
        # o.stderr = np.ascontiguousarray(dYs)
        return o
    #
    def copy(self):
        xx = [x.copy() for x in self.x]
        yy = [y.copy() for y in self.y]
        dd = SpectraSeries(xx, yy, self.attrs)
        return dd
    #
    def filter(self, kind="m-zscore", **kwds):
        N = len(self.y)
        for k in range(N):
            sp = self[k]
            sp.filter(kind=kind, **kwds)
        #     if kind == "m-zscore":
        #         tau = kwds.get("tau", 3.5)
        #         mu = inventory.robust_mean_2d_t(yk, tau=tau)
        #         dd = abs(yk - mu)
        #         z = inventory.modified_zscore(dd.sum(axis=1))
        #         mask = (abs(z) < tau)
        #         yk2 = np.array([y for m, y in zip(mask, yk) if m])
        #     else:
        #         yk2 = yk
        #     Ys.append(yk2)
        # self.y = Ys
    #
    def flatten(self):
        # Ys = tuple(self.y)
        y = np.concatenate(self.y, axis=0)
        y = np.ascontiguousarray(y)
        x = np.ascontiguousarray(self.x[0])

        o = Spectra(x, y, self.attrs)
        return o
    #
    def apply(self, func, inplace=True):
        if inplace:
            for x,ys in zip(self.x, self.y):
                for y in ys:
                    y2 = func(x, y)
                    y[:] = y2
        else:
            Ys = []
            for x,ys in zip(self.x, self.y):
                ys2 = np.empty_like(ys)
                for i, y in enumerate(ys):
                    y2 = func(x, y)
                    ys2[i,:] = y2
                Ys.append(ys2)
            self.y = Ys

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
    @property
    def key(self):
        return self.attrs["key"]
    #
    def ensure_xs(self):
        iter_xs = iter(self.x)
        x0 = next(iter_xs)
        for x in iter_xs:
            if not np.all(x == x0):
                raise TypeError("some xs not equal")
    #
    def copy(self):
        dd = SpectraSeries(self.x.copy(), self.y.copy(), self.attrs)
        return dd
    #
    def crop(self, start_x, end_x=None):
        mask = (self.x >= start_x)
        if end_x is not None:
            mask &= (self.x <= end_x)
        self.x = self.x[mask]
        self.y = self.y[:,mask]
        self.tau2_values = np.zeros_like(self.x)
        self.bs = np.zeros_like(self.y)
        self.ys_bs = np.zeros_like(self.y)
        self.ys_sm = self.y.copy()
    #
    def take(self, indices):
        self.y = np.ascontiguousarray(self.y[indices])
    #
    def replace_small_values(self, delta, value=0):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            np.putmask(ys, abs(ys)<delta, value)
    #
    def isfinite(self):
        for ys in self.y:
            m = np.sum(1 - np.isfinite(ys).astype("i"))
            if m > 0:
                print(f"Not all finite: {self.key}: {m}")
    #
    def drop_notfinite(self):
        self.y = np.array([y for y in self.y if np.all(np.isfinite(y))])
    #
    def drop_zeros_spectrum(self):
        self.y = np.array([y for y in self.y if not np.all(y == 0)])
    #
    def apply(self, func, inplace=True):
        if not inplace:
            Ys = []
        x = self.x
        for y in self.y:
            y2 = func(x, y)
            if inplace:
                y[:] = y2
            else:
                Ys.append(y)
        if not inplace:
            Ys = np.array(Ys)
            return Spectra(x, Ys, self.attrs)
    #
    def derivative(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            Ys[k,:-1] = ys[1:]-ys[:-1]
            Ys[k,-1] = 0
    #
    def filter(self, kind="m-zscore", **kwds):
        if kind == "m-zscore":
            tau = kwds.get("tau", 3.5)
            mu = inventory.median_2d_t(self.y)
            dd = abs(self.y - mu).sum(axis=1)
            z = inventory.modified_zscore(dd)
            mask = (abs(z) < tau)
            self.y = np.array([y for m, y in zip(mask, self.y) if m])
    #
    def flatten(self):
        return self
    #
    def allign_bottom(self):
        Ys = self.y
        for k in range(len(Ys)):
            ys = Ys[k]
            ys[:] = ys - ys.min()
    #
    # def apply_func(self, y_func, x_func=None, a=1, b=0):
    #     Ys = self.y
    #     xs = self.x
    #     for k in range(len(Ys)):
    #         ys = Ys[k]
    #         ys[:] = y_func(a*ys + b)
    #     if x_func is not None:
    #         for k in range(len(Xs)):
    #             xs[:] = x_func(a*xs + b)
    # #
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
    def select_for_exclusion(self, sigma_mu=None, alpha=None, clear=False):

        if self.excludes and clear:
            self.excludes = []

        i_slider = ipywidgets.IntSlider(value=0, min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        b_exclude = ipywidgets.Checkbox(value=(i_slider.value in self.excludes))

        button = ipywidgets.Button(description="Exclude selected")
        def button_click(_):
            self.exclude_selected()
            i_slider.max = len(self.y)-1
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
            fig = plt.figure("_select_for_exclusion", figsize=(12,5))
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
    def exclude_by_default(self, sigma_mu=None, alpha=3.5, clear=False):
        import matplotlib.pyplot as plt
        import ipywidgets

        N = len(self.y)
        if self.excludes and clear:
            self.excludes = []
        if not self.excludes:
            y = self.y
            mu = inventory.robust_mean_2d_t(y, tau=alpha)
            std = np.sqrt(inventory.robust_mean_2d_t((y - mu)**2, tau=alpha))
            z = (y - mu) / std
            zm =  abs(z).mean(axis=1)
            for i,v in enumerate(zm):
                if v > 3.0:
                    self.excludes.append(i)

        self.exclude_selected()
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
    def multiplicative_scatter_correction(self):
        self.y = multiplicative_scatter_correction(self.y)
    #
    def scale_by_mean(self, power=1.0, scale=1.0):
        N = len(self.y)
        if power == 1.0:
            for k in range(N):
                ys = self.y[k]
                mu = abs(ys).mean()
                ys[:] = (ys / mu) * scale
        else:
            for k in range(N):
                ys = self.y[k]
                mu = ((abs(ys) ** power).mean()) ** (1/power)
                ys[:] = (ys / mu) * scale
    #
    def scale_by_total_mean(self, scale=1.0):
        mu = abs(self.y).mean()
        self.y[:,:] = self.y / mu        
    #
    def scale_by_total_median(self, scale=1.0):
        mu = np.median(abs(self.y))
        self.y[:,:] = self.y / mu        
    #
    def scale_by_robust_zscore(self, tau=3.5, scale=1.0):
        N = len(self.y)
        for k in range(N):
            ys = self.y[k]
            # err = self.stderr[k]
            mu = inventory.robust_mean_1d(ys, tau=tau)
            absdev = abs(ys - mu)
            sigma = inventory.robust_mean_1d(absdev, tau=tau)
            ys[:] = scale * (ys - mu) / sigma
    #
    def scale_by_zscore(self, scale=1.0):
        N = len(self.y)
        for k in range(N):
            ys = self.y[k]
            # err = self.stderr[k]
            mu = ys.mean()
            sigma = ys.std()
            ys[:] = scale * (ys - mu) / sigma
    #
    def scale_by_modified_zscore(self, scale=1.0):
        N = len(self.y)
        for k in range(N):
            ys = self.y[k]
            mu = np.median(ys)
            absdev = abs(ys - mu)
            sigma = np.median(absdev)
            if sigma == 0:
                sigma = np.mean(absdev)
            ys[:] = scale * 0.6748 * (ys - mu) / sigma
    #
    def scale_by_modified_zscore_powered(self, scale=1.0):
        self.y = np.log(self.y)
        self.scale_by_modified_zscore(tau=tau, scale=1)
        self.y = np.exp(self.y)
        self.y *= scale
    #
    def scale_by_zscore_powered(self, scale=1.0):
        self.y = np.log(self.y)
        self.scale_by_zscore(scale=1)
        self.y = np.exp(self.y)
        self.y *= scale
    #
    def scale_by_max(self, scale=1.0):
        N = len(self.y)
        for k in range(N):
            ys = self.y[k]
            # err = self.stderr[k]
            mu = abs(ys).max()
            ys[:] = (ys / mu) * scale
            # err[:] = (err / mu) * scale
    #
    def despike(self, tau=2.0, mode=None, beta=0.5):
        for ys in self.y:
            ys[:] = despiking(ys, tau, mode=mode, beta=beta)[0]
    #
    def denoise_svd(self, method='energy', threshold=0.9975, k=None, transpose=False):
        #
        self.y, _ = denoise_svd(self.y, method=method, threshold=threshold, k=k, transpose=transpose)
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
    def smooth(self, method="irsa2", tau2=1.0, **kwargs):
        xs = self.x
        Ys = self.y
        if method == "runpy":
            for k in range(len(Ys)):
                ys = Ys[k]
                ys[:] = rampy.smooth(xs, ys, Lambda=tau2)
        elif method == "irsa":
            for k in range(len(Ys)):
                ys = Ys[k]
                ys[:] = smooth.whittaker_smooth(ys, tau2=tau2)
        elif method == "irsa2":
            func = kwargs.get("func", funcs.Square())
            func2 = kwargs.get("func2", None)
            for k in range(len(Ys)):
                ys = Ys[k]

                diff2_i = inventory.diff2(ys)
                mu_i = np.median(diff2_i)
                dd2_i = (np.percentile(diff2_i, 75) - np.percentile(diff2_i, 25)) / 2
                # dd2_k = np.median(abs(diff2_k - mu_k))

                if func2 is None:
                    def smooth_func2(x, mu=mu_i, scale=dd2_i):
                        v = (x - mu)/scale
                        return 1/(1 + v*v)
                else:
                    smooth_func2 = func2

                ys[:] = smooth.whittaker_smooth(
                                ys, tau2=tau2,
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
    def plot_smoothed(self, tau2=1000.0, tau1=0.0, override_tau2=True,
                         beta=1.0e2, xrange=None,
                         func=None, func2=None, func2_e=None,
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

        if xrange is None:
            xrange_slider = ipywidgets.FloatRangeSlider(
                value=(min(self.x), max(self.x)),
                min=min(self.x),
                max=max(self.x))
        else:
            xmin, xmax = xrange
            xrange_slider = ipywidgets.FloatRangeSlider(
                value=(xmin, xmax),
                min=min(self.x),
                max=max(self.x))
        xrange_slider.layout.width="80%"

        # button = ipywidgets.Button(description="Smooth baselines")
        # def button_click(_):
        #     self.replace_spectra_with_corrected()
        # button.on_click(button_click)
        # display(button)

        @ipywidgets.interact(i=i_slider, tau2=tau2_slider, #tau1=tau1_slider, 
                             xrange=xrange_slider, continuous_update=False)
        def _plot_smoothed(i, tau2, xrange):
            fig = plt.figure("_plot_smoothed", figsize=(12,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.5, alpha=0.5)

            ys_i = self.y[i]

            # W = np.ones(len(self.y[0]), "d")
            W2 = np.ones(len(self.y[0]), "d")
            if self.windows is not None:
                for xa,xb in self.windows:
                    W2[(xs >= xa) & (xs <= xb)] = beta

            nonlocal func

            plt.plot(xs, ys_i, linewidth=1.0, color='DarkBlue', marker='o', ms=1, label=f"current ({i})")

            ys_sm = smooth.whittaker_smooth(
                ys_i,
                func=func,
                func2=func2,
                func2_e=func2_e,
                tau2=tau2,
                d=d)

            plt.plot(xs, ys_sm, linewidth=1.0, color='r', label=rf"smoothed ($\lambda=${tau2:.1f})")

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
            plt.show()
    #
    def logscale(self):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        for k, ys in enumerate(self.y):
            self.y[k,:] = logscale(ys)
    #
    def loglog(self):
        self.x = np.log(self.x)
        for k,ys in enumerate(self.y):
            self.y[k,:] = np.log(ys - ys.min() + 1)
    #
    def plot_logscale(self):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        Zs = np.empty_like(self.y)
        for k, ys in enumerate(self.y):
            Zs[k,:] = logscale(ys)

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_logscale_items(i=i_slider):
            xs = self.x

            plt.close("_zscore_i")
            fig = plt.figure("plot_logscale_i", figsize=(13,4))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            plt.title(f"Logscale: {self.key} ({len(xs)} spectra)")

            for zs in Zs:
                plt.plot(xs, zs, linewidth=0.25, alpha=0.25)

            plt.plot(xs, Zs[i], color="k", linewidth=1.5)

            plt.minorticks_on()
            plt.show()
    #
    def plot_hist(self, bins=20):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_hist_items(i=i_slider):
            xs = self.x

            plt.figure(figsize=(13,4))
            plt.title(f"{self.key} ({len(xs)} spectra)")

            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.25, alpha=0.25)
            plt.plot(xs, self.y[i], color="k", linewidth=1.5)

            plt.grid(ls=':', lw=0.5)
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(13,3))
            plt.hist(self.y[i], bins=bins, rwidth=0.9)
            plt.grid(ls=':', lw=0.5)
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()
    #
    def plot_laser_tail(self, offset):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"
        ss_slider = ipywidgets.FloatSlider(min=0, max=200, step=1, value=1)
        ss_slider.layout.width="50%"
        dd_slider = ipywidgets.FloatSlider(min=0, max=100, step=1, value=1)
        dd_slider.layout.width="50%"

        @ipywidgets.interact(i=i_slider, ss=ss_slider, dd=dd_slider, continuous_update=False)
        def _plot_laser_tail(i=i_slider, ss=ss_slider, dd=dd_slider):
            xs = self.x

            pp = lambda xx, ss, dd: np.pow(xx/dd, ss/2-1) * np.exp(-xx / (2*dd))

            plt.figure(figsize=(13,4))
            plt.title(f"{self.key} ({len(xs)} spectra)")

            for ys in self.y:
                plt.plot(xs, ys, linewidth=0.25, alpha=0.25)
            
            plt.plot(xs, self.y[i], color="k", linewidth=1.5)

            xx = xs - offset
            ix_max = np.argmax(self.y[i])
            pv = pp(xx, ss, dd)
            pv = pv / np.max(pv) * self.y[i,ix_max]
            plt.plot(xs, pv, color="m", linewidth=1.5)
            plt.plot(xs, self.y[i] - pv, color="b", linewidth=1.5)
            

            plt.grid(ls=':', lw=0.5)
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()            
    #
    def plot_msc(self):
        import matplotlib.pyplot as plt
        import ipywidgets

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1)
        i_slider.layout.width="50%"

        y_mean = inventory.robust_mean_2d_t(self.y)

        @ipywidgets.interact(i=i_slider, continuous_update=False)
        def _plot_msc(i=i_slider):
            plt.figure(figsize=(13,4))
            plt.title(f"MSC {self.key} ({len(self.y)} spectra)")

            for ys in self.y:
                plt.scatter(y_mean, ys, s=4, alpha=0.25)

            y_i = self.y[i]
            y_mean_s, y_i_s = list(zip(*sorted(zip(y_mean, y_i))))
            plt.plot(y_mean_s, y_i_s, color="k", lw=1.5)
            plt.plot(y_mean_s, y_mean_s, color="b", lw=1.5, ls='--')
            plt.grid(ls=':', lw=0.5)
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()            
    #
    def select_baselines(self, tau2=1000.0, tau1=0.0, tau2_smooth=0.0, override_tau2=True,
                         bs_scale=3.0, alpha=0.001, eps=0.001, beta=1.0e2,
                         func=None, func2=None, func2_e=None, xrange=None, use_sigma=False,
                         d=2, **kwargs):

        N = len(self.y)

        xs = self.x

        if override_tau2:
            for i in range(N):
                self.tau2_values[i] = tau2
                self.bs[i,:] = 0
                self.ys_bs[i,:] = 0
        else:
            mask = (self.tau2_values == 0)
            if mask.sum() > 0:
                self.tau2_values[mask] = tau2
                self.bs[i,:] = 0
                self.ys_bs[i,:] = 0

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

        if xrange is None:
            xrange_slider = ipywidgets.FloatRangeSlider(
                value=(min(self.x), max(self.x)),
                min=min(self.x),
                max=max(self.x))
        else:
            xmin, xmax = xrange
            xrange_slider = ipywidgets.FloatRangeSlider(
                value=(xmin, xmax),
                min=min(self.x),
                max=max(self.x))
        xrange_slider.layout.width="80%"

        button = ipywidgets.Button(description="Subtract baselines")
        def button_click(_):
            self.replace_spectra_with_corrected()
        button.on_click(button_click)
        display(button)

        # b_all = ipywidgets.Checkbox(value=False, description="Recalculate_baselines")


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
                plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            ys_i = self.y[i]

            # W = np.ones(len(self.y[0]), "d")
            W2 = np.ones(len(self.y[0]), "d")
            if self.windows is not None:
                for xa,xb in self.windows:
                    W2[(xs >= xa) & (xs <= xb)] = beta

            nonlocal func
            if tau2_smooth > 0:
                ys_i_smooth = smooth.whittaker_smooth(
                        ys_i, tau2=tau2_smooth,
                        # W=W,
                        W2=W2,
                        d=2)
                sigma = (ys_i - ys_i_smooth).std()
                if func is None:
                    func = funcs.Step(sigma)
                # plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkGreen', label=f"current smoothed ({i})")
            else:
                ys_i_smooth =  ys_i

            plt.plot(xs, ys_i_smooth, linewidth=1.5, color='DarkBlue', label=f"current ({i})")

            scale = np.abs(ys_i_smooth).mean()

            bs = smooth.whittaker_smooth(
                ys_i_smooth / scale,
                func=func,
                func2=func2,
                func2_e=func2_e,
                tau2=tau2,
                tau1=tau1,
                use_sigma=use_sigma,
                d=d)

            bs *= scale
            self.bs[i,:] = bs
            self.ys_bs[i,:] = self.y[i] - bs

            plt.plot(xs, bs, linewidth=1.5, color='m', label=f"baseline")

            x_min, x_max = xrange
            plt.xlim(0.99*x_min, 1.01*x_max)

            # bs_xrange = bs[(x_min <= xs) & (xs <= x_max)]
            # bs_max, bs_min = np.max(bs_xrange), np.min(bs_xrange)
            ys_xrange = ys_i[(x_min <= xs) & (xs <= x_max)]
            plt.ylim(scale_min(ys_xrange.min()), scale_max(ys_xrange.max()))
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
            ax2 = plt.subplot(2,1,2, sharex=ax1)
            for ys_bs in self.ys_bs:
                plt.plot(self.x, ys_bs, linewidth=0.5, alpha=0.25)
            plt.plot(self.x, self.ys_bs[i], linewidth=0.75, color='DarkBlue', label="current")
            ms = []
            for i,ys_bs in enumerate(self.ys_bs):
                if not np.all(ys_bs == 0):
                    ms.append(i)
            if ms:
                ms = np.array(ms)
                ys_bs = self.ys_bs[ms]
                plt.plot(self.x, inventory.robust_mean_2d_t(ys_bs, 3.5), linewidth=1.5, color='k', label="mean")
            plt.xlim(0.99*x_min, 1.01*x_max)
            # plt.ylim(0, np.max(ys_i))
            plt.legend()
            plt.minorticks_on()
            plt.grid(1)
            ax1.label_outer()
            ax2.label_outer()
            plt.tight_layout()
            plt.show()
    #
    # def select_baseline_param(self,
    #                      tau2=1000.0,
    #                      func=None, func2=None, func2_e=None,
    #                      d=2,
    #                      override_tau2=False,
    #                      tau2_smooth=1.0, beta=100,
    #                      **kwargs):
    #     import matplotlib.pyplot as plt
    #     import ipywidgets

    #     xs = self.x

    #     if override_tau2 or not self.tau2s:
    #         self.tau2s = 3*[tau2]

    #     tau2_low_slider = ipywidgets.FloatSlider(value=self.tau2s[0], 
    #                                              min=self.tau2s[0]/50, 
    #                                              max=self.tau2s[0]*25, 
    #                                              step=self.tau2s[0]/100)
    #     tau2_low_slider.layout.width="80%"
    #     tau2_mid_slider = ipywidgets.FloatSlider(value=self.tau2s[1], 
    #                                             min=self.tau2s[1]/50, 
    #                                             max=self.tau2s[1]*25, 
    #                                             step=self.tau2s[1]/100)
    #     tau2_mid_slider.layout.width="80%"
    #     tau2_high_slider = ipywidgets.FloatSlider(value=self.tau2s[2], 
    #                                              min=self.tau2s[2]/50, 
    #                                              max=self.tau2s[2]*25, 
    #                                              step=self.tau2s[2]/100)
    #     tau2_high_slider.layout.width="80%"

    #     def tau2_low_on_value_change(change):
    #         self.tau2s[0] = tau2_low_slider.value
    #     def tau2_mid_on_value_change(change):
    #         self.tau2s[1] = tau2_mid_slider.value
    #     def tau2_high_on_value_change(change):
    #         self.tau2s[2] = tau2_high_slider.value

    #     tau2_low_slider.on_trait_change(tau2_low_on_value_change, name="value")
    #     tau2_high_slider.on_trait_change(tau2_high_on_value_change, name="value")
    #     tau2_mid_slider.on_trait_change(tau2_mid_on_value_change, name="value")

    #     ys_high = np.quantile(self.y, 0.83333, axis=0)
    #     ys_mid = np.quantile(self.y, 0.5, axis=0)
    #     ys_low = np.quantile(self.y, 0.16666, axis=0)

    #     self.Ys = np.vstack((ys_low, ys_mid, ys_high))
    #     labels = ["low", "mid", "high"]

    #     W2 = np.full(len(ys_high), 10.0, "d")
    #     if self.windows is not None:
    #         for xa,xb in self.windows:
    #             W2[(xs >= xa) & (xs <= xb)] = beta

    #     Ys_smooth = []
    #     # scale = ys_mid.mean()
    #     for ys in self.Ys:
    #         ys_smooth = smooth.whittaker_smooth(
    #             ys, tau2=tau2_smooth,
    #             W2=W2,
    #             d=2)
    #         Ys_smooth.append(ys_smooth)


    #     @ipywidgets.interact(tau2_high=tau2_high_slider,
    #                          tau2_mid=tau2_mid_slider,
    #                          tau2_low=tau2_low_slider,
    #                          continuous_update=False)
    #     def _plot_select_baseline_param(tau2_high, tau2_mid, tau2_low):
    #         nonlocal func

    #         fig = plt.figure("select_baseline_param", figsize=(12,5))
    #         fig.canvas.header_visible = False
    #         fig.canvas.footer_visible = False
    #         fig.canvas.toolbar_position = 'right'
    #         fig.canvas.toolbar_visible = True
    #         fig.clear()
    #         plt.title(self.key, fontdict={"size":10}, loc="left")
    #         xs = self.x

    #         colors=["DarkRed", "DarkGreen", "DarkBlue"]
    #         colors2=["Chocolate", "Lime", "DodgerBlue"]
    #         for i in range(3):
    #             plt.plot(xs, self.Ys[i], linewidth=1.25, color=colors[i], label=labels[i])

    #         Bs = []
    #         for i in range(3):
    #             if func is None:
    #                 sigma = abs(self.Ys[i] - Ys_smooth[i]).mean()
    #                 func = funcs.Step(sigma)

    #             bs = smooth.whittaker_smooth(
    #                 Ys_smooth[i],
    #                 func=func,
    #                 func2=func2,
    #                 tau2=self.tau2s[i],
    #                 d=d,
    #                 func2_mode=func2_mode)
    #             Bs.append(bs)

    #         for i in range(3):
    #             if i == 0:
    #                 plt.plot(xs, Bs[i], linewidth=1.25, color=colors2[i], label="baseline")
    #                 # plt.plot(xs, self.Ys[i]-Bs[i], linewidth=0.75, color='Grey', label="corrected")
    #             else:
    #                 plt.plot(xs, Bs[i], linewidth=1.25, color=colors2[i])
    #                 # plt.plot(xs, self.Ys[i]-Bs[i], linewidth=0.75, color='Grey')

    #         self.Bs = Bs

    #         plt.minorticks_on()
    #         plt.grid(1)
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    # #
    
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
#                 bs = smooth.whittaker_smooth(
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
    def subtract_baselines(self, func=None, func2=None, func2_e=None, d=2, tau2=1.0e3, tau1=0, verbose=True, n_jobs=1):
        import ipywidgets
        from IPython.display import display

        N, n = self.y.shape
        if verbose and n_jobs == 1:
            fp = ipywidgets.FloatProgress(min=0, max=N, value=0, bar_style="success",
                                          description=self.key)
            fp.layout.width = "auto"
            display(fp)
        else:
            fp = None
            print(self.key)


        def _subtract_baseline(i, y_i):
            scale = abs(y_i).mean()
            bl = smooth.whittaker_smooth(
                y_i / scale,
                func=func,
                func2=func2,
                func2_e=func2_e,
                tau2=tau2,
                tau1=tau1,
                d=d)
            bl *= scale
            y_i[:] = y_i - bl
            if verbose and n_jobs == 1:
                fp.value = i+1
            return y_i

        if n_jobs <= 0:
            n_jobs = 1

        if n_jobs == 1:
            for i in range(N):
                y_i = self.y[i]
                y_i = _subtract_baseline(i, y_i)
                # bl = smooth.whittaker_smooth(
                #     y_i,
                #     func=func,
                #     func2=func2,
                #     func2_e=func2_e,
                #     tau2=tau2,
                #     tau1=tau1,
                #     d=d)
                # self.y[i,:] = y_i - bl
                if verbose:
                    fp.value = i+1
        else:
            with joblib.Parallel(n_jobs=n_jobs, max_nbytes=None) as parallel:
                result = parallel(
                    joblib.delayed(_subtract_baseline)(i, self.y[i]) for i in range(N)
                )
            for y_i, y in zip(self.y, result):
                y_i[:] = y
        if verbose and n_jobs == 1:
            fp.close()
    #
    def smooth_spectra(self, func=None, func2=None, func2_e=None, d=2, tau2=1.0e3, tau1=0, verbose=True, n_jobs=1):
        import ipywidgets
        from IPython.display import display

        N, n = self.y.shape
        if verbose and n_jobs == 1:
            fp = ipywidgets.FloatProgress(min=0, max=N, value=0, bar_style="success",
                                          description=self.key)
            fp.layout.width = "auto"
            display(fp)
        else:
            fp = None
            print(self.key)


        def _smooth_spectra(i, y_i):
            scale = abs(y_i).mean()
            yy = smooth.whittaker_smooth(
                y_i / scale,
                func=func,
                func2=func2,
                func2_e=func2_e,
                tau2=tau2,
                tau1=tau1,
                d=d)
            yy *= scale
            if verbose and n_jobs == 1:
                fp.value = i+1
            return yy

        if n_jobs <= 0:
            n_jobs = 1

        if n_jobs == 1:
            for i in range(N):
                y_i = self.y[i]
                y_i[:] = _smooth_spectra(i, y_i)
                if verbose:
                    fp.value = i+1
        else:
            with joblib.Parallel(n_jobs=n_jobs, max_nbytes=None) as parallel:
                result = parallel(
                    joblib.delayed(_smooth_spectra)(i, self.y[i]) for i in range(N)
                )
            for y_i, yy in zip(self.y, result):
                y_i[:] = yy
        if verbose and n_jobs == 1:
            fp.close()
    #
    def plot_egor(self, W, S):
        import sys
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1, description="i")
        i_slider.layout.width="50%"

        @ipywidgets.interact(ii=i_slider, continuous_update=False)
        def _plot_egor(ii):
            fig = plt.figure("_plot_egor", figsize=(12,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            plt.title(self.key, fontdict={"size":10}, loc="left")
            xs = self.x
            xs2 = xs[:-W-1]
            Ys2 = []
            for ys in self.y:
                m = len(ys)
                ys1 = np.zeros_like(ys)
                for i in range(m-W-1):
                    dy = 0
                    for j in range(1,W+1):
                        dy += ys[i+j] - ys[i]
                    dy /= W
                    ys1[i] = dy
                ys2 = ys1 **2 / S - ys1
                ys2 = ys2[:-W-1]
                Ys2.append(ys2)
                plt.plot(xs2, ys2, linewidth=0.5, alpha=0.25)

            # print(i)
            # sys.stdout.flush()
            ys2_i = Ys2[ii]
            plt.plot(xs2, ys2_i, linewidth=1.0, color='DarkBlue',
                     label="current", alpha=0.5)

            plt.minorticks_on()
            plt.tight_layout()
            # plt.xlim(x_min, x_max)
            plt.legend()
            plt.show()
    #
    def topf(self, tval=0, crop=0, sub_bg=False, sub_valleys=True):
        import sys
        import quick_topf as topf

        transformer = topf.PersistenceTransformer()
        xs = self.x
        Ys = []
        for ys in self.y:
            # data = np.c_[xs, ys]
            # peaks = transformer.fit_transform(data)
            # # y = peaks[:,1]
            # y = np.ascontiguousarray(peaks[:,1])
            # y[0] = 0
            y = topf.get_peaks(xs, ys, transposed=True)[1]
            if tval > 0:
                tv = np.percentile(y[y > 0], tval)
                y[y < tv] = 0
            if sub_valleys:
                z = topf.get_valleys(xs, ys, transposed=True)[1]
                z[-10:] = 0
                if tval > 0:
                    tv = np.percentile(z[z > 0], tval)
                    z[z < tv] = 0
                y -= z
            elif sub_bg:
                y = topf.subtract_background(ys, y)
            Ys.append(y)

        if crop > 0:
            mask = (xs >= crop)
            # if end is not None:
            #     mask &= (self.x <= end)
            xs = self.x[mask]
            # self.y = self.y[:,mask]
            Ys = [y[mask] for y in Ys]

        Ys = np.array(Ys)

        sp = Spectra(xs, Ys, attrs=self.attrs)
        return sp
    #
    def plot_topf(self, tval=0, n_peaks=None, crop=0):
        import sys
        import quick_topf as topf
        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1, description="i")
        i_slider.layout.width="50%"

        transformer = topf.PersistenceTransformer()
        xs = self.x
        Ys = self.y
        Peaks = []
        Valleys = []
        for ys in self.y:
            peaks = topf.get_peaks(xs, ys, transposed=True, n_peaks=n_peaks)
            valleys = topf.get_valleys(xs, ys, transposed=True, n_peaks=n_peaks)
            valleys[1,-10:] = 0
            y = peaks[1]
            z = valleys[1]
            if tval > 0:
                tv = np.percentile(y[y > 0], tval)
                y[y < tv] = 0
                tv = np.percentile(z[z > 0], tval)
                z[z < tv] = 0
            Peaks.append(peaks)
            Valleys.append(valleys)

        if crop > 0:
            mask = (xs >= crop)
            # if end is not None:
            #     mask &= (self.x <= end)
            xs = self.x[mask]
            Ys = [y[mask] for y in Ys]
            Peaks = [peaks[:,mask] for peaks in Peaks]
            Valleys = [valleys[:,mask] for valleys in Valleys]

        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(xs), max(xs)),
            min=min(xs),
            max=max(xs))
        xrange_slider.layout.width="80%"

        @ipywidgets.interact(ii=i_slider, xrange=xrange_slider, continuous_update=False)
        def _plot_topf(ii, xrange):
            # nonlocal Ys, xs

            fig = plt.figure("_plot_topf", figsize=(12,6))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            plt.subplot(2,1,1)
            plt.title(self.key, fontdict={"size":10}, loc="left")
            # for ys in Ys:
            #     plt.plot(xs, ys, linewidth=0.5, alpha=0.25)

            peaks = Peaks[ii]
            valleys = Valleys[ii]
            ys = Ys[ii]
            p1_mask = peaks[1] > 0
            v1_mask = valleys[1] > 0
            # p0_mask = peaks[0] > 0
            plt.plot(xs, ys, linewidth=1, color='k') #, marker='s', markersize=2)
            plt.vlines(xs[p1_mask], 0, peaks[1,p1_mask], linewidth=1, color='r') #, marker='s', markersize=2)
            plt.vlines(xs[v1_mask], -valleys[1,v1_mask], 0, linewidth=1, color='b') #, marker='s', markersize=2)
            plt.vlines(xs[p1_mask], ys[p1_mask]-peaks[1,p1_mask], ys[p1_mask], linewidth=0.75, color='r',
                       label=f"#peaks: {(peaks[1,p1_mask]>0).sum()}")
            plt.vlines(xs[v1_mask], ys[v1_mask]+valleys[1,v1_mask], ys[v1_mask], linewidth=0.75, color='b',
                       label=f"#peaks: {(valleys[1,v1_mask]>0).sum()}")
            # zs_ii = Zs[ii]
            # plt.plot(xs[zs_ii > 0], ys[zs_ii > 0], linewidth=1, color='m', marker='s', markersize=2)
            plt.minorticks_on()
            plt.grid(1)
            plt.legend()

            x_min, x_max = xrange
            plt.xlim(0.99*x_min, 1.01*x_max)

            plt.subplot(2,1,2)
            ys = Ys[ii]
            ysmax = np.max(ys)
            peaks = Peaks[ii][1]
            valleys = Valleys[ii][1]

            plt.plot(xs, ys, linewidth=1, color='k') #, marker='s', markersize=2)
            plt.vlines(xs[valleys > 0], 0, ys[valleys > 0], linewidth=0.5, color='r', linestyles='--')
            # plt.vlines(xs[up_ii > 0], 0, (ymax - us_ii - up_ii)[up_ii > 0], linewidth=1.0, color='r')
            plt.plot(xs[valleys > 0], ys[valleys > 0], linewidth=1, color='m', marker='s', markersize=2)
            plt.minorticks_on()
            plt.grid(1)

            x_min, x_max = xrange
            plt.xlim(0.99*x_min, 1.01*x_max)

            # plt.subplot(3,1,3)
            # # for y in Ys_p:
            # #     plt.plot(xs, y, linewidth=0.5, alpha=0.25)

            # Zs[ii][-10:]=0
            # plt.vlines(xs, 0, Ys_p[ii], linewidth=0.75, color='r', label=f"#peaks: {(Ys_p[ii]>0).sum()}")
            # plt.vlines(xs, -Zs[ii], 0, linewidth=0.75, color='g', label=f"#valleys: {(Zs[ii]>0).sum()}")
            # # plt.plot(xs, Ys[ii], linewidth=1, color='k')
            # plt.minorticks_on()
            # plt.grid(1)

            # x_min, x_max = xrange
            # plt.xlim(0.99*x_min, 1.01*x_max)
            # plt.legend()


            # plt.tight_layout()
            # # plt.xlim(x_min, x_max)
            plt.show()
    #
    def plot_spectra(self, smoothing=False, kind="wh", #display_data=False,
                     marker='s', ms=0, lw=0.5, plot_mean=True, **kwds):
        # from sklearn.decomposition import PCA, TruncatedSVD

        tau_slider = ipywidgets.FloatSlider(value=3.5, min=1.0, max=3.5, step=0.1, readout_format=".3f")
        tau_slider.layout.width="80%"

        alpha_slider = ipywidgets.FloatSlider(value=3.5, min=1.0, max=3.5, step=0.1)
        alpha_slider.layout.width="80%"

        if kind == "despike":
            alpha_slider.value = kwds.get("alpha", 3.5)
            alpha_slider.min = 1.0
            alpha_slider.max = 4.0
            alpha_slider.step = 0.1
        elif kind in ["wh", "wh_z"]:
            tau0 = kwds.get("tau", 1.0)
            alpha = kwds.get("alpha", 2.0)
            tau_slider.min = tau0/100
            tau_slider.max = 10 * tau0
            tau_slider.step = tau0/100
            tau_slider.value = tau0
            alpha_slider.value = alpha
        elif kind in ["denoise", "pca", "svd"]:
            tau0 = kwds.get("tau", 100)
            tau_slider.value = tau0
            tau_slider.min = 1
            tau_slider.max = len(self.x)
            tau_slider.step = 1

        if not smoothing:
            # tau_slider.layout.visibility = 'hidden'
            # alpha_slider.layout.visibility = 'hidden'
            tau_slider.layout.display = 'none'
            alpha_slider.layout.display = 'none'

        i_slider = ipywidgets.IntSlider(min=0, max=len(self.y)-1, description="i")
        i_slider.layout.width="50%"

        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x), max(self.x)),
            min=min(self.x),
            max=max(self.x),
            description="xrange",)
        xrange_slider.layout.width="75%"

        if smoothing and kind == "svd":
            k = kwds.get("k", None)
            if k is None:
                method = kwds.get("method", "energy")
                threshold = kwds.get("threshold", 0.995)
                transpose = kwds.get("transpose", True)
                Ys, info = denoise_svd(self.y, method=method, threshold=threshold, transpose=transpose)
                print(info["k"], )
            else:
                Ys, info = denoise_svd(self.y, method=None, k=k)
                print(info["k"],)
        else:
            Ys = self.y

        @ipywidgets.interact(i=i_slider, xrange=xrange_slider, tau=tau_slider, alpha=alpha_slider, continuous_update=False)
        def _plot_spectra(i, xrange, tau, alpha):
            # fig = plt.figure("_plot_spectra", figsize=(12,4.5))
            fig = plt.figure(figsize=(12,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            # fig.clear()
            plt.title(self.attrs["key"], fontdict={"size":10}, loc="left")
            xs = self.x

            nonlocal Ys

            # if kind == "svd":
            #     Ys = denoise_svd(self.y, method="elbow")
            # else:
            #     Ys = self.y

            for ys in Ys:
                plt.plot(xs, ys, color='gray', linewidth=lw, alpha=0.33, marker=marker, ms=ms)

            ys_i = Ys[i]
            if smoothing:
                plt.plot(xs, ys_i, linewidth=1.0, color='DarkBlue', label="current", marker=marker, ms=ms, alpha=0.5)
            else:
                plt.plot(xs, ys_i, linewidth=1.0, color='DarkBlue', label="current")
                # plt.hlines(ys_i[abs(inventory.zscore(ys_i)) <= 2].mean(), xs.min(), xs.max(), linewidth=1.0, color='DarkBlue', label="current")
                if plot_mean:
                    plt.plot(xs, inventory.robust_mean_2d_t(self.y), linewidth=1.0, color='k', label="mean")

            out_mask = None
            if smoothing:
                if kind == "wh":
                    beta = kwds.get("beta", 100)
                    ys_i_smooth = smooth_spectra(xs, ys_i, tau2=tau, windows=self.windows, beta=beta, )
                elif kind == "wh_z":
                    mode = kwds.get("mode", None)
                    ys_i_smooth = smooth.whittaker_smooth(ys_i, tau2=tau,
                                                          func=funcs.SoftAbs_Sqrt(1.0e-5).derivative_div_array,
                                                          func2=funcs.SoftAbs_Sqrt(1.0e-5).derivative_div_array,
                                                          # func2_e=inventory.relative_abs_max
                                                        )
                    dy = ys_i - ys_i_smooth
                    dz = inventory.modified_zscore(dy)
                    if mode == "+":
                        out_mask = (dz > alpha)
                    elif mode == "-":
                        out_mask = (dz < -alpha)
                    else:
                        out_mask = (abs(dz) > alpha)
                    for i, ma in enumerate(out_mask):
                        xx1 = abs(dz[i] - X[i-1])
                        xx2 = abs(X[i+1] - X[i])
                        xx_min, xx_max = min(xx1, xx2), max(xx1, xx2)
                        if xx_min < xx_max / 2:
                            mask[i] = False

                    ws = np.ones_like(ys_i)
                    ws[out_mask] = 0
                    ys_ws = smooth.whittaker_smooth_base(ys_i, tau2=tau, W=ws)
                    ys_i_smooth[out_mask] = ys_ws[out_mask]
                elif kind == "pca":
                    # n_component = kwds.get("n_component", None)
                    ys_i_smooth, info = smooth_pca(ys_i, n_component=int(tau))
                elif kind == "svd":
                    ys_i_smooth = Ys[i]
                elif kind == "denoise":
                    # n_component = kwds.get("n_component", None)
                    ys_i_smooth = denoise(ys_i, n_component=int(tau))
                elif kind == "despike":
                    mode = kwds.get("mode", None)
                    beta = kwds.get("beta", 0.5)
                    ys_i_smooth, out_mask = despiking(ys_i, tau=alpha, mode=mode, beta=beta)
                else:
                    ys_i_smooth = ys_i
                plt.plot(xs, ys_i_smooth, linewidth=1.0, color='DarkRed', label=f"smoothed")
                if out_mask is not None:
                    plt.plot(xs[out_mask], ys_i[out_mask], lw=0, color='red', marker='o', ms=4, label=f"outliers")
            else:
                ys_i_smooth = ys_i

            fract = kwds.get("fract", False)
            if fract:
                curve_points = np.empty((len(ys_i),2))
                curve_points[:,0] = xs #range(len(ys_i))
                curve_points[:,1] = ys_i
                fd = hausdorff_dimension(curve_points)["dimension"]
                print(f"fractal dimension: {fd:.2f}")

            x_min, x_max = xrange
            ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
            ymin, y_max = 0.9*np.min(ys_range), 1.5*np.max(ys_range)
            # plt.ylim(ymin, y_max)

            plt.grid(ls=':', lw=0.75)
            plt.minorticks_on()
            plt.tight_layout()
            plt.xlim(x_min, x_max)
            plt.legend()
            plt.show()
            if kind == "pca":
                plt.figure(figsize=(10,3))
                plt.title(f"{info["ls"].sum()}")
                plt.plot(info["ls"], marker='o')
                plt.show()
    #
    def plot_mu_sigma(self, alpha=1.0):
        # from sklearn.decomposition import PCA, TruncatedSVD

        xrange_slider = ipywidgets.FloatRangeSlider(
            value=(min(self.x), max(self.x)),
            min=min(self.x),
            max=max(self.x),
            description="xrange",)
        xrange_slider.layout.width="50%"

        @ipywidgets.interact(xrange=xrange_slider, continuous_update=False)
        def _plot_mu_sigma(xrange):
            fig = plt.figure("_plot_mu_sigma", figsize=(12,4.5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            plt.title(self.attrs["key"], fontdict={"size":10}, loc="left")
            xs = self.x
            for ys in self.y:
                plt.plot(xs, ys, lw=0.25, alpha=0.5)

            mu = self.y.mean(axis=0)
            sigma = self.y.std(axis=0)
            plt.plot(xs, mu, lw=1.5, color='k')
            plt.fill_between(xs, mu - alpha*sigma, mu + alpha*sigma, color='lightblue', alpha=0.75, label=r'$\bar\sigma/\bar\mu$='+f"{abs(sigma/mu).mean():.2f}")

            x_min, x_max = xrange
            # ys_range = ys_i[(x_min <= xs) & (xs <= x_max)]
            # ymin, y_max = 0.9*np.min(ys_range), 1.2*np.max(ys_range)
            # plt.ylim(ymin, y_max)

            plt.grid(ls=':', lw=0.75)
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

        if kind == "AM":
            alpha_arg = fixed(alpha)
        else:
            alpha_slider = ipywidgets.FloatSlider(value=alpha, min=0.0, max=1.0, step=0.01)
            alpha_slider.layout.width="50%"
            alpha_arg = alpha_slider

        # i_slider.on_trait_change(i_on_value_change, name="value")

        @ipywidgets.interact(alpha=alpha_arg, continuous_update=False)
        def _plot_pca_spectra(alpha):
            fig = plt.figure("_plot_pca_spectra", figsize=(12,5))
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.toolbar_position = 'right'
            fig.clear()
            Ys = self.y

            nonlocal n_component
            if n_component is None:
                n_component = Ys.shape[0]
            if kind == "AM":
                c, As, Ls = pca.find_loc_and_pc(Ys, n_component)
            else:
                wma = averaging_function(kind, kwds={'alpha':alpha})
                c, As, Ls = pca.find_robust_loc_and_pc(Ys, wma, n_component)
            Ls /= Ls.sum()
            plt.subplot(1,2,1)
            plt.title(f"{self.key} ({len(self.y)} spectra)")
            # m = 1 + (Ls.cumsum() <= alpha).sum()
            plt.plot(Ls, marker='o', markersize=4)
            plt.minorticks_on()
            plt.grid(1)
            # plt.legend()
            plt.subplot(1,2,2)
            if kind == "AM":
                c, As, Ls = pca.find_loc_and_pc(Ys, 2)
            else:
                wma = averaging_function(kind, kwds={'alpha':alpha})
                c, As, Ls = pca.find_robust_loc_and_pc(Ys, wma, 2)
            U = (Ys - c) @ As.T
            plt.scatter(U[:,0], U[:,1], c='w', edgecolors='k', s=36)
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()
    #
    def pca_transform_spectra(self, n_component=2, kind="AM", alpha=0.95):
        Ys = self.y
        if kind == "AM":
            c, As, Ls = pca.find_loc_and_pc(Ys, n_component)
        else:
            wma = averaging_function(kind, kwds={'alpha':alpha})
            c, As, Ls = pca.find_robust_loc_and_pc(Ys, wma, n_component)
        Ls /= Ls.sum()
        U = (Ys - c) @ As.T
        return U, Ls
    #

class SpectraCollection:
    #
    def __init__(self, spectra=None):
        if spectra is None:
            self.spectra = {}
        else:
            self.spectra = {key:spectra[key] for key in sorted(spectra.keys())}
        self.wg_select_key = None
        self.current_key = None

        self.active_windows = []
    #
    def copy(self):
        dd = SpectraCollection()
        for key, sp in self.spectra.items():
            dd.spectra[key] = sp.copy()
        return dd
    #
    def __getitem__(self, key):
        return self.spectra[key]
    #
    def __setitem__(self, key, val):
        self.spectra[key] = val
    #
    def update(self, spcols, force=False):
        for key, sp in spcols.items():
            if not force and key in self.spectra:
                raise TypeError(f"Spectra collection already has {key}")
            self.spectra[key] = sp
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
    def subcollection_by_attr_value(self, name, val):
        col = SpectraCollection()
        for key, sp in self.spectra.items():
            if sp.attrs[name] == val:
                col.spectra[key] = sp
        return col
    #
    def filter_by_attrs(self, **kwds):
        kwds_items = list(kwds.items())
        for sp in self.spectra.values():
            flag = True
            attrs = sp.attrs
            for name, val in kwds_items:
                if attrs[name] != val:
                    flag = False
                    break
            if flag:
                yield sp
    #
    def union(self, key=None, **kwds):
        ys_list = []
        xs = None
        for sp in self.filter_by_attrs(**kwds):
            # print(sp.y.shape, sp.key)
            ys_list.append(sp.y)
            if xs is None:
                xs = sp.x
            # elif not np.all(xs == sp.x):
            #     raise TypeError("Different xs in spectra collection")
        ys = np.vstack(ys_list)

        attrs = {name:val for name,val in kwds.items()}
        if type(xs) is list:
            xs = xs[0]
        sp = Spectra(xs, ys, attrs)
        if key is None:
            sp.attrs["key"] = "_".join(kwds.values())
        else:
            sp.attrs["key"] = key
        return sp
    #
    def group_by(self, name, val):
        d = {}
        for sp in self.spectra.values():
            if attrs[name] == val:
                col = d.get(val, None)
                if col is None:
                    col = SpectraCollection()
                    col.key = val
                    col.attrs={k:v for k,v in sp.attrs.items() if k != name}
                    d[val] = col
                col.spectra[sp.key] = sp
        return d
    #
    def same_x(self):
        xs = None
        for sp in self.spectra.values():
            if xs is None:
                xs = sp.x
            elif not np.all(xs == sp.x):
                return None
        return xs
    #
    def union_y(self):
        ys_tuple = tuple(sp.y for sp in self.spectra.values())
        ys = np.vstack(ys_tuple)
        return ys
    #
    def _intersect_attrs(self):
        attrs_gen = ({(name,val) for name,val in sp.attrs.items()} for sp in self.spectra.values())
        attrs_iter = attrs_gen()
        attrs = next(attrs_iter)
        for next_attrs in attrs_iter:
            attrs &= next_attrs
        return {name:val for name,val in attrs}
    #
    def select_key(self):
        if self.wg_select_key is None:
            keys = list(self.keys())
            self.wg_select_key = wg_select = \
                    ipywidgets.Dropdown(options=keys, value=keys[0], description="Spectra key:")
            wg_select.layout.width="75%"
            self.current_key = keys[0]

            def on_value_change(change):
                self.current_key = self.wg_select_key.value

            wg_select.on_trait_change(on_value_change, name="value")
        else:
            wg_select = self.wg_select_key

        return wg_select
    #
    def apply(self, func, inplace=True, verbose=True):
        if verbose:
            twg = ipywidgets.Text(value="")
            twg.layout.width="75%"
            display(twg)
        if inplace:
            for key, sp in self.spectra.items():
                if verbose:
                    twg.value = key
                sp.apply(func, inplace=True)
            if verbose:
                twg.value = "OK"
        else:
            col = SpectraCollection()
            for key, sp in self.spectra.items():
                if verbose:
                    twg.value = key
                sp2 = sp.apply(func, inplace=False)
                col[key] = sp2
            if verbose:
                twg.value = "OK"
            return col
    #
    def apply_par(self, func, inplace=True, verbose=True):
        from joblib import Parallel, delayed
        if inplace:
            Parallel(n_job=-1, backend="loky")(
                delayed(sp.apply)(func, inplace=True, verbose=verbose) for key, sp in self.spectra.items()
            )
        else:
            col = SpectraCollection()
            for key, sp in self.spectra.items():
                sp2 = sp.apply(func, inplace=False, verbose=verbose)
                col[key] = sp2
            return col
    #
    def align_bottom(self):
        for sp in self.spectra.values():
            sp.align_bottom()
    #
    def isfinite(self):
        for sp in self.spectra.values():
            sp.isfinite()
    #
    def drop_notfinite(self):
        for sp in self.spectra.values():
            sp.drop_notfinite()
    #
    def drop_zeros_spectrum(self):
        for sp in self.spectra.values():
            sp.drop_zeros_spectrum()
    #
    def drop_empty(self):
        empty_keys = []
        for key in self.spectra.keys():
            sp = self.spectra[key]
            if len(sp.y) == 0:
                empty_keys.append(key)
        for key in empty_keys:
            del self.spectra[key]
    #
    def despike(self, tau=3.0, mode=None, beta=0.5):
        for sp in self.spectra.values():
            sp.despike(tau=tau, mode=mode, beta=beta)
    #
    def denoise_svd(self, method='energy', threshold=0.9975, k=None, transpose=False):
        for sp in self.spectra.values():
            sp.denoise_svd(method=method, threshold=threshold, k=k, transpose=transpose)
    #
    def derivative(self):
        for sp in self.spectra.values():
            sp.derivative()
    #
    def logscale(self):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_logscale(key):
            sp = self.spectra[key]
            sp.logscale()
    #
    def loglog(self):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _logscale(key):
            sp = self.spectra[key]
            sp.loglog()
    #
    def plot_logscale(self):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_logscale(key):
            sp = self.spectra[key]
            sp.plot_logscale()
    #
    def plot_egor(self, W, S):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_egor2(key):
            sp = self.spectra[key]
            sp.plot_egor(W, S)
    #
    def plot_topf(self, tval=0, crop=200, n_peaks=None):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_topf2(key):
            sp = self.spectra[key]
            sp.plot_topf(tval=tval, crop=crop, n_peaks=n_peaks)
    #
    def plot_sigma_mu(self, tau2_smooth=1.0, func=None, n_component=None):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_topf2(key):
            sp = self.spectra[key]
            sp.plot_sigma_mu(tau2_smooth=tau2_smooth, func=func, n_component=n_component)
    #
    def plot_spectra_agg(self, label_keys=None):
        plt.figure(figsize=(12,4))
        for key, sp in self.spectra.items():
            lkey = sp.key
            if label_keys is not None:
                lkey = "_".join(sp.attrs[lkey] for lkey in label_keys)
            plt.plot(sp.x, sp.y.mean(axis=0), lw=1.0, label=lkey)
        plt.tight_layout()
        plt.minorticks_on()
        plt.grid(ls=':', lw=0.75)
        plt.legend(loc="upper right")
        plt.show()
    #
    def spectra_agg(self, aggfunc, key):
        x = None
        ys = []
        for sp in self.spectra.values():
            y = aggfunc(sp.y)
            if x is None:
                x = sp.x
            # elif np.allclose(sp.x, x):
            #     raise TypeError("x != sp.x")
            ys.append(y)
        ys = np.array(ys)
        return Spectra(x, ys, attrs={"key":key})
    #
    def topf(self, tval=0, crop=0, sub_bg=False, sub_valleys=True):
        col = SpectraCollection()
        for key, sp in self.spectra.items():
            col[key] = sp.topf(tval=tval, crop=crop, sub_bg=sub_bg, sub_valleys=sub_valleys)
        return col
    #
    def crop(self, start_x, end_x=None):
        for sp in self.spectra.values():
            sp.crop(start_x, end_x)
    #
    def smooth(self, method="irsa2", tau2=1.0, **kwargs):
        for sp in self.spectra.values():
            sp.smooth(method=method, tau2=tau2, **kwargs)
    #
    def scale_by_robust_mean(self, tau=3.0, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_robust_mean(tau=tau, scale=scale)
    #
    def scale_by_mean(self, power=1.0, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_mean(power=1.0, scale=scale)
    #
    def scale_by_total_mean(self, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_total_mean(scale=scale)
    #
    def scale_by_total_median(self, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_total_median(scale=scale)
    #
    def scale_by_max(self, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_max(scale=scale)
    #
    def scale_by_zscore(self, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_zscore(scale=scale)
    #
    def scale_by_modified_zscore(self, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_modified_zscore(scale=scale)
    #
    def scale_by_zscore_powered(self, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_zscore_powered(scale=scale)
    #
    def scale_by_modified_zscore_powered(self, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_modified_zscore_powered(scale=scale)
    #
    def scale_by_robust_zscore(self, tau=3.5, scale=1.0):
        for sp in self.spectra.values():
            sp.scale_by_robust_zscore(tau=tau, scale=scale)
    #
    def filter(self, kind="m-zscore", **kwds):
        for key in self.spectra.keys():
            sp = self.spectra[key]
            sp.filter(kind, **kwds)
    #
    def flatten(self):
        for key in self.spectra.keys():
            sp = self.spectra[key]
            self.spectra[key] = sp.flatten()
    #
    def multiplicative_scatter_correction(self):
        for key in self.spectra.keys():
            sp = self.spectra[key]
            sp.multiplicative_scatter_correction()
    #
    def take(self, indices):
        for key in self.spectra.keys():
            sp = self.spectra[key]
            sp.take(indices)
    #
    def robust_averaging(self, tau=3.5):
        for key in self.spectra.keys():
            sp = self.spectra[key]
            self.spectra[key] = sp.robust_averaging(tau=tau)
    #
    def plot_spectra(self, smoothing=False, kind="wh", #display_data=False,
                     marker='s', ms=0, lw=0.25, plot_mean=True, **kwds):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_current_spectra(key):
            sp = self.spectra[key]
            sp.plot_spectra(smoothing=smoothing, kind=kind, #display_data=display_data,
                            marker=marker, ms=ms, lw=lw, plot_mean=plot_mean, **kwds)
    #
    def plot_mu_sigma(self, alpha=2.0):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_mu_sigma(key):
            sp = self.spectra[key]
            sp.plot_mu_sigma(alpha=alpha)
    #
    def plot_window_mean_std(self):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_window_mean_std(key):
            sp = self.spectra[key]
            sp.plot_window_mean_std()
    #
    def plot_pca_spectra(self, n_component=None):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_current_pca_spectra(key):
            sp = self.spectra[key]
            sp.plot_pca_spectra(n_component=n_component)
    #
    def plot_pca_spectra_all(self, n_component=2, kind="AM", alpha=0.95):
        attr_names = [
            'вид_бактерий', 'штамм_бактерий',
            # 'резистентность',
            'номер_цикла', 'номер_эксперимента_в_цикле',
            'номер_повтора', 'капля',
        ]
        keys = list(sorted(self.spectra.keys()))
        plt.figure(figsize=(12,5))
        LL = []
        plt.subplot(1,2,1)
        for key in keys:
            sp = self.spectra[key]
            U, Ls = sp.pca_transform_spectra(n_component=n_component, kind=kind, alpha=alpha)
            LL.append(Ls)
            # print(U.shape)
            label = "-".join([sp.attrs.get(name, "_") for name in attr_names])
            plt.scatter(U[:,0], U[:,1], s=25, label=label)
        plt.minorticks_on()
        plt.grid(ls=':', lw=0.5)
        plt.legend()
        plt.subplot(1,2,2)
        for i, key in enumerate(keys):
            sp = self.spectra[key]
            label = "-".join([sp.attrs.get(name, "_") for name in attr_names])
            plt.plot(LL[i], marker='o', ms=4, label=label)
        plt.minorticks_on()
        plt.grid(ls=':', lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    #
    def select_for_exclusion(self, sigma_mu=0.10, alpha=3.5, clear=False):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_current_spectra(key):
            sp = self.spectra[key]
            sp.select_for_exclusion(sigma_mu=sigma_mu, alpha=alpha, clear=clear)
    #
    def exclude_selected(self):
        for sp in self.spectra.values():
            sp.exclude_selected()
    #
    def exclude_by_default(self, sigma_mu=0.10, alpha=3.5, clear=False):
        for sp in self.spectra.values():
            sp.exclude_by_default(sigma_mu=sigma_mu, alpha=alpha, clear=clear)
    #
    def plot_hist(self, bins=20):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_hist(key):
            sp = self.spectra[key]
            sp.plot_hist(bins=bins)
    #
    def plot_msc(self):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_msc(key):
            sp = self.spectra[key]
            sp.plot_msc()
    #
    def plot_laser_tail(self, offset):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_laser_tail(key):
            sp = self.spectra[key]
            sp.plot_laser_tail(offset=offset)
    #
    def plot_smoothed(self, tau2=1.0, func=funcs.Square(), func2=None, func2_e=None, xrange=None, d=2):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _plot_smoothed(key):
            sp = self.spectra[key]
            sp.plot_smoothed(tau2=tau2, func=func, func2=func2, func2_e=func2_e, xrange=xrange, d=d)
    #
    def select_baselines(self, tau2=1000.0, tau1=0, tau2_smooth=0, xrange=None,
                         func=funcs.Expit(-10.0), func2=None, func2_e=None,
                         use_sigma=False, d=2):
        @ipywidgets.interact(key=self.select_key(), continuous_update=False)
        def _select_baselines(key):
            sp = self.spectra[key]
            sp.select_baselines(tau2=tau2, tau1=tau1, tau2_smooth=tau2_smooth, func=func, func2=func2, func2_e=func2_e, 
                                xrange=xrange, use_sigma=use_sigma, d=d)
    #
    def subtract_baselines(self, func=None, func2=None, func2_e=None, d=2, tau2=1.0e3, tau1=0, verbose=True, n_jobs=1):
        for sp in self.spectra.values():
            sp.subtract_baselines(func=func, func2=func2, func2_e=func2_e, d=d, tau2=tau2, tau1=tau1, verbose=verbose, n_jobs=n_jobs)
    #
    def smooth_spectra(self, func=None, func2=None, func2_e=None, d=2, tau2=1.0e3, tau1=0, verbose=True, n_jobs=1):
        for sp in self.spectra.values():
            sp.smooth_spectra(func=func, func2=func2, func2_e=func2_e, d=d, tau2=tau2, tau1=tau1, verbose=verbose, n_jobs=n_jobs)
    #
    def subtract_selected_baselines(self):
        for sp in self.spectra.values():
            print(sp.key)
            sp.subtract_selected_baselines()
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
                # key = fname[:-4]
                # attrs["key"] = key
                sp = Spectra(x, y, attrs)
                # sp.key = key
                key = sp.attrs["key"]
                self[key] = sp
