from scipy.stats import t
import numpy as np
from irsa.preprocess.utils import modified_zscore

def grubbs_test(y, alpha=0.05, ddof=0):
    import numpy as np

    size = len(y)
    G = max(abs(y - np.mean(y))) / np.std(y, ddof=ddof)
    
    t_dist = t.ppf(1 - alpha / (2 * size), size - 2)
    t_dist2 = t_dist * t_dist
    c_val = ((size - 1) / np.sqrt(size)) * np.sqrt(t_dist2 / (size - 2 + t_dist2))
    
    return G > c_val
     

def generalized_esd(x, n_outl, alpha=0.05, full_output=False, ddof=0):
    """
    Generalized ESD Test for Outliers.

    http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

    Parameters
    ----------
    x : ndarray
        1-d array
    n_outl : int
        Maximum number of outliers in the data set.
    alpha : float, optional
        Significance (default is 0.05).
    full_output : boolean, optional
        Determines whether additional return values
        are provided. Default is False.
    ddof : boolean, optional
        Delta Degrees of Freedom
    
    Returns
    -------
    Number of outliers : int
        The number of data points characterized as
        outliers by the test.
    Indices : list of ints
        The indices of the data points found to
        be outliers.
    R : list of floats, optional
        The values of the "R statistics". Only provided
        if `full_output` is set to True.
    L : list of floats, optional
        The lambda values needed to test whether a point
        should be regarded an outlier. Only provided
        if `full_output` is set to True.  

    """
    import numpy.ma as ma
    from numpy import sqrt as np_sqrt

    xm = ma.array(x)
    n = len(xm)

    R = []
    L = []
    minds = []
    for i in range(n_outl+1):

        xmean = xm.mean()
        xstd = xm.std(ddof=ddof)
        rr = abs((xm - xmean) / xstd)
        i_max = rr.argmax()
        minds.append(i_max)
        if i >= 1:
            R.append(rr[i_max])
            p = 1.0 - alpha / (2.0 * (n - i + 1))
            perPoint = t.ppf(p, n - i - 1)
            L.append((n - i) * perPoint /
                     np_sqrt((n - i - 1 + perPoint**2) * (n - i + 1)))
        # Mask that value and proceed
        xm[i_max] = ma.masked

    is_found = False
    for i in range(n_outl-1, -1, -1):
        if R[i] > L[i]:
            is_found = True
            break

    if is_found:
        if not full_output:
            return i + 1, minds[0:i + 1]
        else:
            return i + 1, minds[0:i + 1], R, L, minds
    else:
        if not full_output:
            return 0, []
        else:
            return 0, [], R, L, minds

def despike_whittaker(y, m=4, tau=3.0, _where=np.where):
    ddy = np.empty_like(y)
    ddy[1:-1] = 2*y[1:-1] - y[:-2] - y[2:]
    ddy[0] = ddy[-1] = 0
    
    z = modified_zscore(ddy)
    I = _where(z > tau)

    for i in I:
        y[i] = (y[i-1] + y[i+1]) / 2.

def despike_whittaker2(y, m=4, tau=3.0, _where=np.where):
    ddy = np.empty_like(y)
    ddy[1:-1] = 2*y[1:-1] - y[:-2] - y[2:]
    ddy[0] = ddy[-1] = 0
    
    z = modified_zscore(ddy)
    I = _where(z > tau)

    for i in I:
        y[i] = (y[i-1] + y[i+1]) / 2.
