import numpy as np

def zscore(y):  
    s = y - y.mean()
    z = s / s.std()
    return z

def modified_zscore(y, _median=np.median):  
    med = _median(y)
    s = y - med
    mad = _median(abs(s))
    if mad == 0:
        z = 0.67449 * s / abs(s).mean()
    else:
        z = 0.67449 * s / mad
    return z

median_median_zscore = modified_zscore

def median_mean_zscore(y, _median=np.median):  
    med = _median(y)
    s = y - med
    z = 0.67449 * s / (abs(s)).mean()
    return z

def modified_zscore2(yy, _median=np.median):
    dyy = yy - _median(yy, axis=0)
    abs_dyy = abs(dyy)
    dd = _median(abs_dyy, axis=0)
    for i, d in enumerate(dd):
        if d == 0:
            dd[i] = abs_dyy[:,i].mean()
    zz = 0.67449 * dyy / dd
    return zz

def robust_mean(y, tau=3.5, zscore=modified_zscore):
    z = zscore(y)
    return y[abs(z) <= tau].mean()

def robust_mean2(yy, tau=3.5, _mean=np.median, _std=np.median, _fromiter=np.fromiter, _C=1/0.67449):
    mu = _mean(yy, axis=0)
    sigma = _C * _std(abs(yy - mu), axis=0)
    th = tau * sigma
    mask = (yy <= (mu + th)) & (yy >= (mu - th))

    N = yy.shape[1]
    Y = _fromiter(
            (yy[mask[:,i],i].mean() for i in range(N)), 
            'd', N)
    return Y

def filter_outliers(ys, tau=3.5, zscore=modified_zscore):
    zz = zscore(ys)
    return ys[abs(zz) <= tau]

def outliers_indexes(y, tau=3.5, zscore=modified_zscore, _nonzero=np.nonzero):
    z = zscore(y)
    return _nonzero(abs(z) > tau)

def outliers_indexes2(yy, tau=3.5, zscore=modified_zscore2, _nonzero=np.nonzero):
    zz = zscore(yy)
    return _nonzero(abs(zz) > tau)

def mark_outliers2(yy, tau=3.5, mean=np.median, std=np.median, marker=np.nan, _nonzero=np.nonzero):
    mu = mean(yy, axis=0)
    sigma = std(abs(yy - mu)) / 0.67449
    mask = (yy > (mu + tau*sigma)) | (yy < (mu - tau*sigma))
    
    indexes = _nonzero(mask)
    
    for i,j in zip(*indexes):
        yy[i,j] = marker

def replace_outliers2(yy, tau=3.5, mean=np.median, std=np.median, _nonzero=np.nonzero):
    mu = mean(yy, axis=0)
    sigma = std(abs(yy - mu), axis=0) / 0.67449
    mask = (yy > (mu + tau*sigma)) | (yy < (mu - tau*sigma))
    
    indexes = _nonzero(mask)
    
    for i,j in zip(*indexes):
        yy[i,j] = mu[j]

