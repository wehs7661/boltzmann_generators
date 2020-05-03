import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def density_estimator(x, x_range, weights=None, k=5, optimize=False):
    """
    Calculates the probability density given input data samples.

    Parameters
    ----------
    x : np.array
        Input data.
    x_range : np.array
        The range of the probability density to be calculated.
    weights : np.array
        List of sample weights attached to the data x.
    k : int
        The number of fold used in k-fold cross validation when optimizing kernel bandwidth.
    optimize : bool
        Whether to use K-fold cross validation to optimize the bandwidth.

    Returns
    -------
    p : np.array
        The probability density of the given range x_range.
    log_p : np.array
        The log of probability density of the given range x_range.
    """
    if optimize is True:
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=KFold(n_splits=k))
        grid.fit(x[:, None], sample_weight=weights)
        #self.bandwidth = grid.best_params_['bandwidth']
        log_p = grid.best_estimator_.score_samples(x_range[:, None])
    else:
        #self.bandwidth = 0.5
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(x[:, None], sample_weight=weights)
        log_p = kde.score_samples(x_range[:, None])
    p = np.exp(log_p)

    return p, log_p
