# credit: https://github.com/wfbradley/CDF-confidence
# Compute confidence interval for a quantile.
#
# Suppose I'm interested in estimating the 37th percentile.  The
# empirical CDF gives me one estimate for that.  I'd like
# to get a confidence interval: I'm 90% confident that the 37th percentile
# lies between X and Y.
#
# You can compute that with two calls to the following function
# (supposing you're interested in [5%-95%] range) by something like the
# following:
# n = len(sorted_data)
# X_index = CDF_error(n,0.37,0.05)
# Y_index = CDF_error(n,0.37,0.95)
# X=sorted_data[X_index]
# Y=sorted_data[Y_index]
# 90% confidence interval is [X,Y]

# imports
from scipy.stats import binom, beta
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np


def cdf_error_beta(n, target_quantile, quantile_quantile):
    """
    The beta distribution is the correct (pointwise) distribution
    across *quantiles* for a given *data point*; if you're not
    sure, this is probably the estimator you want to use.
    :param n:
    :param target_quantile:
    :param quantile_quantile:
    :return:
    """
    k = target_quantile * n
    return beta.ppf(quantile_quantile, k, n + 1 - k)


def cdf_error_analytic_bootstrap(n, target_quantile, quantile_quantile):
    """
    Bootstrapping can give you a distribution across *values* for a given
    *quantile*.  Warning: Although it is asymptotically correct for quantiles
    in (0,1), bootstrapping fails for the extreme values (i.e. for quantile=0
    or 1.  Moreover, you should be suspicious of confidence intervals for
    points within 1/sqrt(data size).
    :param n:
    :param target_quantile:
    :param quantile_quantile:
    :return:
    """
    target_count = int(target_quantile * float(n))

    # Start off with a binary search
    small_ind = 0
    big_ind = n - 1
    small_prob = 1 - binom.cdf(target_count, n, 0)
    big_prob = 1 - binom.cdf(target_count, n, float(big_ind) / float(n))

    while big_ind - small_ind > 4:
        mid_ind = (big_ind + small_ind) / 2
        mid_prob = 1 - binom.cdf(target_count, n, float(mid_ind) / float(n))
        if mid_prob > quantile_quantile:
            big_prob = mid_prob
            big_ind = mid_ind
        else:
            small_prob = mid_prob
            small_ind = mid_ind

        # Finish it off with a linear search
    prob_closest = -100
    for p_num in range(small_ind, big_ind + 1):
        p = float(p_num) / float(n)
        coCDF_prob = 1 - binom.cdf(target_count, n, p)
        prob_index = None
        if abs(coCDF_prob - quantile_quantile) < abs(prob_closest - quantile_quantile):
            prob_closest = coCDF_prob
            prob_index = p_num

    return prob_index


def cdf_error_dkw_band(n, target_quantile, quantile_quantile):
    """
    Compute Dvoretzky-Kiefer-Wolfowitz confidence bands.
    :param n:
    :param target_quantile:
    :param quantile_quantile:
    :return:
    """
    # alpha is the total confidence interval size, e.g. 90%.
    alpha = 1.0 - 2.0 * np.abs(0.5 - quantile_quantile)
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2.0 * float(n)))
    if quantile_quantile < 0.5:
        return max((0, target_quantile - epsilon))
    else:
        return min((1, target_quantile + epsilon))


def plot_cdf_confidence(data, num_quantile_regions='all', confidence=0.90, plot_ecdf=True,
                        data_already_sorted=False, color='green', label='', alpha=0.3, estimator_name='beta',
                        ax='use default axes'):
    """
    Plot empirical CDF with confidence intervals.
    num_quantile_regions=100 means estimate confidence interval at 1%,2%,3%,...,99%.
    confidence=0.90 mean plot the confidence interval range [5%-95%]
    :param data:
    :param num_quantile_regions:
    :param confidence:
    :param plot_ecdf:
    :param data_already_sorted:
    :param color:
    :param label:
    :param alpha:
    :param estimator_name:
    :param ax:
    :return:
    """
    data = np.array(data)
    if len(np.shape(data)) != 1:
        raise NameError('Data must be 1 dimensional!')
    if isinstance(num_quantile_regions, int) and num_quantile_regions > len(data) + 1:
        num_quantile_regions = len(data) + 1
    if len(data) < 2:
        raise NameError('Need at least 2 data points!')
    if num_quantile_regions == 'all':
        num_quantile_regions = len(data)
    if num_quantile_regions < 2:
        raise NameError('Need num_quantile_regions > 1 !')
    if not data_already_sorted:
        data = np.sort(data)
    if ax == 'use default axes':
        ax = plt.gca()
    if confidence <= 0.0 or confidence >= 1.0:
        raise NameError('"confidence" must be between 0.0 and 1.0')
    low_conf = (1.0 - confidence) / 2.0
    high_conf = 1.0 - low_conf

    quantile_list = np.linspace(1.0 / float(num_quantile_regions), 1.0 - (1.0 / float(num_quantile_regions)),
                                num=num_quantile_regions - 1)
    low = np.zeros(np.shape(quantile_list))
    high = np.zeros(np.shape(quantile_list))
    emp_quantile_list = np.linspace(1.0 / float(len(data) + 1), 1.0 - (1.0 / float(len(data) + 1)), num=len(data))

    # Some estimators give confidence intervals on the *quantiles*,
    # others give intervals on the *data*; which do we have?
    if estimator_name == 'analytic bootstrap':
        estimator_type = 'data'
        cdf_error_function = cdf_error_analytic_bootstrap
    elif estimator_name == 'DKW':
        estimator_type = 'quantile'
        cdf_error_function = cdf_error_dkw_band
    elif estimator_name == 'beta':
        estimator_type = 'quantile'
        cdf_error_function = cdf_error_beta
    else:
        raise NameError('Unknown error estimator name %s' % estimator_name)

    if estimator_type == 'quantile':
        if num_quantile_regions == len(data) + 1:
            interpolated_quantile_list = data
        else:
            invCDF_interp = interpolate.interp1d(emp_quantile_list, data)
            interpolated_quantile_list = invCDF_interp(quantile_list)

    if estimator_type == 'quantile':
        for i, q in enumerate(quantile_list):
            low[i] = cdf_error_function(len(data), q, low_conf)
            high[i] = cdf_error_function(len(data), q, high_conf)
        ax.fill_between(interpolated_quantile_list, low, high, alpha=alpha, color=color)
    elif estimator_type == 'data':
        for i, q in enumerate(quantile_list):
            low[i] = data[cdf_error_function(len(data), q, low_conf)]
            high[i] = data[cdf_error_function(len(data), q, high_conf)]
        ax.fill_betweenx(quantile_list, low, high, alpha=alpha, color=color)
    else:
        raise NameError('Unknown error estimator type %s' % (estimator_type))

    if plot_ecdf:
        ax.plot(data, emp_quantile_list, label=label, color=color)
