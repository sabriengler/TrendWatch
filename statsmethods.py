import numpy as np
from scipy.stats import norm, poisson, nbinom
﻿
﻿
def glm(data, baseline_period, alpha=0.05, include_trend=True):
    import statsmodels.api as sm
    y = np.asarray(data, dtype=float)
    n = len(y)
﻿
    if baseline_period is None or baseline_period < 2 or baseline_period >= n:
        raise ValueError("baseline_period must be >= 2 and < len(data).")
    if not (0< alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
﻿
    # Design matrix for baseline 
    t = np.arange(n, dtype=float)
    y_base = y[:baseline_period]
    t_base = t[:baseline_period]
﻿
    if include_trend:
        # mu_t = b0 + b1*t
        X_base = np.column_stack([np.ones(baseline_period), t_base])
        X_all  = np.column_stack([np.ones(n), t])
    else:
        # mu_t = b0
        X_base = np.ones((baseline_period, 1))
        X_all  = np.ones((n, 1))
    
    model = sm.GLM(y_base, X_base, family=sm.families.Gaussian())
    fit = model.fit()
﻿
    # expected values
    mu_hat = fit.predict(X_all)
﻿
    # residual scale from fitted GLM
    sigma = float(np.sqrt(fit.scale))
    
    # approx limits
    z = float(norm.ppf(1 - alpha / 2.0))
    ucl = mu_hat + z * sigma
    lcl = mu_hat - z * sigma
﻿
    out_of_control_index = None
    for i in range(baseline_period, n):
        if y[i] > ucl[i] or y[i] < lcl[i]:
            out_of_control_index = i
            break
﻿
    return {
        "series": y,
        "ucl": ucl,
        "lcl": lcl,
        "expected": mu_hat,
        "sigma": sigma,
        "out_of_control_index": out_of_control_index
    }
def _farrington_weights(r, threshold=2.58):
    r = np.asarray(r, dtype=float)
    ar = np.abs(r)
    w = np.ones_like(r)
    mask = ar > threshold
    w[mask] = threshold / np.maximum(ar[mask], 1e-12)
    return w
﻿
﻿
def _ucl_count(mu_t, phi, alpha):
    mu_t = float(max(mu_t, 0.0))
    if mu_t <= 1e-12:
        return 0.0
﻿
    if phi <= 1.0000001:
        return float(poisson.ppf(1 - alpha, mu_t))
﻿
    size = mu_t / max(phi - 1.0, 1e-12)
    p = size / (size + mu_t)
    return float(nbinom.ppf(1 - alpha, size, p))
def _overdispersion_phi(y, mu, p):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    pearson = (y - mu) / np.sqrt(np.maximum(mu, 1e-12))
    dof = max(len(y) - p, 1)
    phi = float(np.sum(pearson ** 2) / dof)
    return max(1e-6, phi)
﻿
﻿
def _anscombe_residual(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu_safe = np.maximum(mu, 1e-12)
    return 1.5 * ((np.maximum(y, 0.0) ** (2/3)) - (mu_safe ** (2/3))) / (mu_safe ** (1/6))
﻿
def farrington(
    data,
    baseline_period,
    alpha=0.01,
    period=26,             
    b=3,                   # years/cycles back
    w=4,                   # +/- seasonal window
    no_periods=1,         # seasonal groups (Farrington Flexible style)
    reweight=True,
    weights_threshold=2.58,
    include_trend=False,
    p_threshold_trend=0.05,
    past_not_included=0    # exclude most recent reference points if desired
):
    import statsmodels.api as sm
    data = np.asarray(data, dtype=float)
    n = len(data)
﻿
    if baseline_period < 10 or baseline_period >= n:
        raise ValueError("baseline_period must be >= 10 and < len(data)")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if np.any(data < 0) or np.any(data != np.floor(data)):
        raise ValueError("Farrington expects nonnegative integer count data.")
    if period <= 1:
        raise ValueError("period must be > 1")
    if b < 1 or w < 0 or no_periods < 1:
        raise ValueError("b >= 1, w >= 0, and no_periods >= 1 are required")
﻿
    ucl = np.full(n, np.nan, dtype=float)
    lcl = np.zeros(n, dtype=float)
    expected = np.full(n, np.nan, dtype=float)
    out_of_control_index = None
﻿
    # cache objects used over and over
    poisson_family = sm.families.Poisson()
    season_ids = ((np.arange(n) % period) * no_periods // period).astype(int)
    season_lookup = np.eye(no_periods, dtype=float)[:, 1:]   # first level dropped once
﻿
    def reference_indices(t):
        idx = []
        for k in range(1, b + 1):
            center = t - k * period
            if center < 0:
                continue
            lo = max(0, center - w)
            hi = min(n, center + w + 1)
            idx.extend(range(lo, hi))
﻿
        idx = sorted(set(idx))
﻿
        if past_not_included > 0:
            idx = [j for j in idx if j < (t - past_not_included)]
﻿
        return idx
﻿
    for t in range(baseline_period, n):
        idx = reference_indices(t)
        if len(idx) < 8:
            continue
﻿
        idx = np.asarray(idx, dtype=int)
        y_ref = data[idx]
        t_ref = idx.astype(float)
        t_ref_mean = t_ref.mean()
﻿
        # seasonal factor groups
        season_ref = season_lookup[season_ids[idx]]
﻿
        trend_ref = (t_ref - t_ref_mean) / float(period)
﻿
        if include_trend:
            X_ref_full = np.column_stack([np.ones(len(idx)), trend_ref, season_ref])
        else:
            X_ref_full = np.column_stack([np.ones(len(idx)), season_ref])
﻿
        weights = np.ones(len(idx), dtype=float)
﻿
        # initial fit
        fit = sm.GLM(
            y_ref,
            X_ref_full,
            family=poisson_family,
            freq_weights=weights
        ).fit()
﻿
        mu_ref = fit.fittedvalues
﻿
        # outbreak reweighting
        if reweight:
            for _ in range(2):
                r = _anscombe_residual(y_ref, mu_ref)
                weights = _farrington_weights(r, threshold=weights_threshold)
﻿
                fit = sm.GLM(
                    y_ref,
                    X_ref_full,
                    family=poisson_family,
                    freq_weights=weights
                ).fit()
﻿
                mu_ref = fit.fittedvalues
﻿
        # optionally drop trend if not significant
        X_ref = X_ref_full
        if include_trend:
            try:
                trend_p = float(fit.pvalues[1])
            except Exception:
                trend_p = 1.0
﻿
            if trend_p >= p_threshold_trend:
                X_ref = np.column_stack([np.ones(len(idx)), season_ref])
                fit = sm.GLM(
                    y_ref,
                    X_ref,
                    family=poisson_family,
                    freq_weights=weights
                ).fit()
                mu_ref = fit.fittedvalues
﻿
        # overdispersion estimate
        phi = _overdispersion_phi(y_ref, mu_ref, p=X_ref.shape[1])
﻿
        # predict current point
        season_t = season_lookup[season_ids[t]:season_ids[t] + 1]
﻿
        if X_ref.shape[1] == 1 + season_ref.shape[1]:
            X_t = np.column_stack([np.ones(1), season_t])
        else:
            trend_t = np.array([(t - t_ref_mean) / float(period)])
            X_t = np.column_stack([np.ones(1), trend_t, season_t])
﻿
        mu_t = float(fit.predict(X_t)[0])
        expected[t] = mu_t
        ucl[t] = _ucl_count(mu_t, phi, alpha)
﻿
        if out_of_control_index is None and data[t] >= ucl[t]:
            out_of_control_index = t
            break
﻿
    return {
        "series": np.array(data),
        "ucl": ucl,
        "lcl": lcl,
        "out_of_control_index": out_of_control_index,
        "expected": expected
    }
﻿
def shewhart(data, baseline_mean, sigma, sigma_multiplier, baseline_period):
    n = len(data)
    out_of_control_index = None
    for i, val in enumerate(data):
        if i >= baseline_period:
            if val > baseline_mean + sigma_multiplier * sigma or val < baseline_mean - sigma_multiplier * sigma:
                out_of_control_index = i
                break
    ucl = np.full(n, baseline_mean + sigma_multiplier * sigma)
    lcl = np.full(n, baseline_mean - sigma_multiplier * sigma)
    return {"series": np.array(data), 
            "ucl": ucl, 
            "lcl": lcl, 
            "out_of_control_index": out_of_control_index
            }
def ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, baseline_period):
    n = len(data)
    ewma_series = np.zeros(n)
    ewma_series[0] = baseline_mean
    ucl = np.zeros(n)
    lcl = np.zeros(n)
    out_of_control_index = None
    
    for i in range(1, n):
        ewma_series[i] = lambda_val * data[i] + (1 - lambda_val) * ewma_series[i-1]
    
    for i in range(n):
        sigma_ewma = sigma * np.sqrt(lambda_val / (2 - lambda_val) * (1 - (1 - lambda_val)**(2*i)))
        ucl[i] = baseline_mean + sigma_multiplier * sigma_ewma
        lcl[i] = baseline_mean - sigma_multiplier * sigma_ewma
        if i >= baseline_period:
            if ewma_series[i] > ucl[i] or ewma_series[i] < lcl[i]:
                out_of_control_index = i
                break
    
    return {"series": ewma_series,
            "ucl": ucl, 
            "lcl": lcl, 
            "out_of_control_index": out_of_control_index
            }
﻿
def mc_ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, baseline_period):
    n = len(data)
    mc_ewma_series = np.zeros(n)
    mc_ewma_series[0] = baseline_mean
    ucl = np.zeros(n)
    lcl = np.zeros(n)
    out_of_control_index = None
    
    for i in range(1, n):
        mc_ewma_series[i] = lambda_val * data[i-1] + (1 - lambda_val) * mc_ewma_series[i-1]
    
    for i in range(n):
        ucl[i] = mc_ewma_series[i] + sigma_multiplier * sigma
        lcl[i] = mc_ewma_series[i] - sigma_multiplier * sigma
        if i >= baseline_period:
            if data[i] > ucl[i] or data[i] < lcl[i]:
                out_of_control_index = i
                break
    return {"series": mc_ewma_series, 
            "ucl": ucl, 
            "lcl": lcl, 
            "out_of_control_index": out_of_control_index
            }
   
def cusum(data, baseline_mean, sigma, sigma_multiplier, baseline_period=0, k_val=None, h_val=None):
    n = len(data)
    k = k_val * sigma if k_val is not None else 0.5 * sigma  # reference value
    h = h_val * sigma if h_val is not None else sigma_multiplier * sigma  # decision interval
    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    out_of_control_index = None
﻿
    start = max(1, baseline_period)
    for i in range(start, n):
        c_plus[i] = max(0, c_plus[i-1] + data[i] - baseline_mean - k)
        c_minus[i] = min(0, c_minus[i-1] + data[i] - baseline_mean + k)
        if c_plus[i] > h or abs(c_minus[i]) > h:
            out_of_control_index = i
            break
    ucl = h
    lcl = -h
    return {"series": (c_plus,c_minus), 
            "ucl": ucl, 
            "lcl": lcl, 
            "out_of_control_index": out_of_control_index
            }

