import numpy as np
import matplotlib
import statsmodels.api as sm
from scipy.stats import norm, poisson, nbinom


def glm(data, baseline_period, alpha=0.05, include_trend=True):
    y = np.asarray(data, dtype=float)
    n = len(y)

    if baseline_period is None or baseline_period < 2 or baseline_period >= n:
        raise ValueError("baseline_period must be >= 2 and < len(data).")

    # Design matrix for baseline 
    t = np.arange(n, dtype=float)
    y_base = y[:baseline_period]
    t_base = t[:baseline_period]

    if include_trend:
        # mu_t = b0 + b1*t
        X_base = np.column_stack([np.ones(baseline_period), t_base])
        X_all  = np.column_stack([np.ones(n), t])
    else:
        # mu_t = b0
        X_base = np.ones((baseline_period, 1))
        X_all  = np.ones((n, 1))

    beta, *_ = np.linalg.lstsq(X_base, y_base, rcond=None)
    mu_hat = X_all @ beta

    resid_base = y_base - (X_base @ beta)
    p = X_base.shape[1]
    dof = max(baseline_period - p, 1)
    sigma = float(np.sqrt(np.sum(resid_base**2) / dof))

    z = float(norm.ppf(1 - alpha / 2.0))
    ucl = mu_hat + z * sigma
    lcl = mu_hat - z * sigma

    out_of_control_index = None
    for i in range(baseline_period, n):
        if y[i] > ucl[i] or y[i] < lcl[i]:
            out_of_control_index = i
            break

    return {
        "series": y,
        "ucl": ucl,
        "lcl": lcl,
        "expected": mu_hat,
        "sigma": sigma,
        "out_of_control_index": out_of_control_index
    }

# farrington helper functions
def _overdispersion_phi(y, mu, p):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    pearson = (y - mu) / np.sqrt(np.maximum(mu, 1e-12))
    dof = max(len(y) - p, 1)
    phi = float(np.sum(pearson ** 2) / dof)
    return max(1.0, phi)
def _seasonal_basis(t_idx, period, n_splines=10):
    t_idx = np.asarray(t_idx, dtype=float)
    phase = (t_idx % period) / float(period)

    knots = np.linspace(0, 1, n_splines, endpoint=False)

    # circular distance on [0,1)
    d = np.abs(phase[:, None] - knots[None, :])
    d = np.minimum(d, 1.0 - d)

    bw = 1.0 / n_splines  # bandwidth heuristic
    B = np.exp(-(d ** 2) / (2.0 * (bw ** 2)))
    return B


def _anscombe_residual(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu_safe = np.maximum(mu, 1e-12)
    return 1.5 * (np.power(np.maximum(y, 0.0), 2/3) - np.power(mu_safe, 2/3)) / np.power(mu_safe, 1/6)

def _huber_weights(r, c=2.0):
    r = np.asarray(r, dtype=float)
    ar = np.abs(r)
    w = np.ones_like(r)
    mask = ar > c
    w[mask] = c / np.maximum(ar[mask], 1e-12)
    return w
def poisson_upper(mu, phi, alpha):
      z = norm.ppf(1-alpha)
      return mu + z * np.sqrt(phi * mu) 

def _ucl_count(mu_t, phi, alpha):
    mu_t = float(max(mu_t, 0.0))
    if mu_t <= 1e-12:
        return 0.0

    if phi <= 1.0000001:
        return float(poisson.ppf(1 - alpha, mu_t))

    size = mu_t / max(phi - 1.0, 1e-12)
    p = size / (size + mu_t)  
    return float(nbinom.ppf(1 - alpha, size, p))

def farrington(
    data,
    baseline_period,
    alpha=0.01,
    period=52,        # 52 for weekly, 365 for daily
    b=3,              # number of past "years"/cycles
    w=3,              # +/- window around seasonal position
    n_splines=10,     # seasonal basis size
    reweight_iters=3, # robust reweighting iterations
    c=2.0):
    data = np.array(data, dtype=float)
    n = len(data)

    if baseline_period < 10 or baseline_period >= n:
        raise ValueError("baseline_period must be >= 10 and < len(data)")
    if np.any(data < 0):
        raise ValueError("Farrington expects count data (>= 0).")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    ucl = np.full(n, np.nan, dtype=float)
    lcl = np.zeros(n, dtype=float)
    expected = np.full(n, np.nan, dtype=float)  
    out_of_control_index = None

    def reference_indices(t):
        idx = []
        for k in range(1, b + 1):
            center = t - k * period
            if center < 0:
                continue
            for d in range(-w, w + 1):
                j = center + d
                if 0 <= j < n:
                    idx.append(j)
        return sorted(set(idx))

    for t in range(baseline_period, n):
        idx = reference_indices(t)
        p = 2 + n_splines
        if len(idx) < (p + 5):
            continue

        y_ref = data[idx]
        t_ref = np.array(idx, dtype=float)

        # Build design matrix: intercept + trend + seasonal basis
        trend = (t_ref - np.mean(t_ref)) / float(period)
        B = _seasonal_basis(t_ref, period=period, n_splines=n_splines)
        X = np.column_stack([np.ones(len(t_ref)), trend, B])

        weights = np.ones(len(y_ref), dtype=float)

        fit = None
        mu_ref = None
        phi = 1.0

        for _ in range(reweight_iters):
            model = sm.GLM(y_ref, X, family=sm.families.Poisson(), freq_weights=weights)
            fit = model.fit()

            mu_ref = fit.predict(X)
            # Robust weights from Anscombe residuals
            r = _anscombe_residual(y_ref, mu_ref)
            weights = _huber_weights(r, c=c)

        # Overdispersion (quasi-Poisson style)
        phi = _overdispersion_phi(y_ref, mu_ref, p=X.shape[1])

        # Predict at time t
        t_t = float(t)
        trend_t = (t_t - np.mean(t_ref)) / float(period)
        B_t = _seasonal_basis(np.array([t_t]), period=period, n_splines=n_splines)
        X_t = np.column_stack([np.ones(1), np.array([trend_t]), B_t])

        mu_t = float(fit.predict(X_t)[0])
        expected[t] = mu_t
        ucl[t] = _ucl_count(mu_t, phi, alpha)

        if out_of_control_index is None and data[t] > ucl[t]:
            out_of_control_index = t
    return {
        "series": np.array(data),
        "ucl": ucl,
        "lcl": lcl,
        "out_of_control_index": out_of_control_index,
        "expected": expected
    }


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
   
def cusum(data, baseline_mean, sigma, sigma_multiplier, baseline_period=0):
    n = len(data)
    k = 0.5 * sigma  # reference value
    h = sigma * sigma_multiplier  # decision interval
    c_plus = np.zeros(n)
    c_minus = np.zeros(n)
    out_of_control_index = None

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



