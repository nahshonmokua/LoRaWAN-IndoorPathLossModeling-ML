"""Calibration and validation tools for fade margins (notebooks 17 to 20).

A margin is an upper quantile forecast of the path-loss deviation y from the link's anchor. A base predictor b gives a threshold
for every packet, a calibration adds a shift c from earlier scores s = y - b, and a packet is covered when y <= b + c.
Everything here is causal: a shift only sees scores of earlier packets.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def pinball(y, m, tau):
    """Pinball loss of the quantile forecast m for the outcome y at level tau, per packet."""
    return np.where(y >= m, tau * (y - m), (1 - tau) * (m - y))


def conformal_level(n, tau):
    """Split-conformal level with the finite-sample correction: ceil((n + 1) tau) / n, capped at 1."""
    return min(1.0, np.ceil((n + 1) * tau) / n)


def shift_pooled(scores, tau):
    """One shift for all packets: the corrected tau-quantile of the calibration scores."""
    return float(np.quantile(scores, conformal_level(len(scores), tau), method="higher"))


def shift_groups(scores, g_cal, g_new, tau, min_n=500):
    """One shift per group (Mondrian calibration). A group with fewer than min_n calibration packets, or one never seen, uses the pooled shift."""
    pooled = shift_pooled(scores, tau)
    s = pd.Series(scores)
    table = {g: (shift_pooled(x.to_numpy(), tau) if len(x) >= min_n else pooled) for g, x in s.groupby(np.asarray(g_cal))}
    return np.array([table.get(g, pooled) for g in np.asarray(g_new)])


def shift_weighted(scores, age_days, tau, half_life):
    """Recency-weighted shift (Barber et al. 2023): weights halve every half_life days of age, the new packet carries weight 1 at +infinity."""
    w = 0.5 ** (np.asarray(age_days) / half_life)
    order = np.argsort(scores)
    cum = np.cumsum(w[order]) / (w.sum() + 1)
    i = np.searchsorted(cum, tau)
    return float(scores[order][min(i, len(scores) - 1)])


def aci(base, y, cal_scores, tau, gamma):
    """Adaptive conformal inference (Gibbs and Candes 2021) on one link in time order: the miscoverage level moves after every packet,
    alpha <- alpha + gamma (1 - tau - miss), and the shift is the matching quantile of the fixed calibration scores. Returns the margins."""
    s = np.sort(cal_scores)
    n = len(s)
    alpha = 1 - tau
    m = np.empty(len(y))
    for i in range(len(y)):
        level = 1 - alpha
        c = s[-1] if level >= 1 else s[max(0, min(n - 1, int(np.ceil((n + 1) * level)) - 1))]
        m[i] = base[i] + c
        alpha += gamma * ((1 - tau) - (y[i] > m[i]))
    return m


def spread_groups(s20, cut):
    """Recent-spread terciles from cut points taken on calibration packets: 0 calm, 1 middle, 2 volatile."""
    return np.digitize(s20, cut)


def day_codes(t):
    """Integer code of the UTC day of each timestamp, the resampling unit of every bootstrap here."""
    return pd.factorize(t.dt.floor("D"), sort=True)[0]


def boot_stat(day, cols, f, B=2000, seed=0):
    """Day-block bootstrap. cols is an n x k array of per-packet quantities, f maps a (B, k) array of resampled totals to B statistics.
    Returns the statistic on the data and its 95 % interval."""
    cols = np.asarray(cols, float).reshape(len(day), -1)
    D = day.max() + 1
    S = np.column_stack([np.bincount(day, weights=cols[:, j], minlength=D) for j in range(cols.shape[1])])
    point = float(f(S.sum(0, keepdims=True))[0])
    counts = np.random.default_rng(seed).multinomial(D, np.full(D, 1 / D), size=B)
    lo, hi = np.nanpercentile(f(counts @ S), [2.5, 97.5])
    return point, float(lo), float(hi)


def transitions(t, link, hit, max_gap=180):
    """For every packet, how its hit follows the previous packet of the same link when at most max_gap seconds earlier: the columns
    n00, n01, n10, n11 hold a 1 in the cell (previous, current) and zeros when there is no predecessor. t must be sorted within link."""
    h = pd.Series(np.asarray(hit, float))
    prev = h.groupby(np.asarray(link)).shift(1)
    dt = t.reset_index(drop=True).groupby(np.asarray(link)).diff().dt.total_seconds()
    ok = (dt <= max_gap) & prev.notna()
    return np.column_stack([ok & (prev == 0) & (h == 0), ok & (prev == 0) & (h == 1), ok & (prev == 1) & (h == 0), ok & (prev == 1) & (h == 1)]).astype(float)


def clustering_ratio(T):
    """Chance of a miss right after a miss divided by the chance of a miss after a covered packet; 1 means independent misses."""
    return (T[:, 3] / (T[:, 2] + T[:, 3])) / (T[:, 1] / (T[:, 0] + T[:, 1]))


def christoffersen_independence(n):
    """Likelihood-ratio statistic of Christoffersen (1998) for first-order dependence of the hits, from the counts n00, n01, n10, n11."""
    n00, n01, n10, n11 = n
    pi, p01, p11 = (n01 + n11) / sum(n), n01 / (n00 + n01), n11 / (n10 + n11)
    l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    l1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + (n11 * np.log(p11) if n11 else 0)
    return float(-2 * (l0 - l1))


def dynamic_binary_test(hit, covars, day):
    """Conditional-coverage test in the spirit of Engle and Manganelli (2004) and Dumitrescu et al. (2012): a logistic regression of the
    hit on the lagged hit and the state covariates. Wald test that every slope is zero, with standard errors clustered by day."""
    X = sm.add_constant(covars.astype(float))
    res = sm.GLM(np.asarray(hit, float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": day})
    R = np.zeros((X.shape[1] - 1, X.shape[1]))
    R[np.arange(X.shape[1] - 1), np.arange(1, X.shape[1])] = 1
    w = res.wald_test(R, scalar=True)
    p = res.predict(X)
    return float(w.statistic), float(w.pvalue), float(np.min(p)), float(np.max(p))
