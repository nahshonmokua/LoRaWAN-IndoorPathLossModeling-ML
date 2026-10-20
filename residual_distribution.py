"""The distribution of path-loss residuals that are reported in whole dB.

RSSI is an integer, so a residual is known to +-0.5 dB and the residuals sit on a 1 dB comb. dequantise() spreads each
residual uniformly over its rounding interval (one fixed draw), and every distributional analysis runs on that series.

Gaussian mixtures are fitted by EM on 0.01 dB bins (grouped-data EM) from several starts: equally spaced quantiles,
random means, and every split of the best fit with one component fewer, so the log-likelihood cannot fall as the number
of components K grows; no component is narrower than the spread of the dequantisation. select_mixture() chooses K by
cross-validation over the time blocks of the out-of-fold residuals (fit on all blocks but one, score the left-out block):
the smallest K whose held-out log-likelihood is within one standard error of the best, the standard error from the
per-block differences. BIC is reported for comparison; with millions of autocorrelated residuals it keeps falling as K grows.

block_rows() chooses the block length of the moving-block bootstrap of a quantile. The quantile's uncertainty depends on
the exceedance indicator 1{r > Q_{1-p}}, whose variance over calendar blocks, times the block length, grows with the
length until the blocks are longer than the dependence (batch means) and then levels off at the long-run variance. The
block is the shortest length of a grid from one hour to four weeks at which that variance, summed over the links, is at
least 80 % of its largest value, for every target p. The Politis-White automatic rule is not usable on these residuals:
their dependence lasts for days, beyond the rule's horizon, so it returns its own cap on every link.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import kstest, norm

BIN = 0.01                    # dB, the EM bins
SD_FLOOR = np.sqrt(1 / 12)    # dB, the spread of the dequantisation


class Mixture(NamedTuple):
    w: np.ndarray     # weights, components in order of their means
    mu: np.ndarray    # means (dB)
    sd: np.ndarray    # standard deviations (dB)

    @property
    def k(self):
        return len(self.w)

    def logpdf(self, x):
        return logsumexp(np.log(self.w) + norm.logpdf(np.asarray(x, float)[:, None], self.mu, self.sd), axis=1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        return (self.w * norm.cdf(np.asarray(x, float)[:, None], self.mu, self.sd)).sum(axis=1)

    def quantile(self, q):
        return brentq(lambda v: float(np.sum(self.w * norm.cdf((v - self.mu) / self.sd))) - q,
                      self.mu.min() - 20 * self.sd.max(), self.mu.max() + 20 * self.sd.max(), xtol=1e-9)


def dequantise(residuals, seed=42):
    """Each residual spread uniformly over its +-0.5 dB rounding interval; the same draw for every series of the same length."""
    r = np.asarray(residuals, float)
    return r + np.random.default_rng(seed).uniform(-0.5, 0.5, r.size)


def binned(x):
    """Centres and counts of the occupied bins."""
    idx = np.floor(np.asarray(x, float) / BIN).astype(np.int64)
    lo = idx.min()
    counts = np.bincount(idx - lo).astype(float)
    occupied = np.nonzero(counts)[0]
    return (lo + occupied + 0.5) * BIN, counts[occupied]


def em(c, n_c, w, mu, sd, tol=1e-9, max_iter=200_000):
    """EM on bin centres c with counts n_c from the start (w, mu, sd), until the log-likelihood gains less than tol per residual."""
    ll_old = -np.inf
    for _ in range(max_iter):
        lp = np.log(w) + norm.logpdf(c[:, None], mu, sd)
        lse = logsumexp(lp, axis=1)
        ll = float(n_c @ lse)
        if ll - ll_old < tol * n_c.sum():
            break
        ll_old = ll
        r = np.exp(lp - lse[:, None]) * n_c[:, None]
        nk = r.sum(axis=0)
        w, mu = nk / nk.sum(), (r * c[:, None]).sum(axis=0) / nk
        sd = np.sqrt(np.maximum((r * (c[:, None] - mu) ** 2).sum(axis=0) / nk, SD_FLOOR ** 2))
    order = np.argsort(mu)
    return Mixture(w[order], mu[order], sd[order]), ll


def starts(c, n_c, K, prev, rng, n_random=5):
    """Starts for K components: equally spaced quantiles, n_random random means, and every split of prev (K - 1 components)."""
    cdf = np.cumsum(n_c) / n_c.sum()
    sd0 = np.full(K, np.sqrt(np.average((c - np.average(c, weights=n_c)) ** 2, weights=n_c) / K))
    out = [(np.full(K, 1 / K), c[np.searchsorted(cdf, (np.arange(K) + 0.5) / K)], sd0)]
    out += [(np.full(K, 1 / K), np.sort(c[np.searchsorted(cdf, rng.random(K))]), sd0) for _ in range(n_random)]
    if prev is not None:
        for j in range(prev.k):
            out.append((np.r_[np.delete(prev.w, j), prev.w[j] / 2, prev.w[j] / 2],
                        np.r_[np.delete(prev.mu, j), prev.mu[j] - prev.sd[j] / 2, prev.mu[j] + prev.sd[j] / 2],
                        np.r_[np.delete(prev.sd, j), prev.sd[j] * np.sqrt(0.75), prev.sd[j] * np.sqrt(0.75)]))
    return out


def fit_mixture(x, K, seed=42, tol=1e-9):
    """Best fit with K components, built up from one component."""
    rng = np.random.default_rng(seed)
    c, n_c = binned(x)
    best = None
    for k in range(1, K + 1):
        best = max((em(c, n_c, *s, tol=tol) for s in starts(c, n_c, k, best, rng)), key=lambda f: f[1])[0]
    return best


def refit_mixture(x, m, tol=1e-11):
    """EM from the mixture m on other data x (a bootstrap replicate); the tolerance lets the 99 % quantile settle to about 0.001 dB."""
    return em(*binned(x), m.w, m.mu, m.sd, tol=tol)[0]


def select_mixture(x, blocks, k_max=8, seed=42, tol=1e-9, n_jobs=-1):
    """Mixtures with 1 to k_max components on x, and K chosen by cross-validation over blocks.
    Returns ({K: Mixture}, table, K): the table, indexed by K, holds the log-likelihood, BIC and KS distance on x, the held-out
    log-likelihood per residual, each K's loss to the best K with its standard error, and the selection."""
    x, blocks = np.asarray(x, float), np.asarray(blocks)
    rng = np.random.default_rng(seed)
    idx = np.floor(x / BIN).astype(np.int64)
    idx -= idx.min()
    counts = {"all": np.bincount(idx).astype(float)}
    for b in np.unique(blocks):   # every block but b
        counts[b] = counts["all"] - np.bincount(idx[blocks == b], minlength=counts["all"].size)
    occupied = np.nonzero(counts["all"])[0]
    c = (np.floor(x.min() / BIN) + occupied + 0.5) * BIN
    counts = {d: v[occupied] for d, v in counts.items()}
    best = {d: {} for d in counts}
    for K in range(1, k_max + 1):
        jobs = [(d, s) for d in counts for s in starts(c, counts[d], K, best[d].get(K - 1, (None,))[0], rng)]
        fits = Parallel(n_jobs=n_jobs)(delayed(em)(c, counts[d], *s, tol=tol) for d, s in jobs)
        for (d, _), f in zip(jobs, fits):
            if K not in best[d] or f[1] > best[d][K][1]:
                best[d][K] = f
    for d in best:
        assert np.all(np.diff([best[d][K][1] for K in range(1, k_max + 1)]) > -1e-6), "log-likelihood fell as K grew"
    mixtures = {K: best["all"][K][0] for K in range(1, k_max + 1)}
    heldout = pd.DataFrame({b: [best[b][K][0].logpdf(x[blocks == b]).mean() for K in mixtures] for b in np.unique(blocks)},
                           index=pd.Index(list(mixtures), name="K"))
    loss = heldout.loc[heldout.mean(axis=1).idxmax()] - heldout
    loglik = np.array([m.logpdf(x).sum() for m in mixtures.values()])
    table = pd.DataFrame({
        "log-likelihood": loglik,
        "BIC": (3 * heldout.index.to_numpy() - 1) * np.log(x.size) - 2 * loglik,
        "KS": [kstest(x, m.cdf).statistic for m in mixtures.values()],
        "held-out log-likelihood": heldout.mean(axis=1),
        "loss to best": loss.mean(axis=1),
        "loss SE": loss.std(axis=1, ddof=1) / np.sqrt(heldout.shape[1]),
    }, index=heldout.index)
    k_sel = int(table.index[table["loss to best"] <= table["loss SE"]].min())
    table["selected"] = table.index == k_sel
    return mixtures, table, k_sel


BLOCK_GRID = {"1 h": pd.Timedelta(hours=1), "6 h": pd.Timedelta(hours=6), "1 day": pd.Timedelta(days=1), "3 days": pd.Timedelta(days=3),
              "1 week": pd.Timedelta(weeks=1), "2 weeks": pd.Timedelta(weeks=2), "4 weeks": pd.Timedelta(weeks=4)}


def batch_means(eps, times, links, ps):
    """Variance of the exceedance rate 1{eps > Q_{1-p}} over calendar blocks of each length of BLOCK_GRID, times the block
    length in rows, summed over the links; one column per p. Blocks shorter than half the median block (the ends) are left out."""
    eps, links = np.asarray(eps, float), np.asarray(links)
    times = pd.Series(pd.to_datetime(np.asarray(times)))
    out = pd.DataFrame(0.0, index=pd.Index(list(BLOCK_GRID), name="block"), columns=[f"{100 * (1 - p):g} %" for p in ps])
    for p, col in zip(ps, out.columns):
        ind = (eps > np.quantile(eps, 1 - p)).astype(float)
        for g in np.unique(links):
            sel = np.nonzero(links == g)[0]
            t = times.iloc[sel]
            for name, L in BLOCK_GRID.items():
                grp = pd.Series(ind[sel]).groupby(((t - t.min()) // L).to_numpy())
                m, n = grp.mean(), grp.size()
                full = n >= 0.5 * n.median()
                out.loc[name, col] += m[full].var(ddof=1) * n[full].mean()
    return out


def block_rows(eps, times, links, ps, share=0.8):
    """Rows per block of one link for the moving-block bootstrap: the shortest length of BLOCK_GRID at which the batch-means
    variance is at least `share` of its largest value for every p. Returns (rows, length name, the batch-means table)."""
    table = batch_means(eps, times, links, ps)
    name = table.index[(table >= share * table.max()).all(axis=1).to_numpy()][0]
    times = pd.to_datetime(np.asarray(times))
    rows_per_link = len(eps) / len(np.unique(np.asarray(links)))
    return int(round(rows_per_link * (BLOCK_GRID[name] / (times.max() - times.min())))), name, table
