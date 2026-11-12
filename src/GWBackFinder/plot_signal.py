import numpy as np


def hdi(x, prob):
    """Highest-density interval of 1-D samples (same result as ``arviz.hdi``)."""
    x = np.sort(np.asarray(x))
    n = len(x)
    k = int(np.floor(prob * n))
    if k < 1 or k >= n:
        return np.array([x[0], x[-1]])
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return np.array([x[i], x[i + k]])


def bounds(z, z2):
    """Per-parameter 68/95/99.7 % HDI limits of the slopes ``z`` (n, 26) and amplitude ``z2`` (n,).

    Same output as the original version: lists of 27 values
    ``int2low, int1low, int0low, int2high, int1high, int0high``
    (68, 95, 99.7 % lower limits, then upper limits).
    """
    cols = [z[:, i] for i in range(26)] + [np.asarray(z2)]
    out = {p: np.array([hdi(c, p) for c in cols]) for p in (0.68268, 0.95450, 0.99730)}
    return (list(out[0.68268][:, 0]), list(out[0.95450][:, 0]), list(out[0.99730][:, 0]),
            list(out[0.68268][:, 1]), list(out[0.95450][:, 1]), list(out[0.99730][:, 1]))
