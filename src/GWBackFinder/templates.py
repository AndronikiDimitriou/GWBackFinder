"""Template signal shapes (port of ``Data_different signals/different_signals.jl``).

In Julia each shape had its own ``zPowerLaw``/``zPeak``/... struct and
``different_signals`` dispatched on the struct type. Here a shape is just a
name in :data:`TEMPLATES`; see :meth:`GWBackFinder.simulator.Simulator.different_signals`.
"""
import numpy as np


def Omega_powerlaw(A_1, f, gamma1):
    """``10**A_1 * (f / 1 mHz)**gamma1``."""
    return 10.0**A_1 * (f / 0.001) ** gamma1


def Omega_peak(A_1, f, ftt, Delta1):
    """Log-normal bump of height ``10**A_1`` at ``ftt`` and width ``Delta1``."""
    return 10.0**A_1 * np.exp(-np.log10(f / ftt) ** 2 / Delta1**2)


def Omega_wiggly(A_1, f, Delta, fw):
    return 10.0**A_1 * 10.0 ** np.sin(Delta * np.log10(f / fw))


def Omega_double_peak(A_1, A_2, f, f1, f2, Delta1, Delta2):
    return (10.0**A_1 * np.exp(-np.log10(f / f1) ** 2 / Delta1**2)
            + 10.0**A_2 * np.exp(-np.log10(f / f2) ** 2 / Delta2**2))


def Omega_broken_powerlaw(A_002, f, gamma1, gamma2, ftt):
    """``10**A * (f/ftt)**gamma1`` below ``ftt`` and ``(f/ftt)**gamma2`` above."""
    r = f / ftt
    return 10.0**A_002 * np.where(f < ftt, r**gamma1, r**gamma2)


def Omega_three_peaks(A_1, A_2, A_3, f, f1, f2, f3, Delta1, Delta2, Delta3):
    return (10.0**A_1 * np.exp(-np.log10(f / f1) ** 2 / Delta1**2)
            + 10.0**A_2 * np.exp(-np.log10(f / f2) ** 2 / Delta2**2)
            + 10.0**A_3 * np.exp(-np.log10(f / f3) ** 2 / Delta3**2))


# name -> (function(f, params), parameter names). Parameter order is the same
# as the ``val`` vector of the corresponding Julia struct.
TEMPLATES = {
    "powerlaw": (lambda f, p: Omega_powerlaw(p[0], f, p[1]),
                 ("log10_A", "gamma")),
    "peak": (lambda f, p: Omega_peak(p[0], f, p[1], p[2]),
             ("log10_A", "f_peak", "Delta")),
    "wiggly": (lambda f, p: Omega_wiggly(p[0], f, p[1], p[2]),
               ("log10_A", "Delta", "f_w")),
    "bpowerlaw": (lambda f, p: Omega_broken_powerlaw(p[0], f, p[1], p[2], p[3]),
                  ("log10_A", "gamma1", "gamma2", "f_break")),
    "dpeak": (lambda f, p: Omega_double_peak(p[0], p[1], f, p[2], p[3], p[4], p[5]),
              ("log10_A1", "log10_A2", "f1", "f2", "Delta1", "Delta2")),
    "three_peaks": (lambda f, p: Omega_three_peaks(p[0], p[1], p[2], f, p[3], p[4], p[5], p[6], p[7], p[8]),
                    ("log10_A1", "log10_A2", "log10_A3", "f1", "f2", "f3", "Delta1", "Delta2", "Delta3")),
}

# Julia-style aliases (zPowerLaw -> "powerlaw", ...)
ALIASES = {
    "zpowerlaw": "powerlaw", "power_law": "powerlaw",
    "zpeak": "peak",
    "zwiggly": "wiggly",
    "zbroken_powerlaw": "bpowerlaw", "broken_powerlaw": "bpowerlaw",
    "zdouble_peaks": "dpeak", "double_peak": "dpeak", "double_peaks": "dpeak",
    "zthree_peaks": "three_peaks",
}


def template_omega(name, f, params):
    """Evaluate template ``name`` with parameter vector ``params`` on ``f``."""
    key = ALIASES.get(name.lower(), name.lower())
    if key not in TEMPLATES:
        raise ValueError(f"unknown signal {name!r}; choose from {sorted(TEMPLATES)}")
    fn, names = TEMPLATES[key]
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (len(names),):
        raise ValueError(f"{key} expects {len(names)} parameters {names}, got {params.shape}")
    return fn(np.asarray(f, dtype=np.float64), params)
