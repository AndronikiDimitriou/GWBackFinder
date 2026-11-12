import numpy as np
import pytest

from GWBackFinder import Simulator, binning, frequency_grid, piecewise_powerlaw, f26
from GWBackFinder.binning import inverse_variance_bin
from GWBackFinder.noise import Omega_noiseh2_AA, Omega_noiseh2_TT, noise_basis
from GWBackFinder.templates import template_omega, TEMPLATES

from _reference_f26 import f26_reference


@pytest.fixture(scope="module")
def sim():
    return Simulator()


def test_frequency_grid_matches_julia_range():
    f = frequency_grid()
    assert f.size == 499971
    assert f[0] == 3e-5 and f[-1] == 0.5 and f[2970] == 0.003
    assert np.allclose(np.diff(f), 1e-6, rtol=1e-6)


def test_binning_layout():
    f = frequency_grid()
    idx, idx26, lb26, lb, ff = binning(f)
    assert lb.size == 1000 and lb26.size == 27
    assert ff.size == f.size - 2970 == idx.size
    assert idx.min() == 0 and idx.max() == 999
    counts = np.bincount(idx, minlength=1000)
    assert counts.min() > 0          # no empty bin -> no log10(0)
    assert counts[-1] == 1           # last bin only holds f = 0.5, as in Julia
    assert np.all(np.diff(idx) >= 0) and np.all(np.diff(idx26) >= 0)
    assert idx26.max() == 26
    assert binning(f, 26)[2].size == 26


@pytest.mark.parametrize("seed", range(10))
def test_piecewise_powerlaw_matches_literal_julia_f26(seed):
    rng = np.random.default_rng(seed)
    lb26 = binning(frequency_grid())[2]
    b, S0 = lb26[:25], lb26[12]
    fs = np.concatenate([10 ** rng.uniform(np.log10(3e-5), np.log10(0.5), 500), lb26])
    n = rng.uniform(-12, 12, 26)
    As = 10 ** rng.uniform(-14, -6)
    ref = np.array([f26_reference(s, S0, *n, As, *b) for s in fs])
    np.testing.assert_allclose(piecewise_powerlaw(fs, n, As, b), ref, rtol=1e-12)
    np.testing.assert_allclose(f26(fs, S0, *n, As, *b), ref, rtol=1e-12)


def test_piecewise_powerlaw_batched():
    rng = np.random.default_rng(1)
    b = binning(frequency_grid())[2][:25]
    f = np.logspace(-4.5, -0.3, 50)
    n = rng.uniform(-3, 3, (4, 26))
    As = 10 ** rng.uniform(-12, -8, 4)
    out = piecewise_powerlaw(f, n, As, b)
    assert out.shape == (4, 50)
    for i in range(4):
        np.testing.assert_allclose(out[i], piecewise_powerlaw(f, n[i], As[i], b))


def test_noise_basis_is_exact_decomposition():
    f = np.logspace(-4.5, -0.3, 200)
    for ch, fn in (("AA", Omega_noiseh2_AA), ("TT", Omega_noiseh2_TT)):
        u, v = noise_basis(f, ch)
        np.testing.assert_allclose(fn(f, 2.5e9, 3e8, 13.0, 2.5), 2.5**2 * u + 13.0**2 * v, rtol=1e-12)


def test_inverse_variance_bin():
    idx = np.array([0, 0, 1, 1, 1])
    vals = np.array([1.0, 3.0, 2.0, 2.0, 5.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(inverse_variance_bin(vals, w, idx, 2), [2.0, 14 / 4])


def test_simulate_AA_shapes_and_P(sim):
    theta = np.r_[np.full(27, 0.5), 3.0]
    x, logf = sim.simulate_AA(theta, rng=0, return_freq=True)
    assert x.shape == (3971,) and logf.shape == (3970,)
    assert np.all(np.isfinite(x))
    assert np.all(np.diff(logf) > 0)
    x2 = sim.simulate_AA(theta, rng=0, P=12.0)
    assert x2[-1] == 12.0


def test_simulate_TT_shapes(sim):
    x = sim.simulate_TT(15.0, rng=0)
    assert x.shape == (2970,) and np.all(np.isfinite(x))
    assert np.allclose(sim.simulate_TT(np.array([15.0]), rng=0), x)


def test_AA_data_is_unbiased_estimate_of_signal_plus_noise(sim):
    """Mean of many realisations -> Omega_s + Omega_n (both chunk factors have mean 1)."""
    theta = np.r_[np.full(26, 0.5), 0.75, 3.0]           # flat signal at 10^-8
    P, A = 15.0, 3.0
    xs = np.array([sim.simulate_AA(theta, rng=s, P=P) for s in range(200)])
    data = 10 ** xs[:, :2970]
    expect = 1e-8 + Omega_noiseh2_AA(sim.f[:2970], 2.5e9, 3e8, P, A)
    rel = data.mean(0) / expect - 1
    # each point has std 1/sqrt(94*200) ~ 0.7 %
    assert abs(rel.mean()) < 2e-3 and np.std(rel) < 0.012


@pytest.mark.parametrize("k", [94, 10])
def test_gamma_matches_gaussian_chunk_average(k):
    from scipy import stats

    f = np.logspace(-4, -3, 4000)
    a = Simulator.__new__(Simulator)
    a.n_chunks, a.method = k, "gamma"
    b = Simulator.__new__(Simulator)
    b.n_chunks, b.method = k, "gaussian"
    ga = a._chunk_average(f.size, np.random.default_rng(0))
    gb = b._chunk_average(f.size, np.random.default_rng(1))
    assert stats.ks_2samp(ga, gb).pvalue > 1e-3
    assert abs(ga.mean() - 1) < 0.01 and abs(ga.var() * k - 1) < 0.1


@pytest.mark.parametrize("name,params", [
    ("powerlaw", [-9, 0.6]),
    ("peak", [-9.0, 0.002, 0.2]),
    ("wiggly", [-10, 2, 0.1]),
    ("bpowerlaw", [-11, -1, 2 / 3, 0.002]),
    ("dpeak", [-11.0, -10.0, 0.001, 0.01, 0.1, 0.1]),
    ("three_peaks", [-10.0, -10.0, -10.0, 5e-4, 2e-3, 8e-3, 0.1, 0.1, 0.1]),
])
def test_different_signals(sim, name, params):
    d_AA, d_TT, f_AA, f_TT = sim.different_signals(name, params, (15.0, 3.0), rng=0)
    assert d_AA.shape == (3971,) and d_TT.shape == (3970,)
    assert f_AA.shape == f_TT.shape == (3970,)
    assert d_AA[-1] == 15.0
    assert np.all(np.isfinite(d_AA)) and np.all(np.isfinite(d_TT))
    assert template_omega(name, sim.f[:10], params).shape == (10,)


def test_template_names():
    assert set(TEMPLATES) == {"powerlaw", "peak", "wiggly", "bpowerlaw", "dpeak", "three_peaks"}
    np.testing.assert_allclose(template_omega("zPowerLaw", [1e-3], [-9, 2]), [1e-9])
    with pytest.raises(ValueError):
        template_omega("powerlaw", [1e-3], [1, 2, 3])


def test_batch_reproducible(sim):
    th = np.random.default_rng(0).uniform(size=(5, 28))
    th[:, 27] = 3
    a = sim.simulate_AA_batch(th, seed=1, progress=False)
    b = sim.simulate_AA_batch(th, seed=1, progress=False)
    assert a.shape == (5, 3971) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    t = sim.simulate_TT_batch(np.full(3, 15.0), seed=1, progress=False)
    assert t.shape == (3, 2970)
