"""Tests for ``synaptic_ssl.analysis.puncta_morphology.fit_ellipse``.

Every test draws synthetic Gaussian spots whose true size, shape and
angle are known, and checks that the fit measures them back.
"""
import numpy as np
import pytest

from synaptic_ssl.analysis.puncta_morphology import fit_ellipse


def _spot(shape, row, col, s_major, s_minor, theta_deg, amp=1.0):
    """Elliptical Gaussian; major axis at ``theta_deg`` from the column axis."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    t = np.radians(theta_deg)
    u = (xx - col) * np.cos(t) + (yy - row) * np.sin(t)
    w = -(xx - col) * np.sin(t) + (yy - row) * np.cos(t)
    return amp * np.exp(-0.5 * (u / s_major) ** 2 - 0.5 * (w / s_minor) ** 2)


def test_round_spot():
    img = _spot((41, 41), 20.3, 19.6, 2.0, 2.0, 0.0)
    e = fit_ellipse(img, 20.0, 20.0, 2.0)
    assert e is not None
    assert e["eccentricity"] < 0.05
    assert e["a"] == pytest.approx(4.0, rel=0.1)
    assert e["b"] == pytest.approx(4.0, rel=0.1)
    assert abs(e["row"] - 20.3) < 0.1
    assert abs(e["col"] - 19.6) < 0.1


@pytest.mark.parametrize("theta", [30.0, -60.0, 85.0])
def test_elongated_rotated_spot(theta):
    img = _spot((51, 51), 25.0, 25.0, 3.0, 1.5, theta)
    e = fit_ellipse(img, 25.0, 25.0, 2.2)
    assert e is not None
    assert e["a"] == pytest.approx(6.0, rel=0.1)
    assert e["b"] == pytest.approx(3.0, rel=0.1)
    assert e["theta_deg"] == pytest.approx(theta, abs=3.0)
    assert e["eccentricity"] == pytest.approx(np.sqrt(1 - 0.25), abs=0.03)


def test_constant_offset_does_not_change_result():
    img = _spot((51, 51), 25.0, 25.0, 3.0, 1.5, 30.0)
    e0 = fit_ellipse(img, 25.0, 25.0, 2.2)
    e1 = fit_ellipse(img + 0.4, 25.0, 25.0, 2.2)
    for key in ("a", "b", "theta_deg", "row", "col", "intensity"):
        assert e1[key] == pytest.approx(e0[key], abs=1e-3)


def test_border_and_empty_return_none():
    img = _spot((41, 41), 3.0, 20.0, 2.0, 2.0, 0.0)
    assert fit_ellipse(img, 3.0, 20.0, 2.0) is None
    assert fit_ellipse(np.zeros((41, 41)), 20.0, 20.0, 2.0) is None


def test_theta_range():
    rng = np.random.default_rng(0)
    for theta in rng.uniform(-180, 180, 20):
        img = _spot((51, 51), 25.0, 25.0, 3.0, 1.5, theta)
        e = fit_ellipse(img, 25.0, 25.0, 2.2)
        assert -90.0 < e["theta_deg"] <= 90.0


def test_noise_does_not_inflate_size_or_favour_diagonals():
    """With 10 % noise the median size stays within 5 % of the truth, and
    the angles of round spots are not piled up at +-45 deg (the artifact
    of the old square-window moment fit)."""
    rng = np.random.default_rng(1)
    sizes, angles = [], []
    for _ in range(150):
        r, c = 20.0 + rng.uniform(-0.5, 0.5, 2)
        img = _spot((41, 41), r, c, 1.8, 1.8, 0.0) + rng.normal(0, 0.1, (41, 41))
        e = fit_ellipse(img, round(r), round(c), 1.8)
        if e is not None:
            sizes.append(np.sqrt(e["a"] * e["b"]))
            angles.append(e["theta_deg"])
    assert len(sizes) > 140
    assert np.median(sizes) == pytest.approx(3.6, rel=0.05)
    near_diag = np.mean(np.abs(np.abs(angles) - 45.0) < 15.0)
    assert 0.2 < near_diag < 0.47  # uniform angles give 1/3


def test_noisy_elongated_spot_keeps_shape():
    rng = np.random.default_rng(2)
    ecc, theta = [], []
    for _ in range(50):
        img = _spot((51, 51), 25.0, 25.0, 3.0, 1.5, 30.0) + rng.normal(0, 0.1, (51, 51))
        e = fit_ellipse(img, 25.0, 25.0, 2.2)
        ecc.append(e["eccentricity"])
        theta.append(e["theta_deg"])
    assert np.median(ecc) == pytest.approx(np.sqrt(1 - 0.25), abs=0.05)
    assert np.median(theta) == pytest.approx(30.0, abs=3.0)


def test_neighbour_is_fitted_jointly():
    img = _spot((51, 51), 25.0, 25.0, 1.8, 1.8, 0.0)
    img += _spot((51, 51), 25.0, 31.0, 1.8, 1.8, 0.0, amp=2.0)
    alone = fit_ellipse(img, 25.0, 25.0, 1.8)
    joint = fit_ellipse(
        img, 25.0, 25.0, 1.8, neighbours=np.array([[25.0, 25.0, 1.8], [25.0, 31.0, 1.8]])
    )
    assert joint is not None
    assert joint["a"] == pytest.approx(3.6, rel=0.05)
    assert joint["b"] == pytest.approx(3.6, rel=0.05)
    assert abs(joint["col"] - 25.0) < 0.1
    # Without the neighbour list the bright neighbour stretches the fit.
    assert alone is None or alone["a"] > 1.1 * joint["a"]
