import math
import pytest
from optlab import normal_incidence, oblique


def test_energy():
    f = normal_incidence(1.0, 1.5)
    assert abs(f["R"] + f["T"] - 1) < 1e-12
    for pol in "sp":
        for th in (0.0, 0.3, 0.8, 1.2):
            o = oblique(1.0, 1.5, th, pol)
            assert abs(o["R"] + o["T"] - 1) < 1e-12


def test_normal_matches():
    n = normal_incidence(1.0, 1.5)
    for pol in "sp":
        o = oblique(1.0, 1.5, 0.0, pol)
        assert abs(o["R"] - n["R"]) < 1e-12 and abs(o["T"] - n["T"]) < 1e-12
        assert abs(o["r"] - n["r"]) < 1e-12


def test_brewster():
    n1, n2 = 1.0, 1.5
    assert abs(oblique(n1, n2, math.atan(n2 / n1), "p")["r"]) < 1e-12


@pytest.mark.parametrize("pol", ["s", "p"])
def test_tir(pol):
    for deg in (41.9, 60.0):
        o = oblique(1.5, 1.0, math.radians(deg), pol)
        assert o["R"] == 1.0 and o["T"] == 0.0


def test_rp_sign_convention():
    # hallazgo 4: r_p = r_s en incidencia normal; en Brewster r_p = 0 y cambia de signo
    n1, n2 = 1.0, 1.5
    assert abs(oblique(n1, n2, 0.0, "p")["r"] - oblique(n1, n2, 0.0, "s")["r"]) < 1e-12
    assert oblique(n1, n2, 0.0, "p")["r"] < 0
    thb = math.atan(n2 / n1)
    assert oblique(n1, n2, thb - 0.05, "p")["r"] < 0 < oblique(n1, n2, thb + 0.05, "p")["r"]


def test_energy_dense_to_light_below_critical():
    for pol in "sp":
        for deg in (0.0, 20.0, 40.0):
            o = oblique(1.5, 1.0, math.radians(deg), pol)
            assert abs(o["R"] + o["T"] - 1) < 1e-12
