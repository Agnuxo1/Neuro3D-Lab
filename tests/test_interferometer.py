import numpy as np
from optlab import mzi, is_unitary


def test_unitary_random():
    rng = np.random.default_rng(3)
    for _ in range(50):
        th, ph = rng.uniform(0, 2 * np.pi, 2)
        assert is_unitary(mzi(th, ph), atol=1e-12)


def test_t11():
    for th in np.linspace(0, 2 * np.pi, 17):
        assert abs(abs(mzi(th, 0.0)[0, 0]) ** 2 - np.sin(th / 2) ** 2) < 1e-12


def test_pi_and_zero_states():
    # Con la definicion dada (|T11|^2 = sin^2(theta/2)): theta=pi es estado "bar"
    # (todo el puerto 1 permanece en el 1); theta=0 es estado "cross" (1 -> 2).
    U = mzi(np.pi, 0.0)
    assert abs(abs(U[0, 0]) - 1) < 1e-12 and abs(U[1, 0]) < 1e-12
    U = mzi(0.0, 0.0)
    assert abs(abs(U[1, 0]) - 1) < 1e-12 and abs(U[0, 0]) < 1e-12


def test_is_unitary_rejects_small_scaling():
    # hallazgo 6: sin rtol, un divisor escalado por 1+5e-7 no es unitario a 1e-12
    assert not is_unitary(mzi(0.7, 0.3) * (1 + 5e-7), atol=1e-12)
    assert is_unitary(mzi(0.7, 0.3), atol=1e-12)
