"""Coeficientes de Fresnel en interfaz dielectrica sin absorcion.

Normal: r=(n1-n2)/(n1+n2), R=r^2, T=1-R. Oblicuo: Snell n1 sin(ti)=n2 sin(tt);
r_s=(n1 ci - n2 ct)/(n1 ci + n2 ct); r_p=(n1 ct - n2 ci)/(n1 ct + n2 ci)
(signo elegido para que r_p=r_s en incidencia normal; r_p=0 en Brewster);
t_s=2 n1 ci/(n1 ci+n2 ct); t_p=2 n1 ci/(n1 ct+n2 ci) (amplitud de E);
R=|r|^2, T=(n2 ct)/(n1 ci)|t|^2. Reflexion total interna: R=1, T=0, t=0.
"""
from __future__ import annotations

import math
import cmath


def normal_incidence(n1, n2):
    r = (n1 - n2) / (n1 + n2)
    R = r * r
    return {"r": r, "t": 2 * n1 / (n1 + n2), "R": R, "T": 1.0 - R}


def oblique(n1, n2, theta_i_rad, polarization):
    if polarization not in ("s", "p"):
        raise ValueError("polarization must be 's' or 'p'")
    ci = math.cos(theta_i_rad)
    sin_t = n1 * math.sin(theta_i_rad) / n2
    if abs(sin_t) > 1.0:  # TIR
        ct = cmath.sqrt(1 - sin_t**2)
        if polarization == "s":
            r = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
        else:
            r = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
        return {"r": r, "t": 0.0, "R": 1.0, "T": 0.0}
    ct = math.sqrt(1 - sin_t**2)
    if polarization == "s":
        r = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
        t = 2 * n1 * ci / (n1 * ci + n2 * ct)
    else:
        r = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
        t = 2 * n1 * ci / (n1 * ct + n2 * ci)
    R = r * r
    T = (n2 * ct) / (n1 * ci) * t * t
    return {"r": r, "t": t, "R": R, "T": T}
