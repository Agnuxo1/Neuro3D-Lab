# Modelo físico

Este documento fija las convenciones y ecuaciones que usa `optlab`. Todo el código debe ajustarse a él. Si una ecuación cambia, se cambia aquí primero y después el código, con un contrato nuevo.

## 1. Convenciones

- Campo escalar complejo `u(x, y)` en un plano transversal. Dependencia temporal `exp(-i·ω·t)`.
- Unidades SI: metros y segundos. La escena de Blender se convierte a metros con `scene.unit_settings.scale_length` (por defecto 1 unidad de Blender = 1 m).
- Rejilla cuadrada `n × n` con paso `pixel_m`. Índices de array: `[fila, columna] = [y, x]`.
- Número de onda en un medio de índice `n`: `k = 2π·n/λ`.
- Fase de camino óptico: `φ = 2π·OPL/λ`, con `OPL = Σ n_i·d_i`.

## 2. Propagación libre (espectro angular)

Para una distancia `d` en un medio de índice 1:

```
U = fft2(u)
H(fx, fy) = exp(i·2π·d·sqrt(1/λ² − fx² − fy²))   si fx² + fy² < 1/λ²
H(fx, fy) = 0                                     en otro caso (evanescente)
P_d(u) = ifft2(H · U)
```

`fx`, `fy` son las frecuencias de `numpy.fft.fftfreq(n, d=pixel_m)`.

- Es exacta para componentes propagantes, sin aproximación paraxial.
- Las componentes evanescentes se descartan. Esto pierde energía si el campo tiene componentes de frecuencia espacial mayor que `1/λ`, y lo mide el contrato de energía.
- Adjunto respecto al producto interno complejo: `P_d^H(g) = ifft2(conj(H) · fft2(g))`.

## 3. Elementos de fase y absorción

Un plano de altura `h(x, y)` (metros) con índice complejo `n_mat + iκ` en un medio de índice `n_med`:

```
t(x, y) = exp(i·2π·(n_mat − n_med)·h/λ) · exp(−2π·κ·h/λ)
```

- `|t| = 1` si `κ = 0`.
- Intensidad absorbida: `exp(−4π·κ·h/λ)`, es decir, coeficiente de absorción `α = 4πκ/λ`.
- Capas de fase pura: `t = exp(iφ_k)` con `φ_k` libre en `[0, 2π)`. Para el D2NN, `φ_k` se obtiene de la altura o se optimiza directamente.

## 4. Interfaces: Fresnel

Incidencia normal entre `n1` y `n2`, sin absorción:

```
r = (n1 − n2)/(n1 + n2)      R = r²      T = 1 − R
```

Incidencia oblicua con ley de Snell `n1·sinθi = n2·sinθt`:

```
r_s = (n1 cosθi − n2 cosθt)/(n1 cosθi + n2 cosθt)
r_p = (n2 cosθi − n1 cosθt)/(n2 cosθi + n1 cosθt)
R = |r|²
T = (n2 cosθt)/(n1 cosθi) · |t|²
```

Reflexión total interna cuando `n1·sinθi > n2`: `R = 1`, `T = 0`.

Las reflexiones entre capas no están incluidas en la versión 0.1 (ver §8).

## 5. Interferometría

Divisor 50/50 `B = (1/√2)·[[1, i], [i, 1]]`. Desfasador `P(θ) = diag(e^{iθ}, 1)`.

```
MZI(θ, φ) = B · P(θ) · B · P(φ)
```

Propiedades comprobadas por el contrato de interferometría: unitariedad para todo `θ, φ`; con `φ = 0`, `|T11|² = sin²(θ/2)`.

## 6. Red difractiva (D2NN)

Entrada `u0`. `L` capas de fase y `L+1` propagaciones:

```
b_k = P_{d_k}(a_{k−1}),   a_0 = u0,   a_k = b_k · exp(iφ_k)     (k = 1..L)
f   = P_{d_{L+1}}(a_L)
I   = |f|²
s_c = scale · Σ_{x ∈ región_c} I(x)
p   = softmax(s),   L_ce = −log p_y
```

Gradiente analítico, con la convención `ḡ = ∂L/∂z*`:

```
ḡ_f       = (∂L/∂I) · f,      ∂L/∂I = scale · Σ_c (p_c − δ_{cy}) · mask_c
ḡ_{a_L}   = P^H_{d_{L+1}}(ḡ_f)
para k = L..1:
    ∂L/∂φ_k = −2 · Im(ḡ_{a_k} · a_k)
    ḡ_{b_k} = ḡ_{a_k} · conj(exp(iφ_k))
    ḡ_{a_{k−1}} = P^H_{d_k}(ḡ_{b_k})
```

El gradiente se verifica con diferencias finitas centradas en el contrato `CONTRATO-001`.

## 7. Caminos ópticos desde Blender

Blender no simula ondas. Se usa para geometría: posición de los planos, índices de cada material y, si hace falta, longitudes de camino medidas por trazado de rayos. Un rayo `origin + t·direction` que atraviesa planos en `z_1 < z_2 < ...` da `d_i = z_{i+1} − z_i` y `OPL = Σ n_medium_i · d_i`. La comprobación del add-on verifica que los planos caen donde dice el JSON.

## 8. Limitaciones explícitas (versión 0.1)

- Modelo escalar: sin polarización ni campos vectoriales. Para incidencia oblicua en planos de fase la aproximación escalar no es válida.
- Sin reflexiones múltiples entre capas. Solo se propaga hacia delante.
- Sin dispersión espectral: `n` es constante en `λ`.
- Sin calibración en vatios: la intensidad es relativa, normalizada.
- Las regiones del detector son cajas rectangulares y no se solapan por contrato.
- Los resultados son simulaciones. Ninguna figura de este repo implica un dispositivo fabricado.

## 9. Contratos de validación

Cada comprobación numérica tiene un contrato escrito antes de ejecutar la prueba, con umbrales fijados de antemano. Un umbral no se relaja después de ver el resultado. Un fallo se publica igual, con su valor numérico. Ver [CONTRACTS/](CONTRACTS/).
