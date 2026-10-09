# CONTRATO-002: validación del propagador frente a soluciones analíticas

**Estado:** preregistrado el 2026-10-09 (UTC), antes de ejecutar `experiments/e01…e03`. Commit de referencia: el que contiene este archivo.
**Pregunta:** ¿reproduce `optlab.propagation.angular_spectrum` soluciones de difracción conocidas dentro de tolerancias fijadas de antemano?
**Regla:** los umbrales no se modifican después de ver resultados. Un fallo se publica con su valor.

## Condición de validez del método (no es un umbral, es un requisito)

El espectro angular sin banda limitada es fiable mientras la distancia cumpla `z < z_max = N·dx²/λ` (aliasing de la función de transferencia). Cada experimento declara su `z` máximo y comprueba que se cumple. Si no se cumple, el resultado no cuenta y se reporta como fuera de dominio.

## E1: haz gaussiano (solución exacta de Gouy)

- Parámetros: λ = 633 nm, `w0` = 20 µm, `N` = 1023 (impar, el centro cae en un píxel), `dx` = 2 µm, campo inicial `u = exp(−r²/w0²)`.
- `z_R = π·w0²/λ` ≈ 2,0 mm. Distancias: z ∈ {0,25; 0,5; 1; 2} · z_R. `z_max` = 1023·(2e-6)²/633e-9 ≈ 6,5 mm: se cumple.
- Analítico: `w(z) = w0·sqrt(1+(z/z_R)²)`; amplitud en el eje `|u(0,z)| = 1/sqrt(1+(z/z_R)²)`; fase en el eje `arg u(0,z) − k·z = −arctan(z/z_R)` (fase de Gouy).
- Métricas: `w_num = 2·sqrt(⟨x²⟩)` con la densidad de intensidad; amplitud y fase en el píxel central.
- **Umbrales:** `|w_num/w(z) − 1| ≤ 0,005` para todo z; `|amp_num/amp(z) − 1| ≤ 0,005`; `|Δfase| ≤ 0,005 rad`; `|E(z)/E(0) − 1| ≤ 1e-9` (energía, banda limitada, sin pérdidas).

## E2: autoimagen de Talbot (red de amplitud cosenoidal)

- Parámetros: λ = 633 nm, `N` = 1024, `dx` = 2 µm, periodo `p` = 16 µm (8 píxeles, exactamente periódico en la rejilla). Transmitancia `t(x) = (1 + cos(2πx/p))/2`.
- Distancia de Talbot `z_T = 2p²/λ` ≈ 0,81 mm. `z_max` = 1024·(2e-6)²/633e-9 ≈ 6,5 mm: se cumple.
- Analítico: `I(z_T) = I(0)` (autoimagen exacta); en `z_T/2` el patrón no es una autoimagen.
- **Umbrales:** error RMS relativo `‖I(z_T) − I(0)‖/‖I(0)‖ ≤ 0,01`; `‖I(z_T/2) − I(0)‖/‖I(0)‖ ≥ 0,5` (control negativo: el patrón de medio periodo no debe confundirse con la autoimagen).

## E3: apertura circular, intensidad en el eje (Fresnel)

- Parámetros: λ = 633 nm, `N` = 1023, `dx` = 8 µm, apertura circular de radio nominal `a` = 0,2 mm, campo incidente uniforme de amplitud 1.
- `z_max` = 1023·(8e-6)²/633e-9 ≈ 103 mm. Distancias: z ∈ {20; 40; 80} mm, números de Fresnel `N_F = a²/(λz)` ≈ 3,2; 1,6; 0,8.
- Analítico: `I(0,z)/I0 = 4·sin²(π·N_F/2)`.
- Corrección de discretización (declarada aquí, no ajustada a los datos): el radio efectivo es `a_eff = sqrt(A_muestreada/π)`, donde `A_muestreada` es el número de píxeles dentro de la apertura multiplicado por `dx²`. El analítico usa `a_eff`.
- **Umbral:** `|I_num(0,z) − 4·sin²(π·N_F(a_eff)/2)| ≤ 0,05` (en unidades de I0) para las tres distancias.

## E4 (exploratorio, sin umbral de aprobación): dominio de validez

Se mide el error de E1 (haz gaussiano, amplitud en el eje) para `z` = 0,5·z_max, z_max y 2·z_max, con `dx` fijo. Se reporta el error para documentar cuándo deja de valer el método. No cuenta como aprobado ni suspendido.

## Qué no cubre

- Dispersión espectral, polarización, reflexiones entre planos (ver docs/MODEL.md §8).
- Validación contra un solver de ondas vectorial completo (pendiente).
- Banda limitada (Matsushima & Shimobaba 2009, Opt. Express 17, 19662): no implementada en la versión 0.1. Se evaluará en un contrato posterior si E4 muestra un fallo relevante.

## Registro de resultados

**Ejecución:** `python experiments/run_validation.py` (CPU, ~7 s). Resumen en `experiments/results/validation_summary.json`; datos completos fuera del repo.

| Exp. | Medida | Valor máximo observado | Umbral | Veredicto |
|---|---|---|---|---|
| E1 | error relativo de `w(z)` | 4,1e-5 | 0,005 | cumple |
| E1 | error relativo de amplitud en el eje | 5,6e-6 | 0,005 | cumple |
| E1 | error de fase de Gouy (rad) | 1,2e-5 | 0,005 | cumple |
| E1 | error relativo de energía | 3,3e-16 | 1e-9 | cumple |
| E2 | error RMS en `z_T` | 2,05e-6 | 0,01 | cumple |
| E2 | error RMS en `z_T/2` (debe ser grande) | 1,35 | ≥ 0,5 | cumple |
| E3 | error absoluto de I(0,z), z = 20, 40, 80 mm | 7,2e-3 | 0,05 | cumple |
| E4 | exploratorio: haz gaussiano a 0,5, 1 y 2 · z_max | error ≤ 5e-5 | sin veredicto | — |

**Control negativo del arnés (no pre-registrado, comprobación de poder):** propagar con `−z` en lugar de `+z` produce un error de fase de Gouy de 0,49 rad (z = 0,25 z_R) y 1,57 rad (z = z_R). Ambos superan el umbral de 0,005 por varios órdenes de magnitud, así que el criterio sí detecta un error de signo.

**Lectura honesta:**
- La validación cubre el propagador para campos suaves y periódicos. Un haz gaussiano y un patrón periódico tienen poco contenido de alta frecuencia, así que E4 no muestra degradación ni a 2·z_max. El límite `z_max` es conservador para campos suaves; no implica que el método falle a esa distancia para campos con bordes nítidos. Esto no se ha probado aquí.
- E3 usa una apertura con borde nítido y sí es el caso que discrimina el dominio, pero está dentro de `z_max` (103 mm). Un contrato posterior debería probar bordes nítidos más allá de `z_max`.
- E1 y E2 están dentro de tolerancias muy holgadas. Las tolerancias se fijaron antes de medir; no las ajusto a la baja ahora.
