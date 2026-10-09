# CONTRATO-001: núcleo físico (optlab 0.1.0)

**Fecha de registro:** 2026-10-09, antes de ejecutar los tests del núcleo.
**Alcance:** `src/optlab/` y `tests/`.
**Regla:** los umbrales de esta página no se relajan después de ver resultados. Si un test falla, se publica el valor medido y la causa.

## Pruebas y umbrales

| Área | Prueba | Umbral |
|---|---|---|
| Propagación | Campo uniforme se propaga como `exp(i·2π·d/λ)` | error absoluto ≤ 1e-12 |
| Propagación | Conservación de energía de un haz gaussiano, `waist = 20·pixel`, `n = 256`, `pixel = λ/4`, `d = 2 mm` | error relativo ≤ 1e-10 |
| Propagación | Inversión `P_{−d}(P_d(u)) = u` para haz suave | error relativo ≤ 1e-10 |
| Propagación | Adjunto: `<P u, v> = <u, P^H v>` | error relativo ≤ 1e-10 |
| Fresnel | `R + T = 1` sin TIR, normal y oblicua | error absoluto ≤ 1e-12 |
| Fresnel | Coincidencia s/p con incidencia normal en θ = 0 | error absoluto ≤ 1e-12 |
| Fresnel | Brewster: `r_p = 0` en `tan θB = n2/n1` | |r_p| ≤ 1e-12 |
| Fresnel | TIR con `n1 = 1.5 → n2 = 1.0` y θ > 41.81° | R = 1, T = 0 |
| Interferómetro | Unitariedad de `MZI(θ, φ)` en 50 pares aleatorios | error ≤ 1e-12 |
| Interferómetro | `|T11|² = sin²(θ/2)` con φ = 0 | error absoluto ≤ 1e-12 |
| Elementos | `|t| = 1` si κ = 0; `|t| = exp(−2πκh/λ)` si κ > 0 | error absoluto ≤ 1e-12 |
| D2NN | Gradiente analítico frente a diferencias finitas centradas (paso 1e-6), n = 32, L = 2, λ = 633 nm, pixel 16 µm | error relativo máximo ≤ 1e-6 en 5 componentes aleatorias |
| D2NN | Aprendizaje sintético de 3 clases, 60 pasos de Adam | pérdida final < 50 % de la inicial y precisión de entrenamiento ≥ 0.90 |
| Bench | Round-trip JSON `optlab.bench/1`, n = 64, conservación sin absorción | error relativo ≤ 1e-6 |
| Bench | Validación: z no creciente lanza `ValueError` | obligatorio |
| Estadística | Intervalo de Wilson para 50/100 | (0.4038, 0.5962) con tolerancia 1e-3 |

## Qué no cubre este contrato

- Validación contra un solver de ondas completo (FDTD o BEM): pendiente de CONTRATO-002.
- Medidas experimentales: no existen.
- Rendimiento y tiempos de cómputo.

## Registro de resultados

Se completa tras ejecutar `python -m pytest -q`. Cada línea: fecha, commit, resumen de pytest y fallos, si los hay.

- **2026-10-09 ~20:00 UTC, primera ejecución (sin commit).** `python -m pytest -q` → `20 passed in 3.48s`. Self-test de Blender headless (`blender/selftest/run_selftest.py`, Blender 4.5.14): register, build, export, trace, schema, simulate, import, unregister → todos PASS. Log en `D:/PROJECTS/.cognition/neuro3d-lab/selftest_run1.log`.
- **Fe de erratas (2026-10-09, tarde UTC).** Auditoría del coordinador: `optlab.elements.detector_masks` asignaba `y` creciente con el índice de fila, es decir `+y` hacia abajo, contradiciendo la convención de imagen de `bench_io.py` (`+y` hacia arriba). Corregido en `elements.py` (`Y` = `-coords`). Se añade `tests/test_orientation.py` (2 pruebas de regresión). No se modificó ningún umbral. Efecto: las regiones de detector con `cy_m > 0` iluminaban la mitad inferior de la imagen; el self-test no lo detectaba porque no comprueba la orientación.
- **2026-10-09 20:56 UTC, tras la corrección.** `python -m pytest -q` → `22 passed in 4.90s`. Comprobado a mano con la fórmula antigua: una región en `cy = +8 µm` (n = 32, pixel 1 µm) cae en las filas 23–24, es decir, la mitad inferior, y la prueba de regresión fallaría; con la fórmula nueva cae en las filas 7–8.
- Auditoría independiente del código núcleo y add-on: hecha (T-001, `coordinacion/respuestas/T-001-auditoria.md`). Correcciones aplicadas en T-002 (`coordinacion/respuestas/T-002-correccion.md`): `medium_index` en `optlab.bench/1`, `is_unitary` sin `rtol`, empaquetado de alturas, trazado de rayos con épsilon en BU, exportación sin ruido float32, Build idempotente. No se modificó ningún umbral.
