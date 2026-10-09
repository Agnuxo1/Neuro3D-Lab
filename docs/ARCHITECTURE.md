# Arquitectura

Neuro3D-Lab separa la **geometría** (Blender) del **cálculo físico** (Python/numpy). Blender es el banco de laboratorio: aquí se colocan los planos, se pintan las alturas de fase y se ven los resultados. El campo lo calcula `optlab`.

```
Blender (banco)                         optlab (física)                  Blender (resultados)
───────────────                         ───────────────                  ────────────────────
SOURCE, LAYER_k, DETECTOR  ──export──▶  bench.json + *_height.npy  ──▶  simulate_bench()
alturas (imagen float)                                                  │
                                                                        ▼
imagen de intensidad  ◀──import───────  detector_intensity.npy  ◀──────┘
```

## Módulos

| Ruta | Responsabilidad |
|---|---|
| `src/optlab/units.py` | Número de onda, fase desde OPL, camino óptico |
| `src/optlab/propagation.py` | Espectro angular: `transfer_function`, `angular_spectrum`, adjunto |
| `src/optlab/elements.py` | Elementos de fase con absorción, máscaras de detector |
| `src/optlab/fresnel.py` | Coeficientes de Fresnel normales y oblicuos, TIR |
| `src/optlab/interferometer.py` | Divisor, desfasador, MZI unitario |
| `src/optlab/d2nn.py` | Red difractiva: propagación, pérdida, gradiente analítico, Adam |
| `src/optlab/bench.py` | Lectura y validación de `optlab.bench/1`, simulación del banco |
| `src/optlab/stats.py` | Intervalo de Wilson |
| `src/optlab/registry.py` | Registro de ejecuciones en JSONL, solo añadir |
| `blender/neuro3d_lab/` | Add-on: construir escena demo, exportar, importar, trazado de rayos |
| `blender/selftest/run_selftest.py` | Prueba headless del add-on y del contrato con optlab |
| `tests/` | Pruebas de cada módulo, con umbrales de `docs/CONTRACTS/` |

## Formato `optlab.bench/1`

```json
{
  "schema": "optlab.bench/1",
  "wavelength_m": 6.33e-7,
  "grid": {"n": 64, "pixel_m": 8e-6},
  "source": {"z_m": 0.0},
  "layers": [
    {"name": "LAYER_1", "z_m": 0.02, "n_medium": 1.0, "n_material": 1.5,
     "kappa": 0.0, "height_npy": "LAYER_1_height.npy", "height_max_m": 1e-6}
  ],
  "detector": {"z_m": 0.06,
               "regions": [{"label": 0, "cx_m": 0.0, "cy_m": 0.0, "w_m": 1e-4, "h_m": 1e-4}]},
  "output": {"intensity_npy": "detector_intensity.npy"}
}
```

Reglas:

- Las `z_m` deben crecer estrictamente: fuente < capas < detector.
- Distancia entre planos = diferencia de `z_m`.
- Altura real = `height_npy · height_max_m`, con forma `(n, n)`.
- Rutas relativas al directorio del JSON.
- Campo inicial sin `amplitude_npy`: onda plana de amplitud 1.

## Convención de imagen (Blender ↔ numpy)

Blender guarda las imágenes con el origen abajo a la izquierda. El exportador (`blender/neuro3d_lab/bench_io.py`) las convierte a arrays numpy con esta regla:

- `a[i, j]`: `i` es la fila contada desde la **parte superior** de la imagen; `j` es la columna contada desde la izquierda. Es `np.flipud` de los píxeles de Blender.
- Coordenadas físicas del plano: `+x` = columna creciente (hacia la derecha); `+y` = hacia arriba, es decir, fila decreciente (la fila 0 tiene el `y` máximo).
- `optlab.elements.detector_masks` usa la misma regla: `x_j = (j−(n−1)/2)·pixel`, `y_i = −(i−(n−1)/2)·pixel`.
- La altura exportada va en `[0, 1]`. La altura real es `height_npy · height_max_m`.

Esta convención quedó fijada tras una fe de erratas: la versión 0.1 inicial tenía `+y` hacia abajo en `detector_masks`. La prueba `tests/test_orientation.py` la protege contra una regresión.

## Por qué no Cycles ni Eevee

Cycles y Eevee trazan rayos o rasterizan, pero no mantienen fase coherente entre trayectos. Un interferómetro o una red difractiva necesita sumar campos complejos con su fase. Por eso Blender aporta geometría y visualización, y el campo lo calcula `optlab`.
