# CONTRATO-003: red difractiva D2NN de fase en dígitos manuscritos

## English summary

**Purpose.** Tests whether a three-layer phase-only diffractive neural network (D2NN) classifies handwritten digits (`sklearn` `load_digits`, 8×8 images upsampled to 64×64) above chance and above a logistic regression on the same input, with three controls.

**Pre-registration status.** Pre-registered on 2026-10-09 (UTC), before running `experiments/run_d2nn_digits.py`. An addendum made the same day, before any data existed, fixes further details: the McNemar test (seed 0), the definition of control C3, mini-batch training, and the use of TRAIN and DEV only. Git history shows the contract (1a4fc37) and the addendum (f28ecb9) committed before the run started. The thresholds were not changed.

**Design (fixed in the contract).** Three phase layers with 18 mm between planes. Grid N = 128 with dx = 0.4 mm and λ = 0.75 mm (dx/λ = 0.53). Detector: ten 12×12 pixel regions, with logits scaled by β chosen on DEV. Adam optimizer, mini-batches of 32, 30 epochs. Hyperparameters (β, learning rate, logistic C) are chosen on DEV only, and TEST is used once.

**Hypotheses and thresholds.**
- H1 (primary): mean TEST accuracy over 5 seeds ≥ 0.85, and every seed ≥ 0.80.
- H2: paired difference (D2NN minus logistic regression) ≥ 0, and exact McNemar p < 0.05.
- C1: training labels permuted; mean TEST accuracy ≤ 0.15.
- C2: untrained random phases; mean TEST accuracy ≤ 0.20.
- C3: no phase layers (free propagation); TEST accuracy ≤ 0.20.

**Status: complete.** TEST was evaluated once per seed with the DEV-selected configuration; verdicts are in the registry below. Validation (DEV) accuracies are preliminary and are not results, so they are not reported here.

**What will be reported.** DEV and TEST accuracy per seed (mean and standard deviation), the TEST confusion matrix for the final configuration, all controls, compute time and network size.

**Post-hoc notes.**
- Provenance: the audit found that `experiments/run_d2nn_digits.py` was committed (af52aff) about nine minutes after its run started (log timestamp 23:02:43). Git history therefore cannot show that the committed script is the version that produced that run.
- Grid: CONTRACT-004 (T2) found that, at 18 mm, the N = 128 grid used here changes the detector intensity by about 4 % (relative L2 4.4e-2, threshold 2e-2) compared with N = 256. A robustness rerun of the final configuration at N = 256 is required before any D2NN number from this contract is cited. It will be labelled post hoc.

**Scope.** Simulation only; there is no hardware experiment. Fabrication noise, misalignment, inter-plane reflections and spectral dispersion are not covered.

---

**Estado:** preregistrado el 2026-10-09 (UTC), antes de ejecutar `experiments/run_d2nn_digits.py`. Commit de referencia: el que contiene este archivo.
**Pregunta:** ¿clasifica una D2NN de fase pura (3 capas) dígitos manuscritos con más precisión que el azar y que una regresión logística de parámetros comparables sobre la misma entrada?
**Regla:** la partición TEST se usa una sola vez, con la configuración elegida en DEV. No se selecciona nada con TEST. Todos los umbrales están fijados aquí.

## Datos

- `sklearn.datasets.load_digits`: 1797 imágenes 8×8, 10 clases (UCI, reconocimiento óptico de dígitos manuscritos).
- Partición estratificada, semilla 2026: TRAIN 60 % (1078), DEV 20 % (359), TEST 20 % (360).

## Codificación de la entrada

- Amplitud `a = img/16` ∈ [0, 1]; ampliación ×8 por repetición (`np.kron`) → 64×64 píxeles.
- Normalización de energía por muestra: `u0 = a/‖a‖₂`, de modo que `Σ|u0|² = 1`. Así la energía total a la salida es comparable entre muestras.
- Colocación centrada en una rejilla `N` = 128, con `dx` = 0,4 mm, `λ` = 0,75 mm (dx = 0,53 λ). `z_max` = 128·(0,4e-3)²/0,75e-3 ≈ 27 mm.

## Arquitectura (fijada)

- L = 3 capas de fase pura, `t = exp(iφ)`, inicialización uniforme en [0, 2π) con semilla por ejecución.
- Distancias entre planos: 18 mm (cuatro propagaciones: entrada→capa 1→capa 2→capa 3→detector). Todas < `z_max` = 27 mm.
- Detector: 10 regiones cuadradas de 12×12 píxeles (4,8 mm). Posiciones: dos filas (`y` = +20 px y −20 px) y cinco columnas (`x` = −40, −20, 0, 20, 40 px). Clase k → columna `k mod 5`, fila `k div 5` (fila 0 arriba). No se solapan.
- Logits: `s_k = β · E_k`, con `E_k` = energía en la región k y `β` ∈ {20, 50, 100} (elegido en DEV).
- Pérdida: entropía cruzada softmax. Gradiente analítico de `optlab.d2nn.DiffractiveNetwork.loss_and_grad`.

## Entrenamiento (fijado)

- Adam (β1 = 0,9; β2 = 0,999; ε = 1e-8), mini-lotes de 32, 30 épocas.
- Selección en DEV (seed 0, única): rejilla pre-registrada `β` ∈ {20, 50, 100} × `lr` ∈ {0,02; 0,05}. Ganador = mayor precisión DEV; empate → `β` menor.
- Final: configuración ganadora, seeds 0–4, TEST una sola vez.

## Hipótesis y umbrales

| Código | Afirmación | Criterio de aprobación |
|---|---|---|
| H1 (primaria) | La D2NN clasifica por encima del umbral de precisión | media de TEST sobre 5 seeds ≥ **0,85**, y cada seed ≥ **0,80** |
| H2 | La D2NN supera a una regresión logística de parámetros comparables | diferencia pareada (D2NN − base) ≥ 0 **y** p de McNemar exacto < 0,05 |
| C1 (control) | Con etiquetas permutadas en entrenamiento, no hay aprendizaje espurio | precisión TEST media ≤ **0,15** |
| C2 (control) | Sin entrenar (fases aleatorias) el readout no clasifica | precisión TEST media ≤ **0,20** |
| C3 (control) | Sin capas de fase (propagación libre) el readout no clasifica | precisión TEST ≤ **0,20** |

Base de comparación (H2): regresión logística de scikit-learn sobre la entrada 64×64 normalizada (4096 características, 40 970 parámetros; la D2NN tiene 3·128² = 49 152). Se reporta también la base de 8×8 crudas (650 parámetros). `C` de la logística elegido en DEV entre {0,01; 0,1; 1; 10}.

Intervalos: Wilson al 95 % para cada precisión TEST.

## Aclaraciones fijadas antes de ejecutar (addendum, mismo día, sin datos todavía)

Estas precisiones no cambian ningún umbral. Completan detalles que el texto original dejaba abiertos:

1. **McNemar (H2):** se aplica a las predicciones TEST de la configuración final con **seed 0**, frente a la regresión logística de 4096 características. Las seeds 1–4 se reportan sin test estadístico.
2. **Control C3 (sin capas de fase):** se implementa como las mismas cuatro propagaciones de 18 mm con todas las fases fijas a cero (`t = 1`). Es propagación libre por tramos, sin entrenamiento. No se usa una propagación única de 72 mm porque excede `z_max`.
3. **Entrenamiento:** mini-lotes de 32 muestras con el gradiente medio del lote, calculado con `DiffractiveNetwork.loss_and_grad`, y un paso de Adam por lote. No se usa `fit` (que es de lote completo, un paso por época).
4. **Uso de TRAIN y DEV:** el entrenamiento usa solo TRAIN. DEV sirve solo para elegir `β`, `lr` y `C` de la logística. TEST no interviene en ninguna elección.
5. **Ampliación y energía:** la ampliación ×8 se hace antes de normalizar `u0 = a/‖a‖₂`. Después se rellena con ceros hasta 128×128.
6. **Logística de 8×8:** solo se reporta, no entra en ningún criterio de aprobación.

## Qué se reportará siempre

- Precisión DEV y TEST por seed, media y desviación típica.
- Matriz de confusión de TEST para la configuración final.
- Todos los controles, aunque pasen o fallen.
- Tiempo de cómputo y tamaño de la red.

## Qué no cubre

- Precisión experimental (no hay hardware): solo simulación.
- Ruido de fabricación, desalineación, reflexiones entre planos, dispersión espectral.
- Comparación con redes neuronales convolucionales: se menciona en el informe, no se contrasta aquí.
- Escalado a MNIST: pendiente de un contrato propio.

## Registro de resultados

**Result status.** The final TEST run is complete. TEST was used once, with the configuration chosen on DEV. Verdicts: H1 pass, H2 fail, C1 to C3 pass.

**Run record.**
- Result file: `experiments/results/d2nn_digits_summary.json`.
- Date (UTC): run started on 2026-10-09 at about 21:03 UTC (log timestamp 23:02:43 at UTC+2). Runtime 13,019 s (about 3.6 h). The summary file was written on 2026-10-10 at about 00:39 UTC.
- Commit: not yet recorded. At HEAD `81f0d8b` the summary file was untracked, so the commit that adds it must be written here.
- Script: `experiments/run_d2nn_digits.py` (committed in `af52aff`; see the provenance note at the end of this section).
- Network: N = 128, dx = 0.4 mm, λ = 0.75 mm, three phase layers, 18 mm between planes, 3·128² = 49,152 phase parameters.
- Selected configuration (DEV, seed 0): β = 100, lr = 0.02, DEV accuracy 0.9554.
- Logistic baselines: both selected C = 10 on DEV.

**Verdicts (TEST, one use).**

| Code | Criterion (contract) | Result | Threshold | Verdict |
|---|---|---|---|---|
| H1 (primary) | Mean TEST over 5 seeds; each seed | mean 0.9622 (sample SD 0.0032); minimum 0.9583 | mean ≥ 0.85; each seed ≥ 0.80 | Pass (`cumple`) |
| H2 | Paired difference D2NN − logistic 64×64; McNemar exact on seed 0 | difference +0.0094 against 0.9528; p = 0.21875 (b = 5, c = 1) | difference ≥ 0 and p < 0.05 | Fail (`no cumple`) |
| C1 | Training labels permuted | mean 0.1256; per seed 0.0806, 0.1667, 0.1333, 0.1583, 0.0889 | mean ≤ 0.15 | Pass (`cumple`) |
| C2 | Untrained (random) phases | mean 0.1244; per seed 0.1333, 0.1250, 0.1056, 0.1250, 0.1333 | mean ≤ 0.20 | Pass (`cumple`) |
| C3 | No phase layers (zero phases) | 0.025 | ≤ 0.20 | Pass (`cumple`) |

**Per-seed accuracy (DEV and TEST), with Wilson 95 % intervals for TEST (n = 360).**

| Seed | DEV | TEST | Wilson 95 % (TEST) |
|---|---|---|---|
| 0 | 0.9554 | 0.9639 | [0.9392, 0.9788] |
| 1 | 0.9694 | 0.9611 | [0.9358, 0.9767] |
| 2 | 0.9499 | 0.9583 | [0.9324, 0.9746] |
| 3 | 0.9610 | 0.9611 | [0.9358, 0.9767] |
| 4 | 0.9554 | 0.9667 | [0.9426, 0.9808] |
| Mean (sample SD) | 0.9582 (0.0074) | 0.9622 (0.0032) | not reported; see note 4 |

**Confusion matrix (seed 0, TEST, total 360).** Correct counts per digit (diagonal) for 0 to 9: 34, 34, 35, 37, 36, 36, 36, 36, 30, 33 (sum 347, accuracy 0.9639). The summary JSON does not contain the full matrix, so the off-diagonal counts are not reported here.

**Notes.**
1. **H2 failure.** The D2NN does not beat the linear baseline. The difference from the 64×64 logistic regression (0.9528) is +0.0094, and the exact McNemar p is 0.21875 on six discordant pairs (5 with only the D2NN correct, 1 with only the baseline correct). H2 therefore fails. There is no evidence of superiority over a linear model with comparable input information. The 64×64 input is an 8× pixel repetition of the 8×8 image, and the 8×8 logistic baseline has the same TEST accuracy (0.9528). The McNemar test covers seed 0 only.
2. **Boundary selection.** β = 100 is the largest value in the pre-registered grid {20, 50, 100}, and C = 10 is the largest value in the logistic grid {0.01, 0.1, 1, 10}. The grids were not extended, so the selected settings sit at their edges, and better settings may lie outside them. All choices were made on DEV only.
3. **Grid (N = 128).** This is the grid that CONTRATO-004 T2 flags: a window-size effect of about 4 % at 18 mm (relative L2 4.4 × 10⁻², threshold 2 × 10⁻²). The N = 256 robustness check of the final configuration is pending as CONTRATO-005 and is not reported here. Until it is done, D2NN numbers from this record should be read with that caveat.
4. **Aggregated Wilson interval.** The summary's `test_wilson_mean_seeds` (0.9358 to 0.9767) treats the seed-mean accuracy as one binomial proportion over 360 test images. That is mis-specified: the five seeds share one TEST set, and the variation comes from training runs, not from sampling test items. It is not a valid interval for the mean and is not reported as one; the per-seed intervals above are the intervals reported. The figure `docs/img/d2nn_results.gif` still displays the mis-specified interval, and the figure generator needs a correction.

**Note (2026-10-10).** The audit found that `experiments/run_d2nn_digits.py` was committed (af52aff) about nine minutes after its run started (log timestamp 23:02:43). Git history therefore cannot show that the committed script is the version that produced that run. This note changes no threshold and reports no result.
