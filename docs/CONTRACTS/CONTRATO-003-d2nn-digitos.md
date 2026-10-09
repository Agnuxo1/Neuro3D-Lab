# CONTRATO-003: red difractiva D2NN de fase en dígitos manuscritos

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

_Se completa tras ejecutar el experimento. Cada resultado: fecha UTC, commit, valores, veredicto por hipótesis y control._
