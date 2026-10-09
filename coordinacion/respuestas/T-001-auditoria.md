# Respuesta a T-001: auditoría independiente del núcleo y del add-on

**Revisor:** agente de revisión independiente (lectura + ejecución de comprobaciones en `D:/PROJECTS/.cognition/neuro3d-lab/review/`).
**Fecha:** 2026-10-09, tarde UTC. **Commit revisado:** `f28ecb9`. **Pruebas:** 22 passed.
**Estado de corrección:** pendiente de asignar (ver `tareas/T-002-correccion-auditoria.md`).

## ALTA (reproducidas)

1. `blender/neuro3d_lab/bench_io.py:122,125`: los mapas de altura no se empaquetan en el `.blend`. Tras guardar y reabrir, `LAYER_1_height` queda a 0 en todos los píxeles.
2. `blender/neuro3d_lab/bench_io.py:239,251`: `trace_optical_path` usa `Vector` en float32 con `eps = 1e-7`. Con `scale_length` = 0,001 el rayo se queda en LAYER_1 y nunca llega a LAYER_2 ni al detector. Con z ≥ 2 BU falla igualmente.
3. `src/optlab/bench.py:109,116`: `n_medium` solo entra en la fase del elemento; la propagación siempre usa vacío. Con `n_medium = n_material = 1,5` el resultado es idéntico al de `n_medium = 1,0`.

## MEDIA (reproducidas)

4. `src/optlab/fresnel.py:31,38`: `r_p` tiene el signo opuesto al de `docs/MODEL.md §4`. R y T no cambian, pero `test_normal_matches` depende del signo del código.
5. `docs/MODEL.md §6`: la fórmula del gradiente de fase (`−2·Im(ḡ·a)`) no conjuga. El código (`d2nn.py:70`) conjuga y la comparación con diferencias finitas lo confirma. Aplicar el documento tal como está da un error del 199 %.
6. `src/optlab/interferometer.py:26`: `is_unitary` usa `np.allclose` con `rtol = 1e-5`; un divisor escalado por 1,0000005 pasa, aunque el contrato exige 1e-12.
7. `tests/test_bench.py:35`: `test_roundtrip` pasa aunque las capas no hagan nada (se verificó parcheando `phase_element` para devolver 1).
8. `operators.py:383-384` y `bench_io.py:165`: las propiedades son float32. El JSON exporta λ = 6,330000132948044e-7, `pixel_m` = 7,99999998e-6 y z = 0,039999999 en lugar de 0,04.
9. `bench_io.py:119-125` y `operators.py:277`: pulsar «Build» dos veces duplica objetos (SOURCE.001…), borra la imagen de LAYER_1 y «Export» falla.
10. `blender/selftest/run_selftest.py:515`: la prueba de orientación solo verifica ida y vuelta con los mismos dos `flipud`, así que pasaría sin ellos. La convención real es correcta (numpy[1,2] → fila 6 desde abajo de la imagen), pero no está comprobada.

## BAJA

- `bench_io.py:152,165`: se usa `location.z` y no `matrix_world`; con el detector emparentado se exporta z erróneo.
- Validación ausente: `height_max_m < 0`, `w_m ≤ 0`, alturas NaN, distancias ≤ 0, Fresnel con θ > 90°, `fit` con lote vacío, etiquetas −1 que se tratan como la última clase en silencio, clave `output.intensity_npy` sin uso.
- `src/optlab/elements.py:27`: con n impar y región de ancho par, la región queda desplazada 0,5 px.
- `operators.py`: acepta `pixel_m < 0` y `wavelength = 0`; rutas fijas en D:; la vista usa AgX, que altera el gris de la intensidad.
- Pruebas débiles: `test_tir` compara constantes exactas; `test_energy` solo prueba n1 < n2; el test de gradiente elige las 200 mayores componentes en lugar de al azar, como pide el contrato; ninguna prueba ejercita el filtrado de ondas evanescentes en el adjunto.
- CONTRATO-001: el registro de la auditoría seguía «pendiente».

## Sin hallazgos

- La convención `exp(−iωt)` es coherente en propagación, `phase_element`, κ y la fase de camino óptico.
- El gradiente del D2NN y el adjunto de `propagation.py` son correctos.
- Fresnel: R, T, TIR y Brewster son correctos. MZI correcto.
- La orientación de coordenadas es coherente entre `detector_masks`, `bench.py` y `bench_io.py`.

## Lectura del coordinador

Los hallazgos 1–3 son fallos reales que el self-test no detectaba: el self-test no comprueba alturas tras reabrir, ni escala distinta de 1, ni `n_medium` distinto de 1. El 5 es un error de documentación; el código ya era correcto. El 4 es una cuestión de convención que hay que fijar por escrito. Ninguno de estos cambios afecta al experimento de dígitos en curso (CONTRATO-003), que usa `d2nn.py`, `propagation.py`, `elements.py` y `fresnel.py` sin cambios de código.
