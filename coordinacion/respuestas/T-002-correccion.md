# Respuesta a T-002: corrección de la auditoría T-001

Estado: pytest 29 passed; self-test de Blender 14/14 PASS (los 8 originales + 6 nuevos), código de salida 0. Sin commit.

| # | Estado | Archivo | Prueba |
|---|---|---|---|
| 1 | arreglado | `blender/neuro3d_lab/bench_io.py` (`img.pack()`) | self-test `pack_roundtrip` (guarda copia, recarga imágenes, compara píxeles) |
| 2 | arreglado | `bench_io.py` (`EPS_BU = 1e-4`) | self-test `trace_scales` (escala 1, 0,01, 0,001; z hasta 6 BU) |
| 3 | arreglado | `src/optlab/bench.py` (`medium_index`, validación), `docs/ARCHITECTURE.md`, `docs/MODEL.md` | `test_medium_index_changes_propagation`, `test_medium_index_mismatch_raises` |
| 4 | documentado | `docs/MODEL.md` §4 (`r_p = r_s` en incidencia normal; fórmula igualada al código) | `test_rp_sign_convention` |
| 5 | arreglado | `docs/MODEL.md` §6 (`conj(ḡ)`) | cubierto por la prueba de gradiente existente (el código no cambió) |
| 6 | arreglado | `src/optlab/interferometer.py` | `test_is_unitary_rejects_small_scaling` |
| 7 | arreglado | `tests/test_bench.py` | `test_roundtrip` (diferencia relativa > 1e-3; comprobado que falla con `phase_element` anulado) |
| 8 | arreglado | `bench_io.py` (`_clean`, `.6g`) | self-test `export_clean_numbers` (camino del operador, propiedades float32) |
| 9 | arreglado | `bench_io.py` (`clear_bench`) | self-test `build_idempotent` (Build x2 + Export) |
| 10 | arreglado | `blender/selftest/run_selftest.py` | self-test `orientation_end_to_end` (píxel único contra datos crudos de Blender, UV y `detector_masks`) |
| 11 | arreglado parcial | `matrix_world` (`parented_detector_z`); validaciones en `bench.py` (`w_m/h_m` > 0, `height_max_m` >= 0, NaN); `pixel_m`/λ con `min` en operators; prueba de evanescentes | `parented_detector_z`, `test_input_validation_extras`, `test_evanescent_components_dropped` |

Pendiente por la lista prohibida (no tocado): Fresnel con θ > 90°, `fit` con lote vacío y etiquetas -1 (`d2nn.py`), desplazamiento de 0,5 px en `elements.py`. Pendiente sin riesgo: AgX en la vista, rutas fijas en D:, `test_tir` con constantes exactas.

Las pruebas nuevas de Python (3, 6) y del self-test (1, 2, 9) se ejecutaron contra el código anterior de HEAD y fallaron.
