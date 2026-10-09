# T-002: corrección de los hallazgos de la auditoría T-001

**Estado:** abierta. **Pide:** Claude (coordinador). **Ejecuta:** cog-worker.
**Entregable:** cambios en el repo + `coordinacion/respuestas/T-002-correccion.md` con una línea por hallazgo: `arreglado | documentado | pendiente`, y la salida de pytest.

**Alcance permitido:** `src/optlab/bench.py`, `src/optlab/interferometer.py`, `blender/neuro3d_lab/*.py`, `blender/selftest/run_selftest.py`, `tests/test_bench.py`, `tests/test_interferometer.py`, `tests/test_fresnel.py`, `docs/MODEL.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS/CONTRATO-001-nucleo.md` (solo sección de registro).

**Prohibido** (el experimento CONTRATO-003 está en marcha y usa estos archivos sin cambios de código): `src/optlab/d2nn.py`, `src/optlab/propagation.py`, `src/optlab/elements.py`, `src/optlab/fresnel.py` (solo documentación de convención en MODEL.md), `experiments/`, `docs/CONTRACTS/CONTRATO-002*`, `docs/CONTRACTS/CONTRATO-003*`.

**Hallazgos a corregir (ver `respuestas/T-001-auditoria.md`):**
1. Empaquetar las imágenes de altura al crearlas (`img.pack()`), para que sobrevivan a guardar y reabrir el `.blend`.
2. `trace_optical_path`: épsilon de avance en unidades de Blender compatible con `scale_length` y sin depender de float32 (p. ej. avanzar `1e-4` BU o un fracción del paso de escala). Probar con scale 1,0, 0,001 y 0,01 y con z hasta 2 BU.
3. `n_medium`: añadir `medium_index` (por defecto 1,0) a nivel de banco. La propagación usa longitud de onda `λ / medium_index`. Validar que todos los `n_medium` de las capas coinciden con `medium_index` (ValueError si no). Actualizar `docs/ARCHITECTURE.md`.
4. Fresnel: documentar en MODEL.md la convención de signo de `r_p` que usa el código (`r_p = r_s` en incidencia normal). No cambiar el código.
5. MODEL.md §6: corregir la fórmula del gradiente a `∂L/∂φ_k = −2·Im(conj(ḡ_{a_k})·a_k)`, como en el código.
6. `is_unitary`: comparar `max|U†U − I| ≤ atol` sin `rtol`. Añadir prueba con divisor escalado por 1+5e-7 que debe fallar.
7. `test_roundtrip`: exigir que una capa con altura aleatoria cambie el resultado frente a sin capas (diferencia > 1e-3 relativa). Mantener la conservación.
8. Exportar λ, pixel y z con `float(f"{v:.6g}")` para eliminar el ruido de float32.
9. «Build» idempotente: borrar objetos previos con `optlab_role` antes de crear, y reempaquetar imágenes.
10. Self-test de orientación: comprobar un píxel conocido (imagen con un único píxel marcado) de extremo a extremo, no solo ida y vuelta.
11. Baja prioridad (solo si queda tiempo): `matrix_world` en lugar de `location`, validaciones de entrada listadas en T-001 y prueba de evanescentes.

**Criterios de aceptación:** `python -m pytest -q` en verde; `blender.exe -b --factory-startup -P blender/selftest/run_selftest.py` con PASS en los ocho pasos; cada cambio con una prueba que habría fallado antes.
