# T-001: auditoría independiente del núcleo y del add-on

**Estado:** abierta.
**Pide:** Claude (coordinador). **Ejecuta:** cualquier agente con acceso al repo (Codex o revisor independiente).
**Alcance (solo lectura):** `src/optlab/*.py`, `blender/neuro3d_lab/*.py`, `blender/selftest/run_selftest.py`, `tests/*.py`.
**Referencia:** `docs/MODEL.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS/CONTRATO-001-nucleo.md`.

**Preguntas:**
1. ¿Las fórmulas de `propagation.py`, `fresnel.py`, `interferometer.py` y el gradiente de `d2nn.py` coinciden con `docs/MODEL.md`?
2. ¿Hay inconsistencias de orientación entre `elements.py`, `bench.py` y `bench_io.py`?
3. ¿Qué casos borde fallan (distancias nulas, rejillas impares, regiones fuera de la rejilla)?
4. ¿Algún test pasa por accidente o con tolerancia demasiado laxa?

**Entregable:** `respuestas/T-001-auditoria.md` con hallazgos ordenados por severidad, `archivo:línea`, y un ejemplo reproducible. No editar código en esta tarea.
**Aceptación:** cada hallazgo reproducido o marcado como no reproducido.
