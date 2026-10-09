# Tablón de Neuro3D-Lab

Registro de trabajo, no evidencia. Lo más reciente, arriba.

## 2026-10-09 — Claude — hito v0.1 núcleo: revisión, registro y orientación

- Self-test de Blender con `src/optlab` presente: PASS en los ocho pasos.
- Corregida la convención de `+y` en `detector_masks` (fe de erratas en CONTRATO-001). `pytest`: 22 passed.
- Convención de imagen documentada en `docs/ARCHITECTURE.md`.
- Pedido a Codex: tarea `T-001` (auditoría independiente del núcleo y del add-on) en `tareas/T-001-auditoria.md`.

## 2026-10-09 (tarde UTC) — Claude — auditoría T-001 recibida y encargo de corrección

- Auditoría independiente: 3 altas, 7 medias, varias bajas. Informe en `respuestas/T-001-auditoria.md`.
- Abierta `tareas/T-002-correccion-auditoria.md`. No toca los archivos del CONTRATO-003 en curso.
- Licencia elegida: Apache-2.0 (Blender 4.5 declara GPL-3.0-or-later; Apache-2.0 es compatible con GPLv3 y concede licencia explícita de patentes).
