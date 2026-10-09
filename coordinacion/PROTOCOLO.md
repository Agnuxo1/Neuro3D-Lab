# Protocolo Codex–Claude de Neuro3D-Lab

Adaptado del protocolo de Neuro3D (`9_NEBULA_NEW/coordinacion/PROTOCOLO.md`).

1. Antes de trabajar, leer `CHECKPOINT.md`, `TABLON.md` y `DECISIONES.md`.
2. Cada encargo es un archivo `tareas/<ID>.md` con contexto, alcance, archivos permitidos, criterios de aceptación y entregable. Dos tareas activas no pueden tocar el mismo archivo.
3. El agente que ejecuta entrega `respuestas/<ID>.md` (o `.json`). Una auditoría solo lee; una edición necesita archivos asignados.
4. Quien pide la tarea comprueba el entregable con las pruebas o los datos, decide, y registra la decisión en `DECISIONES.md`.
5. Discrepancias: `TABLON.md` como hechos, inferencias y propuestas. El acuerdo entre agentes no es validación; la validación es el contrato con sus umbrales fijados antes de medir.
6. Las pruebas de hipótesis se preregistran en `docs/CONTRACTS/` antes de ejecutarlas. Los umbrales no se relajan después. Los fallos se publican con su valor.
7. Resultados grandes y crudos van a `D:/PROJECTS/.cognition/neuro3d-lab/`, no al repo. Al repo solo llegan resúmenes.
8. Trabajo GPU: solo por `gpuq.py run` y con autorización. Sin GPU, todo lo de esta carpeta corre en CPU.
