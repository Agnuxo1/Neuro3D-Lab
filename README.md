# Neuro3D-Lab

Laboratorio virtual en Blender para experimentos de redes neuronales ópticas con física óptica real.

La idea: Blender es el banco de laboratorio (geometría, planos, alturas de fase, visualización) y un núcleo numérico propio calcula el campo coherente. Así la fase de cada trayecto sale de la geometría y de la física, no de un modelo de juguete.

> **Estado: versión 0.1 en desarrollo.** El núcleo físico y el add-on se verifican con pruebas ejecutables. No hay resultados experimentales ni dispositivos fabricados. Ver [docs/MODEL.md §8](docs/MODEL.md) para las limitaciones.

## Qué incluye

- **Propagación por espectro angular** (difracción libre exacta sin aproximación paraxial), con adjunto para retropropagación.
- **Elementos de fase** a partir de mapas de altura, con absorción de índice complejo.
- **Interfaces de Fresnel** normales y oblicuas, con reflexión total interna.
- **Interferómetro Mach-Zehnder** unitario.
- **Red difractiva (D2NN)** con gradiente analítico verificado por diferencias finitas y optimización Adam.
- **Add-on de Blender** para construir un banco demo, exportarlo a `optlab.bench/1`, simular e importar la intensidad de detector como imagen.
- **Contratos de validación** escritos antes de medir, con umbrales fijados de antemano ([docs/CONTRACTS/](docs/CONTRACTS/)).

## Inicio rápido

Pruebas del núcleo (Python 3.13, numpy, scipy, pytest):

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Self-test del add-on en Blender headless:

```bash
"D:/TOOLS/Blender/blender-4.5.14-windows-x64/blender.exe" -b --factory-startup -P blender/selftest/run_selftest.py
```

## Estructura

```
src/optlab/         núcleo físico (numpy)
tests/              pruebas del núcleo
blender/neuro3d_lab add-on de Blender
blender/selftest/   prueba headless del add-on
docs/               modelo, arquitectura y contratos
```

## De dónde viene cada pieza

Este laboratorio junta trabajo previo de cuatro repositorios de Agnuxo1:

| Repositorio | Qué aporta aquí |
|---|---|
| [Neuro3D](https://github.com/Agnuxo1/Neuro3D) | Convención de campo coherente `exp(-i·ω·t)`, fase `2π·L/λ`, contrato de unidades, oráculos independientes, complemento de Blender |
| [Silice-Neuro3D-Cube](https://github.com/Agnuxo1/Silice-Neuro3D-Cube) | Propagación BPM split-step, pruebas de conservación de potencia, contratos previos con umbrales no relajables, publicación de fallos |
| [NeuroPixel](https://github.com/Agnuxo1/NeuroPixel) | Controles anti-atajo, intervalo de Wilson, separación DEV/TEST sellada, registro de ejecuciones en JSONL |
| [OpenCLAW-P2P](https://github.com/Agnuxo1/OpenCLAW-P2P) | Convenciones de documentación, plantillas de reproducibilidad y citación |

Ninguna fórmula ni código se copia sin revisar. Cuando una pieza de referencia tiene límites (por ejemplo, el modelo escalar de Neuro3D o el BPM de Silice con pérdida de material fijada a cero en el núcleo), aquí se documenta el límite.

## Documentación

- [docs/MODEL.md](docs/MODEL.md): convenciones y ecuaciones.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): módulos y formato del banco.
- [docs/CONTRACTS/](docs/CONTRACTS/): contratos de validación.

## Licencia

Sin licencia por ahora. Repositorio privado hasta que el autor decida publicarlo y elegir licencia.

## Autor

Francisco Angulo de Lafuente (Agnuxo1).
