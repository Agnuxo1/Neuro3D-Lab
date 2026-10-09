<p align="center">
  <img src="docs/img/header.gif" alt="Neuro3D-Lab: propagación de un haz gaussiano y anchura numérica frente a analítica" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="Licencia Apache-2.0" src="https://img.shields.io/badge/licencia-Apache--2.0-22d3ee"></a>
  <img alt="Python 3.13" src="https://img.shields.io/badge/python-3.13-a78bfa">
  <img alt="Blender 4.5.14" src="https://img.shields.io/badge/Blender-4.5.14-fbbf24">
  <img alt="Pruebas" src="https://img.shields.io/badge/pruebas-29%20passed-34d399">
  <img alt="Estado" src="https://img.shields.io/badge/estado-v0.1%20en%20validación-f87171">
</p>

<p align="center"><b>Un laboratorio óptico virtual: Blender es el banco, la física la calcula un núcleo verificado.</b></p>

---

## ¿Qué es?

Neuro3D-Lab permite diseñar y simular **redes neuronales ópticas** en un banco virtual de Blender. La geometría (fuente, planos de fase, detector) vive en la escena de Blender. El campo óptico coherente lo calcula `optlab`, un núcleo en numpy que implementa difracción, interferencia y reflexión con fórmulas comprobables.

Cada resultado tiene un **contrato preregistrado**: los umbrales se fijan antes de medir y no se relajan después. Los fallos también se publican.

<p align="center">
  <img src="docs/img/pipeline.gif" alt="Flujo: Blender exporta el banco, optlab calcula el campo, el resultado vuelve a Blender" width="85%">
</p>

El banco funciona en tres pasos: Blender exporta la geometría y las alturas de fase (`optlab.bench/1`), `optlab` propaga el campo y devuelve la intensidad del detector, y Blender la muestra como una imagen emisiva.

---

## Física implementada

### Propagación por espectro angular

Difracción libre sin aproximación paraxial. El campo se propaga en el dominio de frecuencias y se vuelve al espacio con una función de transferencia exacta para ondas propagantes.

$$U = \mathcal{F}^{-1}\{H\cdot\mathcal{F}\{u_0\}\},\qquad H = \exp\!\left(i\,2\pi d\sqrt{1/\lambda^2 - f_x^2 - f_y^2}\right)$$

<p align="center">
  <img src="docs/img/propagation.gif" alt="Haz gaussiano propagándose; la anchura numérica coincide con la analítica" width="90%">
</p>

**Validación (CONTRATO-002, E1):** el haz gaussiano coincide con la solución exacta de Gouy. El error relativo de la anchura es de 4,1·10⁻⁵ frente a un umbral de 5·10⁻³. La fase de Gouy coincide dentro de 1,2·10⁻⁵ rad.

### Interfaces de Fresnel

Reflectancia y transmitancia para incidencia normal y oblicua, con reflexión total interna y ángulo de Brewster. Se comprueba la conservación de energía R + T = 1.

<p align="center">
  <img src="docs/img/fresnel.gif" alt="Reflectancia de Fresnel en aire-vidrio con el ángulo de Brewster" width="90%">
</p>

### Interferómetro Mach-Zehnder

Matriz unitaria de dos divisores 50/50 y un desfasador. Con la entrada en el puerto 1, la potencia de salida sigue $\sin^2(\theta/2)$, y la matriz es unitaria dentro de 10⁻¹² en todo el rango.

<p align="center">
  <img src="docs/img/mzi.gif" alt="Reparto de potencia de un interferómetro Mach-Zehnder al variar la fase" width="90%">
</p>

### Autoimagen de Talbot

La red de amplitud se reproduce sin lente en $z_T = 2p^2/\lambda$. En $z_T$ el patrón coincide con la entrada; en $z_T/2$ el patrón es distinto (no es una autoimagen).

<p align="center">
  <img src="docs/img/talbot.gif" alt="Autoimagen de Talbot: el perfil de intensidad se repite a z_T" width="90%">
</p>

**Validación (CONTRATO-002, E2):** en $z_T$ el error RMS es de 2·10⁻⁶ frente a un umbral de 10⁻². En $z_T/2$ el patrón es distinto del de entrada (error 1,35, criterio ≥ 0,5).

### Difracción de Fresnel por una apertura circular

Intensidad en el eje de una apertura circular, comparada con la solución de Fresnel $I/I_0 = 4\sin^2(\pi N_F/2)$.

<p align="center">
  <img src="docs/img/aperture.gif" alt="Intensidad en el eje de una apertura circular, numérica frente a analítica" width="90%">
</p>

**Validación (CONTRATO-002, E3):** el error absoluto máximo es de 7·10⁻³ frente a un umbral de 5·10⁻². El radio se toma del área muestreada, no del radio nominal, y eso está declarado en el contrato.

---

## Red difractiva de fase (D2NN)

Tres capas de fase pura, 18 mm entre planos, 128 × 128 píxeles, con un detector de diez regiones. La entrada es un dígito manuscrito normalizado en energía.

<p align="center">
  <img src="docs/img/d2nn_flow.gif" alt="Flujo de un dígito a través de la red: la energía se reparte entre 10 regiones del detector" width="95%">
</p>

> La animación usa fases aleatorias para mostrar el flujo de la luz. No es un resultado entrenado.

### Resultados en dígitos manuscritos (CONTRATO-003)

<p align="center">
  <img src="docs/img/d2nn_results.gif" alt="Precisión en TEST de la D2NN por semilla, bases y controles" width="95%">
</p>

_Resultados pendientes: el experimento está en ejecución. Esta sección se completará con los valores medidos, sin cambiar los umbrales._

---

## Método científico: contratos y veredictos

Cada afirmación del proyecto tiene un contrato con umbrales fijados antes de medir.

| Contrato | Qué prueba | Estado |
|---|---|---|
| [CONTRATO-001](docs/CONTRACTS/CONTRATO-001-nucleo.md) | Pruebas unitarias del núcleo, orientación y self-test de Blender | 29 pruebas en verde; self-test 14/14 |
| [CONTRATO-002](docs/CONTRACTS/CONTRATO-002-validacion-analitica.md) | Propagador frente a soluciones analíticas | E1, E2, E3 cumplen; E4 exploratorio |
| [CONTRATO-003](docs/CONTRACTS/CONTRATO-003-d2nn-digitos.md) | D2NN en dígitos con controles y base de parámetros comparables | En ejecución |

<p align="center">
  <img src="docs/img/verdicts.gif" alt="Veredictos de cada criterio preregistrado" width="90%">
</p>

**Controles de poder del arnés:** propagar con el signo invertido produce errores de fase de Gouy de 0,49 y 1,57 rad, muy por encima del umbral. Eso demuestra que la prueba detecta errores reales.

**Auditoría independiente:** una revisión de código encontró tres fallos de alta severidad y siete de media. Todos están corregidos o documentados en [`coordinacion/respuestas/`](coordinacion/respuestas/). Los dos más importantes eran que `n_medium` no afectaba a la propagación y que las alturas de fase se perdían al guardar el `.blend`.

---

## Usar el laboratorio

Requisitos: Python 3.13 y Blender 4.5.14 (para el add-on).

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Validación analítica (CONTRATO-002, unos 7 s en CPU):

```bash
python experiments/run_validation.py
```

Self-test del add-on en Blender en modo headless:

```bash
"D:/TOOLS/Blender/blender-4.5.14-windows-x64/blender.exe" -b --factory-startup -P blender/selftest/run_selftest.py
```

Regenerar los GIF del README (necesita matplotlib, pillow y scikit-learn):

```bash
python docs/figures/make_figures.py
```

---

## Estructura

```
src/optlab/           núcleo físico en numpy
blender/neuro3d_lab/  add-on de Blender: banco, exportación, importación
blender/selftest/     self-test headless del add-on
experiments/          scripts de los contratos
experiments/results/  resúmenes versionados de los contratos
docs/                 modelo físico, arquitectura y contratos
docs/figures/         generador de los GIF del README
coordinacion/         protocolo de trabajo, tablón, tareas y respuestas
tests/                pruebas del núcleo
```

---

## Límites conocidos

- Modelo escalar: sin polarización ni campos vectoriales. Con incidencia oblicua en planos de fase, el modelo escalar no basta.
- Sin reflexiones entre planos y sin dispersión espectral.
- Intensidad relativa, sin calibración en vatios.
- Solo simulación. Ningún resultado aquí es un dispositivo fabricado ni una medida experimental.
- El espectro angular sin banda limitada tiene un límite de validez $z < N\,dx^2/\lambda$. Para campos suaves es conservador; para bordes nítidos más allá del límite, no está validado aún.

---

## Referencias

- Lin, X. et al. *All-optical machine learning using diffractive deep neural networks.* Science (2018). [doi:10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084) · [arXiv:1804.08711](https://arxiv.org/abs/1804.08711)
- Matsushima, K. & Shimobaba, T. *Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields.* Optics Express 17, 19662 (2009).
- Clements, W. R. et al. *An optimal design for universal multiport interferometers.* Optica 3, 1460 (2016). [doi:10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460)

Trabajo previo del autor que se integra aquí: [Neuro3D](https://github.com/Agnuxo1/Neuro3D), [Silice-Neuro3D-Cube](https://github.com/Agnuxo1/Silice-Neuro3D-Cube), [NeuroPixel](https://github.com/Agnuxo1/NeuroPixel) y [OpenCLAW-P2P](https://github.com/Agnuxo1/OpenCLAW-P2P).

---

## Licencia

Apache License 2.0. Ver [LICENSE](LICENSE) y [NOTICE](NOTICE). El add-on se ejecuta dentro de Blender, que se distribuye bajo GPL-3.0-or-later; Blender no se redistribuye en este repositorio.

Autor: Francisco Angulo de Lafuente ([Agnuxo1](https://github.com/Agnuxo1)).
