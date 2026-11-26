# AION — Fases 1–3 / 0.3.1

Infraestructura ejecutable de investigación. **No es todavía un agente de League of Legends, ni un modelo entrenado.** Esta versión sólo ejecuta diagnóstico y servicios internos. No captura pantalla, no controla juegos, no descarga modelos y no contiene un entrenador ni simulador.

## Inicio (Windows, PowerShell)

Se recomienda Python 3.11 o 3.12 de 64 bits. Desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[neural,dev]"
.\.venv\Scripts\python.exe -m aion --config config/default.toml doctor
.\.venv\Scripts\python.exe -m aion --config config/default.toml neural-probe
.\.venv\Scripts\python.exe -m aion --config config/default.toml run --seconds 3
.\.venv\Scripts\python.exe -m aion --config config/default.toml status
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS: crear el entorno con python3 -m venv .venv y utilizar .venv/bin/python. Requiere internet para instalar dependencias si no están disponibles. Para el núcleo sin prueba neuronal: pip install -e ".[dev]". PyTorch con GPU requiere la distribución apropiada para el hardware; consultar https://pytorch.org/get-started/locally/ . No se probó Windows, CUDA ni macOS en esta fase.

## Comandos

- doctor: detección real de CPU, RAM, PyTorch y dispositivos CUDA visibles.
- architecture --output data/architecture.json: manifiesto de la arquitectura objetivo, no un inventario de pesos existentes.
- neural-probe: una red pequeña real de 212 parámetros, ejecución y round-trip safetensors con hash. Excluida de las 858 redes; no entrenada.
- run --seconds 3: iniciar núcleo, reservar presupuesto, persistir manifiesto, emitir eventos y detener; Ctrl+C también detiene durante la espera.
- status: sesiones y eventos persistidos.
- models: auditar hashes de los pesos registrados localmente (vacío inicialmente).

## Estructura

aion/catalog.py: catálogo; contracts.py: mensajes tipados; bus.py: mensajería local concurrente acotada; resources.py: detección y presupuesto cooperativo; storage.py: persistencia SQLite; models.py: registro local y carga safetensors; runtime.py: ciclo de vida; probe.py: prueba de inferencia; cli.py: interfaz de diagnóstico. config/ contiene configuración y catálogo visual. tests/ contiene pruebas. docs/ contiene la hoja de ruta y evidencia de verificación.

## Decisiones y límites

- 824 redes principales, 32 intermedias, 2 finales y 160 roles son requisitos catalogados, no instancias. Se mantienen los 16×2048 por red principal, 27.000.832 unidades ocultas y 51.841.597.440 pesos densos entre capas ocultas (sin contar entradas, cabezas ni sesgos).
- Se eligió Python y ejecución local mediante hilos, con contratos extensibles. Ray, C++ y ejecución distribuida quedan pendientes, no incluidos como servicios simulados. No se han implementado los modelos visuales externos. Los catálogos son referencias del PDF; comprobar licencias y versiones antes de integrarlos.
- El presupuesto de RAM es cooperativo, no impone límites al proceso ni reserva VRAM. El diagnóstico consulta VRAM visible, sin probar ejecución CUDA. Registro de modelos diseñado para un escritor; no utilizarlo concurrentemente desde varios procesos. La verificación SHA-256 comprueba integridad contra un hash registrado, no autenticidad ni licencia. Un cierre abrupto puede dejar sesiones marcadas como running; no hay recuperación automática aún.
- Los contratos no habilitan acciones sobre juegos. El único entorno aceptado por configuración es diagnostics. La integración futura requiere un entorno propio o autorización expresa del titular. No se incluye código de evasión, inyección o automatización del cliente oficial.
- No hay instalador ni build final todavía. La fase 2 debe crear redes reales y medir parámetros únicos, empezando por un subconjunto explícito según presupuesto. Nunca contar un catálogo, tensores meta ni la prueba diagnóstica como redes entrenadas.
- El PDF maestro original se conserva en el ZIP de la fase 1, fuera de este repositorio. El manifiesto puede regenerarse con python -m aion architecture --output docs/architecture.json.

## Fase 2

Se añade el banco neuronal de 824 expertos con estructuras meta, materialización bajo demanda y enrutamiento top-k. Consulte [informe de Fase 2](docs/phase02/README.md) para ejecución, pruebas, parámetros y límites. El núcleo original de Fase 1 permanece disponible.

## Fase 3 (parcial)

Canal de visión para imágenes locales, OpenCV, OCR Tesseract y puente hacia los expertos neuronales. Los adaptadores de modelos externos todavía requieren pesos y validación. Consulte [informe de Fase 3](docs/phase03/README.md). Las fases anteriores se conservan.

## Fase 3B

Cuatro adaptadores adicionales (OmniParser V2, SAM2, ByteTrack, PaddleOCR), con contratos probados pero pesos y backends externos pendientes. Consulte [informe de Fase 3B](docs/phase03b/README.md).
