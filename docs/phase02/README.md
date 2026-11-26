# Fase 2 — arquitectura neuronal (0.2.0)

Implementación PyTorch de 824 expertos independientes de 16 capas ocultas de 2048 unidades, GELU, LayerNorm, conexiones residuales y cabeza de 256 salidas. Se eligió entrada y salida de 256 características como contrato provisional. No son las 32 redes intermedias ni las 2 finales (fase 9). No hay pesos preentrenados propios.

Cada experto tiene 64.061.696 parámetros; total experto 52.786.837.504; enrutador 211.768; total lógico 52.787.049.272. Los módulos meta verifican dimensiones y conteos, no almacenan pesos. En ejecución se materializa un experto por defecto mediante caché LRU, con inicialización determinista o checkpoint local verificado de Fase 1. Los expertos son diferenciables individualmente, pero el enrutador no está entrenado y la inferencia top-k no constituye un bucle de entrenamiento. Los 16 subsistemas siguen siendo etiquetas del catálogo, no especializaciones aprendidas. Sin cuantización, atención, distribución Ray ni aprendizaje todavía. El límite de caché controla bytes de pesos de expertos, no memoria total del proceso, activaciones o router. El preflight es cooperativo. Las redes se ejecutan secuencialmente, no de forma simultánea. No se garantiza latencia de juego. No se ha validado CUDA, integración visual, carga de checkpoint de un experto completo, ni funcionamiento en Windows.

## Reproducción

Instalar con pip install -e ".[neural,dev]".

- python -m aion.neural_cli --config config/default.toml : auditoría de los 824 módulos meta.
- python -m aion.neural_cli --config config/default.toml --infer : inferencia CPU con vector diagnóstico, no datos de juego.
- python -m pytest -q : pruebas (requiere PyTorch y safetensors).

Se ejecutaron localmente 17 pruebas en Linux/PyTorch 2.10.0+cpu (13 originales y 4 nuevas); ver pytest.txt y audit.json. Se verificaron formas, unicidad, cuenta exacta, inferencia a dimensión completa de N001 y N824, agregación, expulsión y recarga determinista, rechazo de entradas y presupuesto, y backward de un experto completo. El CLI seleccionó N684 y produjo una salida finita. La prueba no entrena ningún experto ni verifica todos mediante inferencia. Las herramientas originales de Fase 1 permanecen disponibles y siguen reportando el catálogo sin materialización persistente. Fase 3 pendiente.
