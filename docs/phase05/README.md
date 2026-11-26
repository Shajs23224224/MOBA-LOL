# Fase 5 — memoria y creencias (0.5.0)

Ocho categorías de memoria en SQLite WAL, con separación por scope, vectores normalizados de 256 características, recuperación exacta por similitud y límite FIFO de 256 registros por categoría y scope por defecto (2.048 registros por scope). El máximo configurable es 4.096 por categoría. No existe todavía un límite global de scopes ni compactación del archivo SQLite. Los ocho tipos comparten almacenamiento y recuperación; no son ocho modelos neuronales ni sistemas de aprendizaje independientes. La similitud no implica verdad ni conocimiento semántico.

NumPy está probado. Si FAISS está instalado, la recuperación usa IndexFlatIP reconstruido a partir de SQLite por consulta; no se ha ejecutado ese backend. PyTorch conecta el vector visual con la red seleccionada por un agente. El filtro binario bayesiano utiliza un prior de 0.5, fiabilidad explícita suministrada y decaimiento hacia 0.5 (retención 0.95 por tick). No es un modelo neuronal del mundo, no rastrea posiciones ni predice eventos complejos. Los valores son hipótesis condicionales, no probabilidades calibradas. La ausencia de señal no equivale a una observación negativa. Las actualizaciones repetidas dependen de una hipótesis de independencia condicional que no se ha validado. El estado se conserva en SQLite y rechaza ticks repetidos o regresivos.

Cognition.ingest procesa imágenes locales, recupera memoria previa y consulta al consejo de 160 agentes con contexto suministrado, permite diagnóstico neuronal optativo, actualiza creencias con señales explícitas y registra seis tipos automáticamente. Semántica y procedimental admiten escritura explícita mediante Memory.put, sin consolidación aprendida. El consejo aún no utiliza las memorias recuperadas para modificar sus puntuaciones; se almacenan referencias y se devuelve el resultado al consumidor. El OCR no se convierte en hechos por sí solo. La salida neuronal no determina las creencias ni las decisiones. No hay acciones de juego. La actualización del mundo es transaccional, pero la ingestión completa usa varias transacciones y puede quedar parcial ante fallo; se requiere un escritor por scope y recuperación de ciclos incompletos en una fase posterior.

## Reproducción

Instalar con pip install -e ".[neural,vision,dev,memory]" y Tesseract para toda la regresión. Extra memory-faiss opcional. Ejemplo Python desde la raíz:

    import numpy as np
    from aion.cognition import Cognition
    from aion.agents import Context
    from aion.world import Signal

    c = Cognition("data/cognition.sqlite")
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    ctx = Context(id="diagnostic", options=("review",))
    result = c.ingest("sample", 0, image, ctx, signals=[Signal(key="marker", present=True, reliability=.9)], ocr=False)
    print(result["world"])
    # Para repetir, usar un tick estrictamente mayor o un scope nuevo.

## Verificación

38 pruebas aprobadas en Linux Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5. Cinco nuevas: ocho tipos y persistencia/reapertura, ranking y capacidad, fallback NumPy y error explícito de FAISS ausente, actualización bayesiana y aislamiento, integración imagen-consejo-red con recuperación tras reapertura. Véase pytest.txt. Sin entrenamiento ni evaluación sobre partidas. No se validó Windows, CUDA, FAISS ni concurrencia de escritores en el mismo scope.

Se mantienen los pendientes de Fases 2, 3 y 4, especialmente materialización/entrenamiento completo, modelos externos y especialización real de los agentes. No se añaden parámetros neuronales. Fase 5 es un prototipo parcial; falta consolidación aprendida, memoria procedimental adquirida, uso de recuerdos por agentes y modelo predictivo neuronal.
