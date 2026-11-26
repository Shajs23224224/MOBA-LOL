# Fase 7 — evaluación MOBA de investigación (0.7.0)

Evaluador offline de estados explícitos y autorizados. Snapshot recibe salud, recursos, oro, amenaza, apoyo aliado, valor de objetivos y recursos, acceso a tienda, huecos libres, confianza, hasta cuatro objetos y cuatro rutas. Las etiquetas tácticas se deducen con reglas Horn. El evaluador compara mantener, recuperar, obtener recursos, objetivos, cooperación, compras asequibles y rutas suministradas. Valida límites y excluye compras sin oro, espacio o acceso. Utilidades heurísticas ponderan beneficio inmediato (1), beneficio futuro (1.5) y penalización por coste y riesgo (2). Las probabilidades son supuestos, no calibraciones. Se reduce la certidumbre hacia 0.5 cuando baja la confianza. No conoce catálogo real de campeones, equipamiento o mapas; no calcula rutas geométricas, combos, tiempos ni transiciones multietapa. El modelo tiene un paso con dos resultados estocásticos por alternativa; utiliza el cálculo exacto de utilidad esperada y el buscador concurrente UCT de Fase 6 con 256 muestras por opción. No se automatizan compras, movimiento ni otras acciones.

review integra Cognition y analyze: procesa una imagen local, llama al consejo de 160 agentes sobre evidencias explícitas, permite diagnóstico opcional de una red, conserva creencias suministradas y recupera las ocho categorías de memoria por separado. Las estimaciones de planificación no se convierten en evidencias observadas. La coincidencia de etiquetas no prueba verdad; las memorias recuperadas, la visión y las salidas neuronales no generan el Snapshot ni modifican las puntuaciones. Los pesos neuronales y agentes siguen sin especialización aprendida. La planificación se persiste como pronóstico no observado. No hay control del cliente ni conexión a partidas.

## Uso

Instalar las dependencias anteriores con pip install -e ".[neural,vision,memory,dev,reasoning-jax]" y Tesseract. API:

    from aion.moba import Snapshot, evaluate
    s = Snapshot(health=.1, resource=.2, gold=100, threat=.9,
                 ally_support=0, objective_value=.9, farm_value=.4,
                 shop_access=True, free_slots=1, confidence=.95)
    rows, model, facts, trace = evaluate(s)
    print(sorted(rows,key=lambda row:-row['expected_utility']))

Para el flujo integrado: review(cognition, scope, tick, image, snapshot, evidence=..., signals=..., neural_agent=..., ocr=False). Los parámetros de evidencia y señales siguen los contratos de las fases 4 y 5. Los ticks deben aumentar. Se recomienda utilizar imágenes locales sintéticas o autorizadas y datos estructurados del entorno propio.

## Verificación

49 pruebas locales aprobadas en 15.82 segundos (45 anteriores y 4 nuevas). Se comprobó presupuesto, inventario y acceso a tienda, comparación de rutas, recursos y amenaza, cooperación, validación de entradas, incertidumbre, 160 evaluadores, diagnóstico neuronal real y recuperación de las ocho memorias con datos de prueba. El escenario sintético de smoke.json contiene 8 opciones; Monte Carlo ejecutó 256 muestras por opción y coincidió con la utilidad exacta al priorizar recover. No se mide rendimiento de juego, éxito de compras, precisión de percepción ni estrategia aprendida.

## Pendientes

Se conservan las tareas pendientes de fases 1–6: entrenamiento y materialización de expertos, redes intermedias y finales, modelos externos y sus pesos, especialización de agentes, FAISS, consolidación de memoria y modelo aprendido del mundo, mctx, razonamiento neuro-simbólico aprendido, distribución y validación Windows/CUDA. Fase 7 es parcial: faltan adaptación aprendida, integración perceptiva automática, mecánicas y datos reales de un entorno autorizado, y planificación multietapa específica. No se añaden parámetros neuronales.
