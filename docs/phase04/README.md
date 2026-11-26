# Fase 4 — consejo multiagéntico (0.4.0)

Se instancian 160 evaluadores locales, uno por cada rol original, agrupados en los 16 departamentos del catálogo. Cada rol consume evidencias estructuradas dirigidas a su ID, evalúa opciones, emite propuestas mediante EventBus y se abstiene si no tiene observaciones. El coordinador utiliza fuentes deduplicadas, utilidad ponderada, riesgo máximo y umbrales explícitos; bloquea opciones con contradicciones fuertes, solicita corroboración y no recomienda en caso de empate o evidencia insuficiente. Una ronda posterior permite reevaluar un contexto nuevo. El contador de rondas es volátil, no memoria cognitiva persistente. Se utiliza concurrencia local por hilos; no Ray ni LangGraph.

Cada rol tiene un conjunto de candidatos de los subsistemas pertinentes de ExpertBank; la unión cubre las 824 redes. Puede solicitar inferencia diagnóstica con un vector de 256 valores, pero su salida no interviene en las puntuaciones porque los pesos y el enrutador no están entrenados. Los agentes comparten el algoritmo de evaluación; todavía no generan conocimientos específicos de su especialidad ni evidencias a partir de imágenes. La confianza, utilidad y riesgo iniciales son entradas proporcionadas por una fuente, no estimaciones calibradas por IA. La deduplicación depende de identificadores de fuente fiables; no verifica independencia o autenticidad real. La deliberación es de una ronda con coordinación central, no conversación autónoma entre pares. No hay ejecución de acciones ni adaptación al cliente de juego.

## Ejecución

Instalar los extras originales: pip install -e ".[neural,vision,dev]" y Tesseract para ejecutar la regresión completa. Desde la raíz:

- python -m aion.agents_cli docs/phase04/example.json
- python -m aion.agents_cli docs/phase04/example.json --neural-agent A001
- python -m pytest -q

El ejemplo es un diagnóstico sintético, no datos de una partida. La opción review tiene evidencia contradictoria, hold tiene riesgo elevado y defer es la recomendación diagnóstica. Las recomendaciones son texto, nunca acciones ejecutables.

## Verificación

33 pruebas aprobadas en CPU Linux, incluidas 27 anteriores. Las 6 nuevas prueban 160 mensajes y propuestas, cobertura de 824 candidatos, abstenciones y rondas, contradicciones, riesgo, deduplicación de fuentes, empate, validación de datos e inferencia real de un experto a tamaño completo. No se verifica calidad estratégica ni aprendizaje. Véase pytest.txt.

## Pendientes preservados

Fase 3 sigue parcial: RT-DETR, Grounding DINO, Qwen3-VL, OmniParser V2, SAM2, ByteTrack y PaddleOCR requieren ejecución real con dependencias y pesos cuando corresponda. UI-TARS necesita verificación de compatibilidad. Faltan integración semántica de máscaras y trayectorias, reconocimiento específico de partidas autorizadas y evaluación temporal. No se añaden parámetros neuronales en Fase 4. Las 32 redes intermedias y 2 finales, aprendizaje continuo, memoria permanente y especialización de roles siguen pendientes. No se afirma que los 160 evaluadores sean 160 expertos entrenados.
