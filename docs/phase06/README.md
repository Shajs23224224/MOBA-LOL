# Fase 6 — razonamiento y planificación (0.6.0)

Prototipo de investigación de sólo lectura sobre modelos finitos explícitos. No contiene política de juego entrenada, captura de pantalla ni ejecución de acciones.

- reasoning.py: clausura de reglas Horn positivas con trazas, abducción de hipótesis mínimas suficientes (hasta 12 candidatas), inducción mediante media posterior Beta(1,1) para observaciones Bernoulli suministradas, modelos causales estructurales booleanos con tablas explícitas, probabilidad exacta, intervenciones y contrafactuales por abducción-intervención-predicción. Hasta 12 raíces y 64 ecuaciones; las hipótesis causales son proporcionadas, no descubiertas.
- planning.py: MCTS/UCT con muestreo de resultados aleatorios, estadísticas por estado y profundidad, Beam Search para modelos deterministas (rechaza resultados estocásticos), expansión jerárquica de tareas explícitas con detección de ciclos y evaluación concurrente de raíces independientes. Suma ponderada de hasta 8 objetivos, no optimización Pareto. Hasta 256 estados, 16 alternativas y 16 resultados por transición, profundidad máxima 16, entre 16 y 1024 iteraciones por búsqueda, hasta 8 trabajadores, haz máximo 64. Los hilos no garantizan aceleración CPU por el GIL. No hay garantía de optimalidad, calibración, ni latencia en tiempo real. La utilidad y las transiciones vienen del usuario, no del visor.
- expected_reward: cálculo NumPy y backend JAX opcional, ambos probados. JAX 0.9.0.1 CPU disponible. mctx no instalado; no se afirma haberlo integrado. La búsqueda UCT es propia, no acelerada con JAX.
- research.py: integra el pipeline Cognition de Fase 5 y compara la alternativa simulada con la recomendación del consejo. Guarda la planificación como experiencia no observada, conservando referencias a recuerdos y creencias. Si no coinciden, marca revisión. Los resultados simulados no se convierten en evidencia observada. Las memorias y creencias todavía no modifican automáticamente las transiciones; el banco neuronal se conserva como diagnóstico opcional mediante neural_agent, sin utilizar pesos no entrenados como heurística. Las reglas causales y lógicas están disponibles por API, no se construyen automáticamente desde píxeles. No constituye un motor neuro-simbólico plenamente aprendido.

## Reproducción

Instalar con pip install -e ".[neural,vision,memory,dev]" y Tesseract para regresión. Extra reasoning-jax opcional.

    from aion.planning import Model, Outcome, mcts, beam, parallel_roots
    m = Model({"s": {"small": [Outcome("end",1,(1.,0.))],
                      "large": [Outcome("next",1,(0.,0.))]},
               "next": {"finish": [Outcome("end",1,(5.,1.))]}}, (1.,.5))
    print(mcts(m,"s"), beam(m,"s"), parallel_roots(m,"s"))

Ejecutar python -m pytest -q. Las 45 pruebas pasaron localmente, incluidas 38 de regresión. Las 7 nuevas cubren lógica y abducción, contrafactuales condicionados, búsqueda determinista y estocástica, límites, cálculo JAX real, descomposición jerárquica e integración Cognition/consejo/memoria/creencias. El ejemplo de smoke.json es sintético y se ejecutó localmente; ambos métodos de búsqueda y la evaluación por raíces prefieren large con utilidad acumulada 5.5 frente a small con 1.0. No se mide rendimiento de partidas. No se añaden parámetros neuronales.

Pendientes: entrenamiento y materialización de redes, pesos y ejecución real de backends visuales externos, especialización aprendida de agentes, FAISS, consolidación de memoria y modelo neuronal del mundo, causalidad aprendida, integración semántica automática de recuerdos y predicciones, mctx, latencia real y optimización distribuida. La Fase 6 es un prototipo parcial; se conservan íntegramente las fases anteriores. Windows y CUDA no validados.
