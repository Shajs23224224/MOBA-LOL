# Fase 9 — integración neuronal global (0.9.0)

Se instancian I01–I32 y F01/F02 como 34 MLP PyTorch independientes, en CPU FP32. Dos integradores por subsistema: 545→128→64 para representación y 609→64→4 para valor, riesgo, confianza e incertidumbre. Las estimaciones son no calibradas; no deben interpretarse como probabilidades verificadas. F01 usa 1092→128→16 logits diagnósticos de alternativas; F02 usa 1108→128→3 logits de alternativas abstractas. Estas dimensiones son decisiones de implementación provisionales. No hay actuadores ni ejecución de acciones: el único ActionIntent válido sigue siendo noop. No se implementan aún contratos completos de StrategicPlan/IntegratedState ni mapeo semántico estable de logits a objetivos y acciones.

Cada integrador recibe 256 características visuales, 256 características de un experto principal (ceros cuando no está disponible), una bandera de disponibilidad y 32 resúmenes: dos por categoría de memoria (presencia y similitud), cuatro de creencias, cuatro de propuestas de agentes pertinentes y ocho del planificador. F01 y F02 reciben 16 representaciones de 64, 16 estimaciones de 4 y tres probabilidades abstractas del Actor-Critic más un valor escalar. F02 también recibe los 16 logits F01. El predictor de siguiente estado se reporta, no alimenta directamente F01/F02. Las 34 redes no se entrenan automáticamente en el ciclo; se verificó backward y un paso SGD diagnóstico, no entrenamiento de calidad. No se emplean los logits para decidir recomendaciones ni establecer confianza. La salida de investigación anterior se mantiene y siempre requiere revisión humana.

Unified.run llama al review MOBA de Fase 7, que integra Cognition, memoria, creencias, consejo y planificación UCT. Permite seleccionar cero a dos subsistemas neuronales para inferencia real bajo demanda; la caché anterior conserva un experto por defecto. El resto de integradores recibe una señal explícita de ausencia. Las ocho memorias se consultan, pero sus resúmenes no son conocimiento semántico aprendido. El Snapshot y las evidencias aún son proporcionados externamente. El Actor-Critic se conecta por sus salidas, no mediante entrenamiento conjunto. Por defecto se crea sin entrenar; se puede pasar un ActorCritic restaurado por Checkpoints si fue entrenado con exactamente la misma representación visual y significado de alternativas. No se valida automáticamente esa compatibilidad. Se conserva el entrenador offline, sin entrenamiento en vivo. Se registra una revisión de fusión en la memoria metacognitiva. El ciclo usa varias transacciones y puede quedar parcial ante error; no se garantiza recuperación automática ni latencia real.

## Reproducción

Instalar pip install -e ".[neural,vision,memory,dev,reasoning-jax]" y Tesseract para la regresión completa. Ejemplo desde la raíz:

    import numpy as np
    from aion.cognition import Cognition
    from aion.neural import ExpertBank
    from aion.moba import Snapshot
    from aion.unified import Unified
    c = Cognition('data/unified.sqlite', bank=ExpertBank())
    u = Unified(c)
    s = Snapshot(health=.1, resource=.2, gold=100, threat=.9,
                 ally_support=0, objective_value=.9, farm_value=.4,
                 shop_access=True, free_slots=1, confidence=.95)
    r = u.run('fixture', 0, np.full((32,32,3),127,np.uint8), s,
              subsystems=('percepcion_visual','memoria'), ocr=False)
    print(r['audit'])

Repetir con tick estrictamente mayor o scope nuevo. Para un Actor-Critic previamente entrenado con entradas visuales compatibles, instanciar ActorCritic y restaurarlo con Checkpoints antes de pasarlo a Unified(learner=...). No se suministran pesos entrenados de juego.

## Auditoría y pruebas

58 pruebas aprobadas en 25.82 s (54 anteriores y 4 nuevas). Se verificaron las 34 estructuras, parámetros y almacenamiento independientes, formas, gradientes finitos, cambio de pesos mediante SGD, máscara de experto ausente, rechazo de entradas inválidas, influencia de Actor-Critic en la salida, inferencia de experto completo, 160 evaluadores, recuperación de las ocho categorías y persistencia de revisión. smoke.json registra otra ejecución real con dos expertos seleccionados secuencialmente y las 34 redes materializadas. Las cifras de parámetros excluyen modelos visuales externos.

I01, I03... I31: 78.144 parámetros cada una; I02, I04... I32: 39.300 cada una; F01: 141.968; F02: 142.339. Total nuevo 2.163.411 (8.653.644 bytes FP32). Total lógico propio incluyendo banco, router y Actor-Critic auxiliar: 52.789.247.311. En smoke quedaron residentes 265.886.012 bytes de parámetros, no RAM total del proceso. Las 824 estructuras principales siguen representadas por tensores meta y se materializan selectivamente; no hay 858 redes entrenadas ni simultáneamente residentes. Los 858 identificadores excluyen el router y el Actor-Critic auxiliar.

## Pendientes

Se mantienen todas las tareas anteriores: pesos visuales y ejecución real de backends externos, entrenamiento de expertos, especialización de agentes, FAISS, consolidación y predicción aprendida, mctx, TorchRL/PEFT, entrenamiento continuo con experiencias reales autorizadas, integración semántica y calibración, contratos finales completos, ejecución distribuida y Windows/CUDA. La Fase 9 es un prototipo parcial verificado. No controla el juego. Fase 10 y build final pendientes.
