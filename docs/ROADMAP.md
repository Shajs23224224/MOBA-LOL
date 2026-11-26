# Plan de continuidad

| Fase | Alcance | Estado |
|---|---|---|
| 1 | Núcleo local, catálogo, contratos, almacenamiento, recursos, diagnóstico | Entregado; ver informe de pruebas |
| 2 | 824 módulos PyTorch, materialización selectiva y auditoría | Prototipo verificado; entrenamiento y materialización completa pendientes |
| 3 | Modelos visuales preentrenados e integración | Parcial: OCR y puente verificados; adaptadores adicionales con contratos probados; pesos externos pendientes |
| 4 | 160 agentes expertos funcionales | Parcial: evaluadores de evidencia verificados; especialización aprendida pendiente |
| 5 | Memoria cognitiva y estado del mundo | Parcial: persistencia y filtro probabilístico verificados; consolidación y modelo neuronal pendientes |
| 6 | Razonamiento y planificación paralela | Parcial: lógica, causalidad y búsqueda sobre modelos explícitos verificadas; integración aprendida pendiente |
| 7 | Estrategia, táctica y economía | Parcial: evaluación MOBA offline verificada; especialización aprendida pendiente |
| 8 | Aprendizaje continuo durante partidas autorizadas | Parcial: entrenador offline auxiliar verificado con datos sintéticos; entrenamiento de expertos y experiencias reales pendientes |
| 9 | 32 integradores y 2 políticas finales | Pendiente |
| 10 | Optimización, interfaz de un botón y pruebas integrales | Pendiente |
| Build | Instalador de Windows, distribución de pesos y verificación final | Pendiente |

La fase 1 entrega infraestructura local. El despliegue Ray distribuido y aceleración C++ se integrarán cuando los trabajadores neuronales tengan contratos y pruebas de rendimiento reales. El catálogo no supone que los agentes puedan deliberar ni que el sistema esté entrenado.
