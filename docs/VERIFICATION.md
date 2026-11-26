# Informe de verificación — Fase 1

Pruebas efectuadas en Linux, Python 3.13.5, PyTorch CPU. Las salidas reales de comandos se conservan en `verification/`. Se ejecutaron 13 pruebas de pytest sin fallos ni omisiones. La prueba neuronal realizó inferencia y restauración exacta de un checkpoint local de 212 parámetros sin entrenamiento. El arranque de la CLI generó una sesión de diagnóstico, almacenó eventos en SQLite y se detuvo. Los 858 módulos y 160 roles del manifiesto son requisitos pendientes, no modelos materializados.

| Componente | Verificación |
|---|---|
| Catálogo | Identificadores únicos, 824/32/2, 16×2048, 160 roles |
| Configuración | Validación de rangos y rechazo de entorno no admitido |
| Contratos | Validación de confianza y rechazo de acciones de juego |
| Bus | Aislamiento de mensajes, saturación y propagación de excepciones |
| Recursos | Reserva y liberación cooperativa, rechazo por presupuesto y falta de CUDA |
| Persistencia | Sesión y evento recuperados desde otra conexión |
| Modelos | Hash válido, detección de alteración, rechazo de rutas fuera del registro y extensiones inseguras |
| PyTorch/safetensors | Inferencia real y restauración de pesos en CPU |
| Motor | Inicio, detención, reinicio y error de arranque |

No evaluados: Windows, CUDA, Ray, visión, juego, estrategia, aprendizaje continuo y carga completa. La prueba de carga no equivale a prueba de rendimiento MOBA. No se entregan pesos externos ni binarios. Consulte README para límites y reproducción.
