# AION Android — 0.9.1

Destino principal: APK Android, no instalador Windows. Android API 26+ (Android 8), objetivo API 35; CPU, sin librerías nativas ni permisos de red. Fase 10 móvil iniciada, no finalizada. No se ha compilado ni instalado todavía: el entorno local carece de Android SDK y Gradle. JDK disponible para la prueba del núcleo Java. Android Studio o un runner con SDK 35, AGP 8.7.3, Gradle 8.9 y JDK 17 pueden intentar `gradle -p android :app:assembleDebug`; resultado esperado si compila: `android/app/build/outputs/apk/debug/app-debug.apk`. No se incluye wrapper ni APK. Ver https://developer.android.com/build/releases/agp-8-7-0-release-notes.

## Qué se ejecuta en el teléfono

Launcher nativo Java, selección de archivos Android, diagnóstico de CPU/heap, importación de vector de 256 características, lectura acotada de imágenes locales (promedio RGB), inferencia Actor-Critic auxiliar FP32 (34.628 parámetros), almacenamiento privado del último resultado. El modelo se importa manualmente, no hay pesos entrenados incluidos. La imagen se examina por separado; no se convierte al vector de Fase 3. No hay permisos de red, capturas continuas, overlays, accesibilidad ni ejecución de acciones. La actividad no ha sido probada en Android. El núcleo Java Policy se prueba con JDK 21 compilando para Java 17.

Exportar desde el núcleo de investigación con `python -m aion.mobile_export policy.aion --diagnostic` (pesos iniciales) o `python -m aion.mobile_export policy.aion --checkpoint-dir data/learning/weights` (checkpoint propio). Copiar archivo al teléfono, importar desde la app y pulsar iniciar. El formato AIN1 incluye versión 1, 34.628 floats little-endian y SHA-256 de cabecera y pesos; 138.552 bytes. Detecta corrupción, no autentica procedencia. El vector importado debe corresponder a la representación y al significado de alternativas del checkpoint; no se certifica automáticamente. Los logits son abstractos y nunca se convierten en controles de un juego. Los checkpoints se exportan desde Python; Android no ejecuta el entrenador.

## Estado de la migración

| Fase | Android |
|---|---|
| 1 | Launcher y diagnóstico básicos; bus, SQLite completo, registro y gestión de recursos Python pendientes de port |
| 2 | 824 expertos conservados en Python; inferencia selectiva móvil pendiente |
| 3 | Lectura local RGB básica; OCR, detección, seguimiento y codificador de 256 pendientes de port |
| 4 | 160 evaluadores conservados; ejecución nativa pendiente |
| 5 | Último diagnóstico persistido; ocho memorias y filtro probabilístico pendientes |
| 6 | Razonamiento y planificación conservados; port móvil pendiente |
| 7 | Evaluador MOBA conservado; port móvil pendiente |
| 8 | Inferencia Actor-Critic con LoRA portada, replay, entrenamiento y actualización pendientes |
| 9 | 34 redes de fusión conservadas; exportación y ejecución nativa pendientes |
| 10 | Proyecto Android iniciado; compilación, pruebas en dispositivo, optimización y APK pendientes |

Las 858 estructuras y 52.789.247.311 parámetros lógicos se mantienen. No son simultáneamente residentes ni entrenados. FP32 completo ~211,16 GB, FP16 ~105,58 GB, INT4 ideal ~26,39 GB. Se necesita perfil selectivo/compacto para teléfono, con cuantización real validada, modelos móviles y presupuesto de memoria. No se ha reducido ni sustituido silenciosamente la arquitectura original. El código Python sigue como núcleo de investigación/exportación independiente de Android. La app no requiere PC conectado para inferencia tras importar el modelo. Para funcionamiento completamente autónomo en el móvil quedan pendientes el entrenamiento local, memoria, percepción, razonamiento y agentes. No se afirma que sea una IA jugable. Todas las tareas anteriores siguen vigentes.

## Actualización 0.10.0

La versión posterior agrega un laboratorio de aprendizaje local independiente y una configuración de GitHub Actions. Véase [informe Fase 10](phase10/README.md) para el estado actual; las cifras y pasos anteriores corresponden a 0.9.1.
