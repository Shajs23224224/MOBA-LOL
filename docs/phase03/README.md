# Fase 3 — visión multimodal, prototipo 0.3.0

Canal de percepción de sólo lectura: imágenes locales RGB, regiones de OpenCV sin clase semántica, OCR Tesseract preentrenado (inglés), representación de 256 características, integración con ExpertBank N001/N002. El vector contiene 192 valores RGB, 48 valores de ocupación espacial y 16 cubetas hash de palabras; no es un embedding semántico aprendido. El resultado neuronal es diagnóstico, no política entrenada ni decisión. No captura el escritorio ni controla el juego.

## Ejecución

Instalar las dependencias de visión y neuronales con `pip install -e ".[neural,vision,dev]"`. Para OCR, instalar Tesseract 5 con datos de idioma eng y añadir su ejecutable al PATH. Para el detector o VLM opcional, instalar `.[vision-hf]` además de los extras anteriores, y preparar checkpoints locales compatibles con Transformers y safetensors (consultar licencias y verificar integridad antes de uso). No se descargan pesos automáticamente. Ejemplos:

- `python -m aion.vision_cli imagen.png --neural`
- `python -m aion.vision_cli imagen.png --no-ocr --rtdetr-dir /ruta/checkpoint --neural`
- `python -m aion.vision_cli imagen.png --grounding-dir /ruta/checkpoint --labels "person. icon."`
- `python -m aion.vision_cli imagen.png --vl-dir /ruta/checkpoint`

## Modelos del PDF maestro y estado

- RT-DETR: adaptador local Transformers, no ejecutado con pesos. El checkpoint R18 original requiere una conversión compatible; no se acepta un .pth bruto.
- Grounding DINO: adaptador local Transformers, no ejecutado con pesos.
- Qwen3-VL-4B-Instruct: adaptador visual-lingüístico local, no ejecutado con pesos; descripción etiquetada como no verificada.
- UI-TARS 1.5-7B: puede evaluarse con el adaptador visual-lingüístico si su arquitectura es compatible; no se ha verificado esa compatibilidad.
- OmniParser V2, SAM 2, ByteTrack y PaddleOCR: pendientes de implementación y validación. Tesseract es una alternativa operativa para OCR, no PaddleOCR.

Fuentes de interfaces: https://huggingface.co/docs/transformers/model_doc/rt_detr ; https://huggingface.co/docs/transformers/model_doc/grounding-dino ; https://huggingface.co/docs/transformers/main/tasks/image_text_to_text . Los pesos externos y licencias no se han auditado en esta fase. La red del entorno no resolvió download.pytorch.org y Transformers no estaba instalado. No se probó Windows, CUDA, imágenes reales de partidas, reconocimiento de campeones, seguimiento temporal ni latencia en vivo. No se hace afirmación de precisión de juego. Los modelos opcionales sólo se han comprobado sintácticamente; pueden requerir ajustes de compatibilidad, memoria y versiones.

## Verificación local

21 pruebas aprobadas, sin omisiones: 17 anteriores y 4 de visión. Pruebas nuevas: PNG generado para diagnóstico, cambios en píxeles, regiones acotadas, OCR real de GOLD 1250, inferencia real de dimensión completa y salida finita. No es un conjunto de entrenamiento o de evaluación de MOBA. Consulte pytest.txt y smoke.json. La fase 3 es parcial: quedan las integraciones indicadas y validación sobre datos autorizados. Las fases 1 y 2 se conservan.
