# Fase 3B — adaptadores visuales adicionales (0.3.1)

Se añaden cuatro adaptadores opcionales de sólo lectura. Las fases anteriores no se eliminan. Ningún modelo externo nuevo fue cargado ni entrenado. El pipeline recibe imágenes locales, no captura ni controla el cliente de un videojuego.

## Instalación y referencias

- OmniParser V2: https://github.com/microsoft/OmniParser y https://huggingface.co/microsoft/OmniParser-v2.0 . Preparar un entorno independiente siguiendo el repositorio oficial y descargar allí el detector de iconos y el modelo de descripción. Ejecutar su servidor oficial en loopback. El adaptador HTTP sólo conecta a 127.0.0.1 y utiliza POST /parse/; valida el formato normalizado xyxy. Las descripciones no tienen una confianza calibrada, por lo que se devuelve null.
- SAM 2.1 Tiny: https://github.com/facebookresearch/sam2 . Instalar el repositorio oficial y obtener su checkpoint tiny del proveedor. El adaptador usa SAM2ImagePredictor con configuración sam2.1_hiera_t.yaml. Exige SHA-256 esperado obtenido de una fuente confiable antes de cargar un archivo .pt. El hash no demuestra por sí solo procedencia ni seguridad. Sólo usar archivos de confianza. La segmentación usa rectángulos proporcionados; no clasifica objetos.
- ByteTrack: https://supervision.roboflow.com/0.26.1/trackers/ . Se eligió supervision==0.26.1 para mantener la API sv.ByteTrack.update_with_detections. Es un algoritmo de asociación de detecciones y no requiere pesos independientes. Requiere detecciones con confianza explícita de un detector; el CLI usa RT-DETR opcional. No se inventan puntuaciones para contornos ni captions de OmniParser.
- PaddleOCR: https://www.paddleocr.ai/v3.3.0/en/version3.x/pipeline_usage/OCR.html . Usar PaddleOCR 3.3 y PaddlePaddle compatible con la plataforma, en un entorno compatible con Python 3.11/3.12. Preparar los directorios de inferencia de detección y reconocimiento (por ejemplo PP-OCRv5 mobile). Se desactivan orientación y corrección documental; se convierte RGB a BGR. Se leen rec_texts, rec_scores y rec_polys. El usuario debe verificar la procedencia e integridad de ambos directorios por separado.

Instalar primero las dependencias de las fases anteriores: pip install -e ".[neural,vision,dev]". Los extras vision-track y vision-paddle son opcionales; SAM2 y el servidor OmniParser se instalan conforme a sus propias instrucciones. No se comprobó la resolución conjunta de dependencias, especialmente Python 3.13. No se descargan pesos implícitamente desde código de AION; los proveedores pueden realizar sus propias comprobaciones de red.

## Ejemplos de CLI

- python -m aion.vision_extra_cli imagen.png --omni-port 8000 --neural
- python -m aion.vision_extra_cli imagen.png --paddle /modelos/det /modelos/rec --neural
- python -m aion.vision_extra_cli imagen.png --sam /modelos/sam2.1_hiera_tiny.pt SHA256_ESPERADO --box 10 10 100 100
- python -m aion.vision_extra_cli f1.png f2.png --rtdetr-dir /modelos/rtdetr --track --fps 30

La salida es JSON por fotograma. Las regiones de OmniParser, Paddle y RT-DETR pueden contribuir a las 256 características de Fase 3. Los resultados SAM y ByteTrack se reportan por separado y aún no se codifican en el vector neuronal. El CLI adicional no sustituye el anterior.

## Verificación

Linux Python 3.13.5, PyTorch 2.10.0+cpu, OpenCV 4.13.0. 27 pruebas aprobadas en 7.92 s, incluidas 21 de regresión. Seis nuevas pruebas: SHA-256 sobre bytes de prueba con alteración posterior; contrato de coordenadas OmniParser; HTTP loopback real con servidor de prueba (no OmniParser real); transformación de polígonos PaddleOCR de ejemplo; contrato SAM2 con predictor de prueba; contrato ByteTrack con objeto de prueba. Estos cuatro contratos no son inferencia ni seguimiento reales con los proveedores.

No estaban instalados SAM2, PaddleOCR, PaddlePaddle, Supervision ni Transformers. Las solicitudes HEAD a Hugging Face, servidor de Meta y servidor Paddle fallaron por resolución DNS. pip download supervision==0.26.1 no encontró una distribución disponible en el entorno. No se obtuvo ningún peso nuevo, no se calculó su hash real, no se probó ningún backend externo ni se verificó exactitud sobre imágenes de partidas. Las pruebas previas de Tesseract, OpenCV y PyTorch sí se ejecutaron. Fase 3 sigue parcial hasta probar todos los modelos y su integración, segmentación temporal y reconocimiento específico.
