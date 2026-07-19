# ms-IA

> **Monorepo:** [README.md](../README.md) · [docs/contexto.md](../docs/contexto.md)

Microservicio de clasificación de audio para CENTINELA (UNEMI).

Usa **YAMNet** (TensorFlow Hub) + clasificador entrenado (`my_yamnet_classifier.h5`) para detectar **disparos** y **gritos** en clips WAV.

## Flujo integrado (con ms-IoT-Bridge + ms-core)

```text
nodo_app → MQTT centinela/evento
    ↓
ms-IoT-Bridge → guarda WAV + eventos.create (subtipo: "otro")
    ↓ NATS eventos.audio.ready
ms-ia → clasifica WAV → eventos.update (subtipo: disparo/grito/otro)
    ↓
ms-core → PostgreSQL app.eventos + alertas
```

## Flujo manual (prueba Postman)

```text
POST /classify (WAV) → YAMNet + H5 → { subtipo, confianza, severidad_sugerida }
```

## Modelo entrenado (copia manual)

El archivo `.h5` **no se sube a Git**. Debes copiarlo manualmente:

1. Ubica tu modelo entrenado, por ejemplo:
   `C:\Users\LENOVO\Downloads\DetectorSonidosAI\DetectorSonidosAI\my_yamnet_classifier.h5`
2. Cópialo a esta carpeta con el nombre exacto:

```text
ms-ia/models/my_yamnet_classifier.h5
```

3. Verifica que exista antes de levantar el servicio:

```powershell
Test-Path .\models\my_yamnet_classifier.h5
```

Si falta, `/health` responderá `model_status: error` con un mensaje indicando que debes copiar el archivo.

## Requisitos

- Python 3.11+
- ~2 GB libres (TensorFlow + caché YAMNet)
- Conexión a internet la **primera vez** (descarga YAMNet a `.cache/tfhub/`)

Si ves el error `contains neither 'saved_model.pb' nor 'saved_model.pbtxt'`:

1. Detén el servicio (`Ctrl+C`)
2. Borra la caché corrupta:

```powershell
Remove-Item -Recurse -Force .\.cache\tfhub -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$env:TEMP\tfhub_modules\9616fd04ec2360621642ef9455b84f4b668e219e" -ErrorAction SilentlyContinue
```

3. Vuelve a ejecutar `python main.py` (descargará YAMNet de nuevo, ~1 min)

## Configuración

```powershell
cd ms-ia
copy .env.example .env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Levantar el servicio

```powershell
python main.py
```

Health check: `GET http://localhost:8200/health`

Respuesta esperada cuando todo está OK:

```json
{
  "service": "ms-IA",
  "status": "ok",
  "model_status": "ready",
  "model_exists": true
}
```

## Probar clasificación

Con Postman o curl, envía un WAV:

```powershell
curl -X POST "http://localhost:8200/classify" -F "file=@ruta\al\clip.wav"
```

Respuesta ejemplo:

```json
{
  "class": "Gunshot",
  "subtipo": "disparo",
  "confidence_pct": 92.5,
  "confianza": 0.925,
  "severidad_sugerida": 3,
  "is_alert": true,
  "fuente": "yamnet"
}
```

## Mapeo de clases → CENTINELA

| Clase del modelo | `subtipo` ms-core | Alerta (≥ 80%) |
|---|---|---|
| `Negative_Class` | `otro` | No |
| `Gunshot` | `disparo` | Sí |
| `Screaming` | `grito` | Sí |

`confianza` se devuelve en escala **0–1** (formato que espera `ms-core`).

## Parámetros de inferencia

Mismos valores probados en `DetectorSonidosAI`:

| Variable | Default | Descripción |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Hz del audio |
| `WINDOW_SIZE` | 14000 | Muestras por ventana |
| `BLOCK_SIZE` | 8000 | Stride entre ventanas |
| `NOISE_GATE` | 0.015 | Umbral de silencio |
| `ALERT_CONFIDENCE_PCT` | 80.0 | Confianza mínima para alerta |

## Estructura

```text
ms-ia/
├── main.py                 # FastAPI :8200
├── config.py
├── inference/
│   ├── engine.py           # YAMNet + clasificador
│   ├── classifier.py       # carga H5 y mapeo subtipo
│   └── audio_loader.py     # WAV → float32 16kHz
├── models/
│   └── my_yamnet_classifier.h5   # ← copia manual, no en Git
└── services/               # NATS (próximo paso)
```
