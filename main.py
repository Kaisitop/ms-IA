import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from nats.aio.client import Client as NATS

from config import settings

# TF Hub debe usar caché del proyecto (evita carpeta TEMP corrupta en Windows)
os.environ.setdefault("TFHUB_CACHE_DIR", settings.tfhub_cache_dir)

from inference.engine import InferenceEngine
from services.audio_processor import AudioProcessor
from services.event_updater import EventUpdater
from services.nats_consumer import NatsConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ms-ia")

engine = InferenceEngine()
nats_client = NATS()
nats_consumer: NatsConsumer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nats_consumer

    try:
        engine.load()
    except Exception as exc:
        engine.status = "error"
        engine.error_message = str(exc)
        logger.error("No se pudieron cargar los modelos: %s", exc)

    await nats_client.connect(settings.nats_servers)
    logger.info("NATS conectado: %s", settings.nats_servers)

    if engine.is_ready:
        updater = EventUpdater(nats_client)
        processor = AudioProcessor(engine, updater)
        nats_consumer = NatsConsumer(nats_client, processor)
        await nats_consumer.start()
    else:
        logger.warning("Consumer NATS no iniciado: modelos no disponibles")

    yield

    if nats_consumer:
        await nats_consumer.stop()
    if nats_client.is_connected:
        await nats_client.close()


app = FastAPI(title="ms-IA", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "service": "ms-IA",
        "status": "ok" if engine.is_ready else "degraded",
        "model_status": engine.status,
        "model_path": settings.model_path,
        "model_exists": Path(settings.model_path).is_file(),
        "model_error": engine.error_message or None,
        "nats_connected": nats_client.is_connected,
        "nats_subject": settings.nats_subject_audio_ready,
        "bridge_root": settings.bridge_root,
        "yamnet_hub_url": settings.yamnet_hub_url,
        "tfhub_cache_dir": settings.tfhub_cache_dir,
        "alert_confidence_pct": settings.alert_confidence_pct,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail=engine.error_message or "Modelos no cargados",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo WAV requerido")

    wav_data = await file.read()
    if not wav_data:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    try:
        result = engine.classify_wav_bytes(wav_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error en inferencia")
        raise HTTPException(status_code=500, detail="Error interno de inferencia") from exc

    if result is None:
        raise HTTPException(status_code=422, detail="No se pudo clasificar el audio")

    return engine.to_dict(result)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.http_host,
        port=settings.http_port,
        reload=False,
    )
