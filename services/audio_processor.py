import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import aiofiles

from inference.engine import InferenceEngine
from services.audio_path import resolve_audio_path
from services.event_updater import EventUpdater

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, engine: InferenceEngine, updater: EventUpdater) -> None:
        self._engine = engine
        self._updater = updater
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def handle(self, payload: dict) -> None:
        evento_id = payload.get("eventoId")
        audio_url = payload.get("audioUrl")

        if not evento_id or not audio_url:
            logger.warning("Payload eventos.audio.ready incompleto: %s", payload)
            return

        if not self._engine.is_ready:
            logger.error("Motor IA no listo; evento %s omitido", evento_id)
            return

        try:
            audio_path = resolve_audio_path(audio_url)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return

        async with aiofiles.open(audio_path, "rb") as file:
            wav_data = await file.read()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._engine.classify_wav_bytes,
            wav_data,
        )

        if result is None:
            logger.warning("Sin clasificación para evento %s", evento_id)
            return

        classification = self._engine.to_dict(result)
        logger.info(
            "IA evento=%s origen=%s -> %s (%.1f%%)",
            evento_id,
            payload.get("evento_id_origen"),
            classification["subtipo"],
            classification["confidence_pct"],
        )

        update_dto = {
            "subtipo": classification["subtipo"],
            "confianza": classification["confianza"],
            "severidad": classification["severidad_sugerida"],
            "fuente": classification["fuente"],
            "procesado": True,
            "metadatos": {
                "ia_class": classification["class"],
                "ia_confidence_pct": classification["confidence_pct"],
                "ia_is_alert": classification["is_alert"],
                "evento_id_origen": payload.get("evento_id_origen"),
                "codigo_nodo": payload.get("codigo_nodo"),
            },
        }

        await self._updater.update_evento(evento_id, update_dto)
        logger.info("Evento %s actualizado en ms-core", evento_id)
