import asyncio
import json
import logging

from nats.aio.client import Client as NATS

from config import settings
from services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class NatsConsumer:
    def __init__(self, nats: NATS, processor: AudioProcessor) -> None:
        self._nats = nats
        self._processor = processor
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Suscrito a NATS %s", settings.nats_subject_audio_ready)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        sub = await self._nats.subscribe(
            settings.nats_subject_audio_ready,
            queue="ms-ia-workers",
        )
        try:
            while self._running:
                try:
                    msg = await sub.next_msg(timeout=1)
                except (TimeoutError, asyncio.TimeoutError):
                    continue

                try:
                    payload = json.loads(msg.data.decode("utf-8"))
                    await self._processor.handle(payload)
                except Exception:
                    logger.exception("Error procesando eventos.audio.ready")
        finally:
            await sub.unsubscribe()
