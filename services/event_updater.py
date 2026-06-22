import json
import logging
import uuid
from typing import Any

from nats.aio.client import Client as NATS

logger = logging.getLogger(__name__)


class EventUpdater:
    def __init__(self, nats: NATS, request_timeout: float = 15.0) -> None:
        self._nats = nats
        self._request_timeout = request_timeout

    async def update_evento(self, evento_id: str, update_dto: dict) -> dict | None:
        payload = {
            "pattern": "eventos.update",
            "data": {"id": evento_id, **update_dto},
            "id": str(uuid.uuid4()),
        }
        message = json.dumps(payload).encode("utf-8")
        response = await self._nats.request(
            "eventos.update",
            message,
            timeout=self._request_timeout,
        )
        body = json.loads(response.data.decode("utf-8"))

        if isinstance(body, dict) and body.get("err"):
            raise RuntimeError(body["err"])

        if isinstance(body, dict) and "response" in body:
            return body["response"]

        return body
