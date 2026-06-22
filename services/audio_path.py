import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def resolve_audio_path(audio_url: str) -> Path:
    raw = Path(audio_url)
    if raw.is_file():
        return raw

    bridge_root = Path(settings.bridge_root)
    candidates = [
        bridge_root / audio_url,
        bridge_root / "data" / "audio" / raw.name,
        Path(audio_url),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No se encontró el audio '{audio_url}'. "
        f"Verifica BRIDGE_ROOT={settings.bridge_root}"
    )
