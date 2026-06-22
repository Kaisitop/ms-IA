import logging
import os
import shutil
from pathlib import Path

import tensorflow_hub as hub

logger = logging.getLogger(__name__)

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_CACHE_KEY = "9616fd04ec2360621642ef9455b84f4b668e219e"


def configure_tfhub_cache(cache_dir: str) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = str(path)
    return path


def _cache_entry_path(cache_dir: Path) -> Path:
    return cache_dir / YAMNET_CACHE_KEY


def _cache_is_valid(cache_dir: Path) -> bool:
    entry = _cache_entry_path(cache_dir)
    if not entry.is_dir():
        return False
    return (entry / "saved_model.pb").is_file() or (entry / "saved_model.pbtxt").is_file()


def _clear_corrupt_cache(cache_dir: Path) -> None:
    entry = _cache_entry_path(cache_dir)
    if entry.exists():
        logger.warning("Eliminando caché TF Hub corrupto: %s", entry)
        shutil.rmtree(entry, ignore_errors=True)

    temp_entry = Path(os.environ.get("TEMP", "/tmp")) / "tfhub_modules" / YAMNET_CACHE_KEY
    if temp_entry.exists() and temp_entry != entry:
        logger.warning("Eliminando caché TF Hub corrupto en TEMP: %s", temp_entry)
        shutil.rmtree(temp_entry, ignore_errors=True)


def load_yamnet(hub_url: str, cache_dir: str):
    cache_path = configure_tfhub_cache(cache_dir)

    if not _cache_is_valid(cache_path):
        _clear_corrupt_cache(cache_path)

    try:
        logger.info("Cargando YAMNet desde %s (cache: %s)", hub_url, cache_path)
        return hub.load(hub_url)
    except ValueError as exc:
        if "saved_model.pb" not in str(exc):
            raise
        logger.warning("Caché inválido detectado, reintentando descarga: %s", exc)
        _clear_corrupt_cache(cache_path)
        return hub.load(hub_url)
