from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_path: str = str(BASE_DIR / "models" / "my_yamnet_classifier.h5")
    yamnet_hub_url: str = "https://tfhub.dev/google/yamnet/1"
    tfhub_cache_dir: str = str(BASE_DIR / ".cache" / "tfhub")

    sample_rate: int = 16000
    block_size: int = 8000
    window_size: int = 14000
    noise_gate: float = 0.015
    alert_confidence_pct: float = 80.0

    nats_servers: str = "nats://localhost:4222"
    nats_subject_audio_ready: str = "eventos.audio.ready"
    bridge_root: str = str(BASE_DIR.parent / "ms-IoT-Bridge")

    http_host: str = "0.0.0.0"
    http_port: int = 8200


settings = Settings()
