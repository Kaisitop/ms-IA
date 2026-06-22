import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import settings
from inference.yamnet_loader import load_yamnet
from inference.audio_loader import wav_bytes_to_float32
from inference.classifier import (
    CLASS_NAMES,
    SUBTIPO_MAP,
    ClassificationResult,
    load_classifier,
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self) -> None:
        self._yamnet = None
        self._classifier = None
        self._classify_fn = None
        self.status = "idle"
        self.error_message = ""

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def load(self) -> None:
        model_path = Path(settings.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}. "
                "Copia manualmente my_yamnet_classifier.h5 a ms-ia/models/ (ver README)."
            )

        self.status = "loading"
        self._yamnet = load_yamnet(settings.yamnet_hub_url, settings.tfhub_cache_dir)

        logger.info("Cargando clasificador %s", model_path)
        self._classifier = load_classifier(str(model_path))

        @tf.function(reduce_retracing=True)
        def _classify(features):
            return self._classifier(features, training=False)

        self._classify_fn = _classify
        self._warmup()
        self.status = "ready"
        self.error_message = ""
        logger.info("Modelos listos")

    def _warmup(self) -> None:
        dummy = np.zeros(settings.window_size, dtype=np.float32)
        _, embeddings, _ = self._yamnet(dummy)
        features = tf.reduce_max(embeddings, axis=0, keepdims=True)
        self._classify_fn(features)

    def _run_classification(self, audio_buffer: np.ndarray) -> ClassificationResult | None:
        _, embeddings, _ = self._yamnet(audio_buffer)
        if len(embeddings) == 0:
            return None

        features = tf.reduce_max(embeddings, axis=0, keepdims=True)
        predictions = self._classify_fn(features).numpy()[0]
        idx = int(np.argmax(predictions))
        class_name = CLASS_NAMES[idx]
        confidence_pct = round(float(predictions[idx]) * 100, 2)
        confidence = round(float(predictions[idx]), 4)

        is_alert = (
            class_name != "Negative_Class"
            and confidence_pct >= settings.alert_confidence_pct
        )

        return ClassificationResult(
            class_name=class_name,
            subtipo=SUBTIPO_MAP[class_name],
            confidence_pct=confidence_pct,
            confidence=confidence,
            index=idx,
            is_alert=is_alert,
        )

    def _classify_window(self, audio_buffer: np.ndarray) -> ClassificationResult | None:
        if np.max(np.abs(audio_buffer)) < settings.noise_gate:
            return ClassificationResult(
                class_name="Negative_Class",
                subtipo="otro",
                confidence_pct=100.0,
                confidence=1.0,
                index=0,
                is_alert=False,
            )
        return self._run_classification(audio_buffer)

    def classify_audio(self, audio: np.ndarray) -> ClassificationResult | None:
        if len(audio) == 0:
            return None

        window_size = settings.window_size
        block_size = settings.block_size

        if len(audio) < window_size:
            window = np.zeros(window_size, dtype=np.float32)
            window[: len(audio)] = audio
            windows = [window]
        else:
            windows = [
                audio[i : i + window_size]
                for i in range(0, len(audio) - window_size + 1, block_size)
            ]
            if not windows:
                windows = [audio[-window_size:]]

        best: ClassificationResult | None = None
        for window in windows:
            result = self._classify_window(window)
            if result is None:
                continue
            threat = (result.index != 0, result.confidence_pct)
            if best is None or threat > (best.index != 0, best.confidence_pct):
                best = result
        return best

    def classify_wav_bytes(self, wav_data: bytes) -> ClassificationResult | None:
        audio = wav_bytes_to_float32(wav_data)
        return self.classify_audio(audio)

    def to_dict(self, result: ClassificationResult) -> dict:
        severidad = 1
        if result.is_alert:
            severidad = 3 if result.subtipo == "disparo" else 2

        return {
            "class": result.class_name,
            "subtipo": result.subtipo,
            "confidence_pct": result.confidence_pct,
            "confianza": result.confidence,
            "severidad_sugerida": severidad,
            "is_alert": result.is_alert,
            "fuente": "yamnet",
        }
