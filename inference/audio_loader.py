import io
import wave

import numpy as np

from config import settings


def _resample_audio(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32)

    target_len = int(round(len(audio) * target_rate / orig_rate))
    if target_len <= 0:
        return np.array([], dtype=np.float32)

    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def wav_bytes_to_float32(wav_data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Formato WAV no soportado (bytes por muestra: {width})")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != settings.sample_rate:
        audio = _resample_audio(audio, rate, settings.sample_rate)

    return audio
