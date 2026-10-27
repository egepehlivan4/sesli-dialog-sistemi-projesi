"""
Ortam değişkenlerinden okunan ayarlar.

Platforma veya makineye bağlı HER ayar burada toplanır; kodun geri kalanı
hangi işletim sisteminde çalıştığını hiç bilmez. Varsayılan değerler
Mac (Apple Silicon, CPU) için uygundur — Windows'taki geliştirici kendi
makinesine göre değerleri .env dosyasından ezer.

Kullanım:
    from config import SETTINGS
    print(SETTINGS.asr_model_size)
"""

import os
from dataclasses import dataclass
from pathlib import Path

# .env dosyası varsa ve python-dotenv kuruluysa otomatik yükle.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _get_opt_device(name: str) -> int | str | None:
    """Ses aygıtını okur: sayısal ID ise int, isim ise str, boşsa None döner."""
    value = os.environ.get(name)
    if not value or not value.strip():
        return None
    val = value.strip()
    try:
        return int(val)
    except ValueError:
        return val


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _get_list(name: str, default: list) -> list:
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    # --- Genel ---
    base_dir: Path
    models_dir: Path
    log_level: str

    # --- Ses I/O (sounddevice) ---
    sample_rate: int
    channels: int
    audio_dtype: str
    input_device: int | str | None
    output_device: int | str | None
    audio_chunk_ms: int

    # --- VAD (silero-vad) ---
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int

    # --- ASR (faster-whisper) ---
    asr_model_size: str
    asr_partial_model: str
    asr_device: str
    asr_compute_type: str
    asr_language: str
    asr_partial_interval_ms: int
    asr_preroll_ms: int
    asr_cpu_threads: int

    # --- LLM (Ollama HTTP API) ---
    llm_base_url: str
    llm_model: str
    llm_timeout_s: float
    llm_temperature: float
    llm_compare_models: list

    # --- TTS (piper-tts) ---
    tts_voice: str
    tts_model_path: Path
    tts_config_path: Path

    # --- Controller (barge-in) ---
    barge_in_strategy: str


def _load() -> Settings:
    models_dir = _get_path("MODELS_DIR", BASE_DIR / "models")
    tts_voice = _get_str("TTS_VOICE", "tr_TR-dfki-medium")

    return Settings(
        base_dir=BASE_DIR,
        models_dir=models_dir,
        log_level=_get_str("LOG_LEVEL", "INFO"),
        # Ses I/O — varsayılan: 16 kHz, mono, float32 (CLAUDE.md gereği)
        sample_rate=_get_int("SAMPLE_RATE", 16000),
        channels=_get_int("CHANNELS", 1),
        audio_dtype=_get_str("AUDIO_DTYPE", "float32"),
        input_device=_get_opt_device("INPUT_DEVICE"),
        output_device=_get_opt_device("OUTPUT_DEVICE"),
        audio_chunk_ms=_get_int("AUDIO_CHUNK_MS", 20),
        # VAD
        vad_threshold=_get_float("VAD_THRESHOLD", 0.5),
        vad_min_speech_ms=_get_int("VAD_MIN_SPEECH_MS", 250),
        vad_min_silence_ms=_get_int("VAD_MIN_SILENCE_MS", 500),
        # ASR — varsayılan: Mac CPU için "small" / "int8"
        asr_model_size=_get_str("ASR_MODEL_SIZE", "small"),
        asr_partial_model=_get_str("ASR_PARTIAL_MODEL", "tiny"),
        asr_device=_get_str("ASR_DEVICE", "cpu"),
        asr_compute_type=_get_str("ASR_COMPUTE_TYPE", "int8"),
        asr_language=_get_str("ASR_LANGUAGE", "tr"),
        asr_partial_interval_ms=_get_int("ASR_PARTIAL_INTERVAL_MS", 500),
        asr_preroll_ms=_get_int("ASR_PREROLL_MS", 500),
        asr_cpu_threads=_get_int("ASR_CPU_THREADS", 4),
        # LLM — Ollama yerel sunucu
        llm_base_url=_get_str("LLM_BASE_URL", "http://localhost:11434"),
        llm_model=_get_str("LLM_MODEL", "gemma3:4b"),
        llm_timeout_s=_get_float("LLM_TIMEOUT_S", 30.0),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
        llm_compare_models=_get_list(
            "LLM_COMPARE_MODELS", ["qwen2.5:3b", "qwen2.5:7b", "gemma3:4b"]
        ),
        # TTS — piper-tts, Türkçe ses
        tts_voice=tts_voice,
        tts_model_path=_get_path("TTS_MODEL_PATH", models_dir / f"{tts_voice}.onnx"),
        tts_config_path=_get_path(
            "TTS_CONFIG_PATH", models_dir / f"{tts_voice}.onnx.json"
        ),
        # Controller — barge-in stratejisi (bkz. modules/strategies.py)
        barge_in_strategy=_get_str("BARGE_IN_STRATEGY", "vad_only"),
    )


SETTINGS = _load()


if __name__ == "__main__":
    # Tek başına çalıştırıldığında çözümlenen ayarları özetle.
    for field_name, value in SETTINGS.__dict__.items():
        print(f"{field_name} = {value!r}")
        