"""
TTS modülü: piper-tts kullanarak Türkçe metni sese çevirir.

Model dosyaları (ONNX ağırlığı + config json) config.py'deki MODELS_DIR
altında aranır; hangi sesin kullanılacağı TTS_VOICE ortam değişkeni ile
seçilir (varsayılan: tr_TR-dfki-medium).

Tek başına çalıştırma:
    python modules/tts.py "Merhaba, nasılsın?"
    python modules/tts.py            # varsayılan cümleyi seslendirir
"""

import sys
import time
import wave
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

MIN_PIPER_VERSION = (1, 4)


def _check_piper_version() -> None:
    try:
        installed = version("piper-tts")
    except PackageNotFoundError:
        return

    parts = installed.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return

    if (major, minor) < MIN_PIPER_VERSION:
        raise RuntimeError(
            f"piper-tts {installed} kurulu ama >=1.4 gerekli. "
            "1.4'ten eski sürümler macOS'ta piper-phonemize derleme hatası verir. "
            'Düzeltmek için: pip install --upgrade "piper-tts>=1.4"'
        )


_check_piper_version()

try:
    from piper import PiperVoice
except ImportError as exc:
    raise ImportError(
        'piper-tts bulunamadı. Kurulum için: pip install "piper-tts>=1.4"'
    ) from exc


DEFAULT_TEXT = "Merhaba, bu bir Türkçe sesli diyalog sistemi denemesidir."


class TTS:
    """piper-tts tabanlı metin-sesli dönüştürücü."""

    def __init__(self, model_path: Path | None = None, config_path: Path | None = None):
        self.model_path = Path(model_path) if model_path else SETTINGS.tts_model_path
        self.config_path = Path(config_path) if config_path else SETTINGS.tts_config_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper ses modeli bulunamadı: {self.model_path}\n"
                f"'{SETTINGS.tts_voice}' sesinin dosyalarını {SETTINGS.models_dir} "
                "klasörüne indirin."
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Piper model config dosyası bulunamadı: {self.config_path}"
            )

        self._voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path))

    @property
    def sample_rate(self) -> int:
        return self._voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Metni sese çevirir. (float32 numpy dizisi, örnekleme hızı) döner."""
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
        return audio.astype(np.float32), self.sample_rate

    def synthesize_to_file(self, text: str, path: Path) -> None:
        path = Path(path)
        with wave.open(str(path), "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)


def main() -> None:
    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT
    print(f"Metin: {text}")

    tts = TTS()

    start = time.perf_counter()
    audio, sample_rate = tts.synthesize(text)
    elapsed = time.perf_counter() - start

    audio_duration = len(audio) / sample_rate if sample_rate else 0.0
    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

    print(f"Örnekleme hızı: {sample_rate} Hz")
    print(f"Üretim süresi: {elapsed:.3f} s")
    print(f"Ses süresi: {audio_duration:.3f} s")
    print(f"RTF (üretim süresi / ses süresi): {rtf:.3f}")

    sd.play(audio, samplerate=sample_rate)
    sd.wait()


if __name__ == "__main__":
    main()
