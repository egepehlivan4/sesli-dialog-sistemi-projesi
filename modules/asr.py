"""
ASR modülü: faster-whisper ile Türkçe konuşmayı metne çevirir.

Model boyutu, cihaz ve hesaplama tipi config.py'den okunur (varsayılan:
small / cpu / int8). Whisper 16000 Hz ses bekler; farklı bir örnekleme
hızıyla gelen ses transcribe() içinde otomatik yeniden örneklenir.

Entegrasyon testi (TTS -> ASR):
    python modules/asr.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

WHISPER_SAMPLE_RATE = 16000
LANGUAGE = "tr"


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Basit lineer interpolasyonla yeniden örnekleme (ek bağımlılık gerektirmez)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    orig_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
    resampled = np.interp(target_times, orig_times, audio)
    return resampled.astype(np.float32)


class ASR:
    """faster-whisper tabanlı konuşma tanıma. Model ilk kullanımda
    yüklenir (lazy), her transcribe() çağrısında yeniden yüklenmez."""

    def __init__(self, model_size: str = None, device: str = None, compute_type: str = None):
        self.model_size = model_size or SETTINGS.asr_model_size
        self.device = device or SETTINGS.asr_device
        self.compute_type = compute_type or SETTINGS.asr_compute_type
        self._model: WhisperModel | None = None

    def load(self) -> WhisperModel:
        """Model zaten yüklenmediyse yükler. Yükleme süresini yazdırır."""
        if self._model is None:
            print(
                f"Whisper modeli yükleniyor: {self.model_size} "
                f"({self.device}, {self.compute_type})..."
            )
            start = time.perf_counter()
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            print(f"Model yüklendi ({time.perf_counter() - start:.3f} s).")
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        model = self.load()
        audio_16k = _resample(np.asarray(audio, dtype=np.float32), sample_rate, WHISPER_SAMPLE_RATE)
        segments, _info = model.transcribe(audio_16k, language=LANGUAGE)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe_file(self, path: Path) -> str:
        model = self.load()
        segments, _info = model.transcribe(str(path), language=LANGUAGE)
        return " ".join(segment.text.strip() for segment in segments).strip()


def main() -> None:
    from modules.tts import TTS

    text = "Yarın öğleden sonra saat üçte toplantımız var."
    print(f"Orijinal metin: {text}")

    tts = TTS()
    audio, sample_rate = tts.synthesize(text)
    audio_duration = len(audio) / sample_rate

    asr = ASR()
    asr.load()  # yükleme süresi ayrı yazdırılsın, transcribe ölçümüne karışmasın

    start = time.perf_counter()
    transcribed = asr.transcribe(audio, sample_rate)
    elapsed = time.perf_counter() - start
    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

    print()
    print(f"Orijinal metin       : {text}")
    print(f"ASR çıktısı          : {transcribed}")
    print(f"Ses süresi           : {audio_duration:.3f} s")
    print(f"Transcribe süresi    : {elapsed:.3f} s")
    print(f"RTF (transcribe/ses) : {rtf:.3f}")


if __name__ == "__main__":
    main()
