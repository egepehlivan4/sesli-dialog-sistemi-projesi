"""
VAD modülü: silero-vad ile gerçek zamanlı konuşma/sessizlik tespiti.

silero-vad modeli 16 kHz'de tam olarak 512 örneklik (32 ms) pencere
bekler. Ses hattımız chunk_ms=20 ms (320 örnek) chunk'lar ürettiği
için VAD, gelen chunk'ları kendi içinde bir buffer'da biriktirir ve
her 512 örnek tamamlandığında bir model çağrısı yapar.

Ham olasılığın üstüne basit bir durum makinesiyle olay üretir:
  SPEECH_START - olasılık threshold'u aştı ve en az min_speech_ms sürdü
  SPEECH_END   - olasılık threshold'un altına düştü ve en az min_silence_ms sürdü

Tek başına çalıştırma (mikrofonla canlı test):
    python modules/vad.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from silero_vad import load_silero_vad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

# silero-vad bu proje için sabit 16 kHz / 512 örneklik pencere bekler
# (ses hattımız config'te zaten 16000 Hz olarak tanımlı).
SILERO_SAMPLE_RATE = 16000
SILERO_WINDOW_SAMPLES = 512


@dataclass
class VADEvent:
    kind: str  # "SPEECH_START" | "SPEECH_END"
    timestamp_ms: float


class VAD:
    """silero-vad tabanlı, chunk boyutundan bağımsız konuşma tespiti.

    feed(chunk) chunk'ı iç buffer'a ekler, biriken her 512 örneklik
    pencereyi modelden geçirir ve en son işlenen pencerenin ham
    konuşma olasılığını döndürür. O feed() çağrısında üretilen
    SPEECH_START/SPEECH_END olayları events özelliğinden okunur.
    """

    def __init__(
        self,
        threshold: float = None,
        min_speech_ms: float = None,
        min_silence_ms: float = None,
    ):
        self.threshold = threshold if threshold is not None else SETTINGS.vad_threshold
        self.min_speech_ms = min_speech_ms if min_speech_ms is not None else SETTINGS.vad_min_speech_ms
        self.min_silence_ms = min_silence_ms if min_silence_ms is not None else SETTINGS.vad_min_silence_ms

        self._model = load_silero_vad()
        self.reset()

    def reset(self) -> None:
        """Modelin iç durumunu (LSTM state) ve akış zamanlayıcısını sıfırlar."""
        self._model.reset_states()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._stream_ms = 0.0
        self._in_speech = False
        self._above_since_ms = None
        self._below_since_ms = None
        self.last_probability = 0.0
        self._last_events = []

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def events(self) -> list:
        """En son feed() çağrısında üretilen olaylar (varsa)."""
        return self._last_events

    def feed(self, chunk: np.ndarray) -> float:
        """chunk'ı (herhangi uzunlukta, mono float32) işler. Buffer'da
        biriken her 512 örneklik pencere için modeli çalıştırır ve en
        son olasılığı döndürür. Hiçbir pencere tamamlanmadıysa önceki
        bilinen olasılık döner."""
        self._last_events = []

        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, chunk])

        while len(self._buffer) >= SILERO_WINDOW_SAMPLES:
            window = self._buffer[:SILERO_WINDOW_SAMPLES]
            self._buffer = self._buffer[SILERO_WINDOW_SAMPLES:]
            self._stream_ms += SILERO_WINDOW_SAMPLES / SILERO_SAMPLE_RATE * 1000

            probability = self._run_model(window)
            self.last_probability = probability
            self._update_state(probability, self._stream_ms)

        return self.last_probability

    def _run_model(self, window: np.ndarray) -> float:
        tensor = torch.from_numpy(window)
        with torch.no_grad():
            output = self._model(tensor, SILERO_SAMPLE_RATE)
        return float(output.item())

    def _update_state(self, probability: float, now_ms: float) -> None:
        if probability >= self.threshold:
            self._below_since_ms = None
            if self._above_since_ms is None:
                self._above_since_ms = now_ms
            if not self._in_speech and (now_ms - self._above_since_ms) >= self.min_speech_ms:
                self._in_speech = True
                self._last_events.append(VADEvent("SPEECH_START", now_ms))
        else:
            self._above_since_ms = None
            if self._below_since_ms is None:
                self._below_since_ms = now_ms
            if self._in_speech and (now_ms - self._below_since_ms) >= self.min_silence_ms:
                self._in_speech = False
                self._last_events.append(VADEvent("SPEECH_END", now_ms))


def _format_bar(probability: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(probability * width))))
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    from modules.audio_io import MicStream

    print("20 saniye dinleniyor; konuşup susarak çubuğu izleyin.")
    print(
        f"threshold={SETTINGS.vad_threshold}  "
        f"min_speech_ms={SETTINGS.vad_min_speech_ms}  "
        f"min_silence_ms={SETTINGS.vad_min_silence_ms}\n"
    )

    mic = MicStream()
    vad = VAD()
    mic.start()

    start = time.perf_counter()
    try:
        while time.perf_counter() - start < 20.0:
            chunk = mic.read_chunk(timeout=0.1)
            if chunk is None:
                continue

            probability = vad.feed(chunk)

            for event in vad.events:
                print(f"\n[{event.timestamp_ms:8.0f} ms] {event.kind}")

            state = "KONUŞMA  " if vad.in_speech else "sessizlik"
            bar = _format_bar(probability)
            print(f"\r{bar} {probability:.2f}  {state}", end="", flush=True)
    finally:
        mic.stop()
        print("\nVAD testi bitti.")


if __name__ == "__main__":
    main()
