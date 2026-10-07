"""
Donanım testi: mikrofonun çalıştığını ve işletim sisteminin mikrofon
iznini verdiğini doğrular. Henüz VAD/ASR/pipeline'a dokunmaz.

Çalıştırma:
    python tests/test_mic.py
"""

import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

DURATION_S = 3.0
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "test_mic.wav"
SILENCE_THRESHOLD = 0.001


def record() -> np.ndarray:
    print("Kullanılabilir ses cihazları:")
    print(sd.query_devices())
    print()

    device = SETTINGS.input_device
    print(f"Kayıt cihazı: {device if device else '(sistem varsayılanı)'}")
    print(f"{DURATION_S:.0f} saniye kayıt başlıyor, konuşun...")

    recording = sd.rec(
        int(DURATION_S * SETTINGS.sample_rate),
        samplerate=SETTINGS.sample_rate,
        channels=SETTINGS.channels,
        dtype=SETTINGS.audio_dtype,
        device=device,
    )
    sd.wait()
    print("Kayıt tamamlandı.")
    return recording


def save_wav(recording: np.ndarray, path: Path) -> None:
    clipped = np.clip(recording, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(SETTINGS.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SETTINGS.sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    print(f"Kayıt dosyaya yazıldı: {path}")


def check_permission(max_amplitude: float) -> None:
    if max_amplitude >= SILENCE_THRESHOLD:
        return
    print()
    print("UYARI: MİKROFON İZNİ YOK (ya da mikrofon sessiz/bağlı değil)")
    print(f"Maksimum genlik {max_amplitude:.6f}, eşik {SILENCE_THRESHOLD}'in altında.")
    print()
    print("macOS icin: System Settings > Privacy & Security > Microphone")
    print("  bolumunden terminal/IDE uygulamasina mikrofon izni verin.")
    print("Windows icin: Ayarlar > Gizlilik ve guvenlik > Mikrofon")
    print("  bolumunden uygulamanin mikrofona erisimine izin verin.")


def main() -> None:
    recording = record()
    max_amplitude = float(np.max(np.abs(recording)))
    print(f"Maksimum genlik: {max_amplitude:.6f}")

    save_wav(recording, OUTPUT_PATH)
    check_permission(max_amplitude)


if __name__ == "__main__":
    main()
