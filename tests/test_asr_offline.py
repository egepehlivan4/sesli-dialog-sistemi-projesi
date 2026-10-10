"""
Eski, streaming olmayan TTS -> ASR entegrasyon testi: tek bir cümleyi
TTS ile üretip ASR.transcribe() ile tek seferde (offline) çözer, RTF
ölçer. modules/asr.py'deki canlı StreamingASR testiyle karşılaştırmak
için ayrı tutulur.

Çalıştırma:
    python tests/test_asr_offline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.asr import ASR
from modules.tts import TTS


def main() -> None:
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
