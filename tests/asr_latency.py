"""
Whisper model boyutlarının (tiny/base/small) transcribe gecikmesini,
TTS ile üretilmiş farklı uzunluktaki Türkçe seslerle tamamen offline
(mikrofonsuz) ölçer.

Amaç: whisper'ın sabit bir "başlangıç maliyeti" mi yoksa ses süresiyle
orantılı bir maliyeti mi olduğunu görmek -- özet tablodaki "0.5s/4s
oranı" sütunu 1'e yakınsa (kısa ve uzun ses neredeyse aynı sürede
işleniyorsa) sabit maliyet hipotezi doğrulanmış olur.

Her model (cpu, int8, cpu_threads=4) için önce bir ısınma transcribe'ı
yapılır (ölçülmez), sonra her ses için 5 kez transcribe edilip ortalama
süre alınır. beam_size=1 ve condition_on_previous_text=False (ikincisi
ASR.transcribe_detailed() içinde zaten sabit) kullanılır -- bu, modeller
arası ham çözümleme hızını, beam genişliği/halüsinasyon önleme
farklarından bağımsız olarak karşılaştırmak içindir.

Çalıştırma:
    python tests/asr_latency.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS
from modules.asr import ASR
from modules.tts import TTS

SENTENCES = {
    "~0.5 s": "Evet.",
    "~1.5 s": "Saat üçte görüşürüz.",
    "~4 s": "Yarın saat üçte randevunuz var, lütfen zamanında gelin.",
}

MODELS = ["tiny", "base", "small"]
N_REPEATS = 5
BEAM_SIZE = 1

RESULTS_PATH = SETTINGS.base_dir / "results" / "asr_latency.txt"
_log_lines = []


def log(line: str = "") -> None:
    print(line)
    _log_lines.append(line)


def main() -> None:
    log("Ses örnekleri TTS ile üretiliyor...")
    tts = TTS()
    audio_samples = {}
    for label, text in SENTENCES.items():
        audio, sr = tts.synthesize(text)
        duration_s = len(audio) / sr if sr else 0.0
        audio_samples[label] = (audio, sr, duration_s)
        log(f'  [{label}] "{text}"  (gerçek süre: {duration_s:.2f} s)')
    log()

    results = {label: {} for label in SENTENCES}  # label -> model -> avg_ms

    for model_size in MODELS:
        log(f"=== {model_size} ===")
        asr = ASR(model_size=model_size, device="cpu", compute_type="int8", cpu_threads=4)
        asr.load()

        # Isınma: herhangi bir sesle tek transcribe, ölçülmez.
        warmup_audio, warmup_sr, _ = next(iter(audio_samples.values()))
        asr.transcribe_detailed(warmup_audio, warmup_sr, beam_size=BEAM_SIZE)

        for label, (audio, sr, duration_s) in audio_samples.items():
            durations_ms = []
            text = ""
            for _ in range(N_REPEATS):
                start = time.perf_counter()
                result = asr.transcribe_detailed(audio, sr, beam_size=BEAM_SIZE)
                durations_ms.append((time.perf_counter() - start) * 1000)
                text = result.text

            avg_ms = sum(durations_ms) / len(durations_ms)
            results[label][model_size] = avg_ms

            log(f'  [{label}] (ses {duration_s:.2f} s) ortalama {avg_ms:.0f} ms ({N_REPEATS}x)')
            log(f'    Metin: "{text}"')
        log()

    # --- Özet tablo ---
    model_col = 10
    col_width = 16
    ratio_col = 16

    labels = list(SENTENCES.keys())
    short_label, long_label = labels[0], labels[-1]

    header = (
        f"{'Model':<{model_col}}"
        + "".join(f"{label:<{col_width}}" for label in labels)
        + f"{'0.5s/4s oranı':<{ratio_col}}"
    )
    log("=" * len(header))
    log("ÖZET (ortalama transcribe süresi, ms)")
    log("=" * len(header))
    log(header)
    log("-" * len(header))

    for model_size in MODELS:
        row = f"{model_size:<{model_col}}"
        for label in labels:
            avg_ms = results[label].get(model_size)
            cell = f"{avg_ms:.0f} ms" if avg_ms is not None else "-"
            row += f"{cell:<{col_width}}"

        short_ms = results[short_label].get(model_size)
        long_ms = results[long_label].get(model_size)
        if short_ms is not None and long_ms:
            row += f"{short_ms / long_ms:<{ratio_col}.2f}"
        else:
            row += f"{'-':<{ratio_col}}"
        log(row)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
    log(f"\nSonuçlar kaydedildi: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
