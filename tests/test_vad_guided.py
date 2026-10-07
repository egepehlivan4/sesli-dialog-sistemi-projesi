"""
VAD'ı elle, yönlendirmeli şekilde test eder.

Canlı olasılık çubuğu ekranı çok hızlı akıttığı için burada onun
yerine: her adım için bekle -> geri say -> 5 (veya 3) saniye kaydet ->
o adımda kaç SPEECH_START/SPEECH_END oldu, olasılığın ortalaması ve
maksimumu ne, onu özetler. Sonunda hepsini bir tabloda gösterir ve
results/vad_test.txt'e kaydeder.

Çalıştırma:
    python tests/test_vad_guided.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS
from modules.audio_io import MicStream
from modules.vad import VAD

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "vad_test.txt"

STEPS = [
    {
        "key": "0",
        "instruction": "Hiç konuşmayın, tamamen sessiz kalın.",
        "duration_s": 3.0,
        "expected": "konuşma yok (arka plan gürültüsü ölçümü)",
    },
    {
        "key": "1",
        "instruction": 'Normal bir cümle söyleyin: "Merhaba, bugün hava çok güzel."',
        "duration_s": 5.0,
        "expected": "1 konuşma",
    },
    {
        "key": "2",
        "instruction": 'Ortasında yarım saniye durarak söyleyin: "Bugün... hava güzel."',
        "duration_s": 5.0,
        "expected": "1 konuşma",
    },
    {
        "key": "3",
        "instruction": "Sadece \"hı hı\" deyin, başka bir şey söylemeyin.",
        "duration_s": 5.0,
        "expected": "? (belirsiz, kısa/alçak ünlem eşiği aşmayabilir)",
    },
    {
        "key": "4",
        "instruction": "Hiç konuşmayın, masaya parmağınızla birkaç kez vurun.",
        "duration_s": 5.0,
        "expected": "0 konuşma",
    },
]


def countdown(seconds: int = 3) -> None:
    for i in range(seconds, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1.0)
    print("başla!")


def run_step(step: dict) -> dict:
    print("\n" + "=" * 70)
    print(f"ADIM {step['key']}: {step['instruction']}")
    print(f"Beklenen: {step['expected']}")
    input("Hazır olunca Enter'a basın...")

    countdown(3)

    mic = MicStream()
    vad = VAD()
    mic.start()

    probabilities = []
    events = []

    print("● kaydediliyor", flush=True)
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < step["duration_s"]:
            chunk = mic.read_chunk(timeout=0.1)
            if chunk is None:
                continue
            probability = vad.feed(chunk)
            probabilities.append(probability)
            events.extend(vad.events)
    finally:
        mic.stop()
    print("kayıt bitti.")

    speech_starts = sum(1 for e in events if e.kind == "SPEECH_START")
    speech_ends = sum(1 for e in events if e.kind == "SPEECH_END")
    avg_probability = float(np.mean(probabilities)) if probabilities else 0.0
    max_probability = float(np.max(probabilities)) if probabilities else 0.0

    print(
        f"-> SPEECH_START: {speech_starts}  SPEECH_END: {speech_ends}  "
        f"ort. olasılık: {avg_probability:.3f}  maks. olasılık: {max_probability:.3f}"
    )

    return {
        "key": step["key"],
        "instruction": step["instruction"],
        "expected": step["expected"],
        "speech_starts": speech_starts,
        "speech_ends": speech_ends,
        "avg_probability": avg_probability,
        "max_probability": max_probability,
    }


def format_table(results: list) -> str:
    row_fmt = "{:<5}{:<14}{:<12}{:<10}{:<10}{:<45}"
    lines = [
        row_fmt.format("Adım", "SPEECH_START", "SPEECH_END", "Ort.P", "Maks.P", "Beklenen"),
    ]
    lines.append("-" * len(lines[0]))
    for r in results:
        lines.append(
            row_fmt.format(
                r["key"],
                r["speech_starts"],
                r["speech_ends"],
                f"{r['avg_probability']:.3f}",
                f"{r['max_probability']:.3f}",
                r["expected"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    print("VAD yönlendirmeli test")
    print(
        f"threshold={SETTINGS.vad_threshold}  "
        f"min_speech_ms={SETTINGS.vad_min_speech_ms}  "
        f"min_silence_ms={SETTINGS.vad_min_silence_ms}"
    )

    results = [run_step(step) for step in STEPS]

    table = format_table(results)
    print("\n" + "=" * 70)
    print("ÖZET")
    print("=" * 70)
    print(table)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("VAD yönlendirmeli test sonuçları\n")
        f.write(
            f"threshold={SETTINGS.vad_threshold}  "
            f"min_speech_ms={SETTINGS.vad_min_speech_ms}  "
            f"min_silence_ms={SETTINGS.vad_min_silence_ms}\n\n"
        )
        for r in results:
            f.write(f"Adım {r['key']}: {r['instruction']}\n")
        f.write("\n")
        f.write(table + "\n")

    print(f"\nSonuçlar kaydedildi: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
