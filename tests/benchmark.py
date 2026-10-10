"""
Tüm modüllerin (TTS, ASR, LLM, SpeakerOut) performansını tek seferde
ölçer ve makine bilgisiyle birlikte results/benchmark_<makine>.txt
dosyasına kaydeder.

Her modül ölçümden önce bir kez ısıtılır (ilk çağrının soğuk başlangıç
süresi -model yükleme, ilk GPU transferi vb.- ortalamaya karışmasın diye).

Önkoşullar:
- Ollama sunucusu çalışıyor olmalı ve config'teki LLM_MODEL kurulu olmalı.
- Piper TTS sesi ve faster-whisper modeli kurulu olmalı.

Çalıştırma:
    python tests/benchmark.py
"""

import platform
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS
from modules.asr import ASR
from modules.audio_io import SpeakerOut
from modules.llm import LLM, SENTENCE_END_RE
from modules.tts import TTS

TTS_SENTENCES = {
    "kısa": "Merhaba, nasılsın?",
    "orta": "Yarın öğleden sonra saat üçte randevunuz var, lütfen zamanında gelin.",
    "uzun": (
        "Bu cümle bilerek oldukça uzun yazıldı, çünkü hem metin-ses "
        "dönüştürücünün hem de konuşma tanımanın daha gerçekçi bir konuşma "
        "süresi üzerinden, kesme testine de yetecek kadar uzun bir ses ile "
        "ölçülmesini istiyoruz."
    ),
}

LLM_QUESTIONS = [
    "Yarın saat kaçta açıksınız?",
    "Randevumu iptal etmek istiyorum.",
    "Bu hafta sonu hava nasıl olacak?",
    "Bana kısa bir tavsiye ver.",
    "Türkiye'nin başkenti neresi?",
]

N_REPEATS = 5
N_CUTOFF_REPEATS = 10
CUTOFF_OFFSET_S = 0.3

RESULTS_DIR = SETTINGS.base_dir / "results"

_log_lines = []


def log(line: str = "") -> None:
    print(line)
    _log_lines.append(line)


def _normalize_words(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Basit kelime düzeyinde WER: kelime dizileri arası edit distance /
    referanstaki kelime sayısı. jiwer gibi ek bir paket gerektirmez."""
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def machine_info() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def machine_slug() -> str:
    node = platform.node() or "unknown"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", node).strip("_")
    return slug or "unknown"


def benchmark_tts(tts: TTS) -> dict:
    log("=== TTS ===")
    results = {}
    for label, text in TTS_SENTENCES.items():
        tts.synthesize(text)  # ısınma, ölçülmez

        rtfs = []
        audio_sample, sr_sample, duration_sample = None, None, None
        for _ in range(N_REPEATS):
            start = time.perf_counter()
            audio, sr = tts.synthesize(text)
            elapsed = time.perf_counter() - start
            duration = len(audio) / sr if sr else 0.0
            rtf = elapsed / duration if duration > 0 else float("inf")
            rtfs.append(rtf)
            if audio_sample is None:
                audio_sample, sr_sample, duration_sample = audio, sr, duration

        avg_rtf = sum(rtfs) / len(rtfs)
        log(f"[{label}] \"{text}\"")
        log(f"  Ses süresi          : {duration_sample:.2f} s")
        log(f"  Ortalama RTF ({N_REPEATS}x) : {avg_rtf:.3f}")

        results[label] = {
            "text": text,
            "avg_rtf": avg_rtf,
            "audio": audio_sample,
            "sample_rate": sr_sample,
        }
    log()
    return results


def benchmark_asr(asr: ASR, tts_results: dict) -> dict:
    log("=== ASR ===")
    asr.load()  # model yükleme süresi ayrı yazdırılır, ölçüme karışmaz

    results = {}
    warmed_up = False
    for label, info in tts_results.items():
        audio, sr, text = info["audio"], info["sample_rate"], info["text"]
        audio_duration = len(audio) / sr if sr else 0.0

        if not warmed_up:
            asr.transcribe(audio, sr)  # ısınma, ölçülmez
            warmed_up = True

        rtfs = []
        transcribed = ""
        for _ in range(N_REPEATS):
            start = time.perf_counter()
            transcribed = asr.transcribe(audio, sr)
            elapsed = time.perf_counter() - start
            rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
            rtfs.append(rtf)

        avg_rtf = sum(rtfs) / len(rtfs)
        wer = word_error_rate(text, transcribed)

        log(f"[{label}]")
        log(f"  Orijinal metin      : {text}")
        log(f"  ASR çıktısı         : {transcribed}")
        log(f"  Ortalama RTF ({N_REPEATS}x) : {avg_rtf:.3f}")
        log(f"  WER                 : {wer * 100:.1f}%")

        results[label] = {"avg_rtf": avg_rtf, "wer": wer}
    log()
    return results


def benchmark_llm() -> dict:
    log("=== LLM ===")
    llm = LLM()
    log(f"Model: {llm.model}")

    log("Isınma isteği gönderiliyor...")
    warmup_elapsed = llm.warmup()
    log(f"Isınma tamamlandı ({warmup_elapsed * 1000:.0f} ms)\n")

    first_token_times = []
    first_sentence_times = []
    tokens_per_sec_list = []

    for question in LLM_QUESTIONS:
        messages = [{"role": "user", "content": question}]
        request_start = time.perf_counter()
        first_token_time = None
        first_sentence_time = None
        token_count = 0
        full_text = ""

        for token in llm.stream_tokens(messages):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            token_count += 1
            full_text += token
            if first_sentence_time is None and SENTENCE_END_RE.search(full_text):
                first_sentence_time = now

        end_time = time.perf_counter()

        if first_sentence_time is None and first_token_time is not None:
            # Cevap tek cümleyse ya da son cümleden sonra boşluk gelmeden
            # stream bittiyse, kalan metin stream bittiği anda fiilen
            # tamamlanmış sayılır (bkz. tests/compare_llm.py).
            first_sentence_time = end_time

        if first_token_time is not None:
            first_token_times.append((first_token_time - request_start) * 1000)
        if first_sentence_time is not None:
            first_sentence_times.append((first_sentence_time - request_start) * 1000)
        decode_duration = (end_time - first_token_time) if first_token_time else 0.0
        if decode_duration > 0 and token_count > 1:
            tokens_per_sec_list.append((token_count - 1) / decode_duration)

        log(f"Soru: {question}")
        log(f"  Cevap: {full_text.strip()}")

    avg_first_token = sum(first_token_times) / len(first_token_times) if first_token_times else 0.0
    avg_first_sentence = (
        sum(first_sentence_times) / len(first_sentence_times) if first_sentence_times else 0.0
    )
    avg_tps = sum(tokens_per_sec_list) / len(tokens_per_sec_list) if tokens_per_sec_list else 0.0

    log(f"\nOrtalama TTFT ({len(LLM_QUESTIONS)} soru)              : {avg_first_token:.0f} ms")
    log(f"Ortalama ilk cümle süresi ({len(LLM_QUESTIONS)} soru) : {avg_first_sentence:.0f} ms")
    log(f"Ortalama token/saniye                        : {avg_tps:.1f}")
    log()

    return {
        "avg_first_token_ms": avg_first_token,
        "avg_first_sentence_ms": avg_first_sentence,
        "avg_tokens_per_sec": avg_tps,
    }


def benchmark_speaker(tts_results: dict) -> dict:
    log("=== SpeakerOut (kesme testi) ===")
    audio, sr = tts_results["uzun"]["audio"], tts_results["uzun"]["sample_rate"]
    duration = len(audio) / sr if sr else 0.0

    cutoff_offset = min(CUTOFF_OFFSET_S, duration / 3) if duration > 0 else 0.0
    if cutoff_offset <= 0:
        raise RuntimeError("Kesme testi için 'uzun' ses örneği çok kısa.")

    speaker = SpeakerOut()
    try:
        # Isınma: bir kez çal + durdur, ölçülmez.
        speaker.play(audio, sr)
        time.sleep(cutoff_offset)
        speaker.stop()
        while speaker.last_silence_at is None:
            time.sleep(0.001)

        latencies = []
        for _ in range(N_CUTOFF_REPEATS):
            speaker.play(audio, sr)
            time.sleep(cutoff_offset)
            stop_call_time = time.perf_counter()
            speaker.stop()
            while speaker.last_silence_at is None or speaker.last_silence_at < stop_call_time:
                time.sleep(0.001)
            latencies.append((speaker.last_silence_at - stop_call_time) * 1000)
    finally:
        speaker.close()

    avg_latency = sum(latencies) / len(latencies)
    worst_latency = max(latencies)

    log(f"Ses süresi        : {duration:.2f} s, kesme noktası: {cutoff_offset * 1000:.0f} ms")
    log(f"Ortalama kesme gecikmesi ({N_CUTOFF_REPEATS}x) : {avg_latency:.2f} ms")
    log(f"En kötü kesme gecikmesi               : {worst_latency:.2f} ms")
    log()

    return {"avg_latency_ms": avg_latency, "worst_latency_ms": worst_latency}


def main() -> None:
    info = machine_info()
    log("=" * 60)
    log("BENCHMARK")
    log("=" * 60)
    log(f"İşletim sistemi : {info['os']}")
    log(f"Mimari          : {info['machine']}")
    log(f"İşlemci         : {info['processor']}")
    log(f"Python          : {info['python']}")
    log("")

    tts_results = benchmark_tts(TTS())
    asr_results = benchmark_asr(ASR(), tts_results)
    llm_results = benchmark_llm()
    speaker_results = benchmark_speaker(tts_results)

    # --- Özet tablo ---
    col1, col2, col3 = 18, 26, 14
    header = f"{'Modül':<{col1}}{'Metrik':<{col2}}{'Değer':<{col3}}"

    log("=" * len(header))
    log("ÖZET")
    log("=" * len(header))
    log(header)
    log("-" * len(header))

    for label in TTS_SENTENCES:
        log(f"{'TTS (' + label + ')':<{col1}}{'ort. RTF':<{col2}}{tts_results[label]['avg_rtf']:<{col3}.3f}")
    for label in TTS_SENTENCES:
        log(f"{'ASR (' + label + ')':<{col1}}{'ort. RTF':<{col2}}{asr_results[label]['avg_rtf']:<{col3}.3f}")
        log(f"{'ASR (' + label + ')':<{col1}}{'WER':<{col2}}{asr_results[label]['wer'] * 100:<{col3}.1f}%")
    log(f"{'LLM':<{col1}}{'ort. TTFT':<{col2}}{llm_results['avg_first_token_ms']:<{col3}.0f} ms")
    log(f"{'LLM':<{col1}}{'ort. ilk cümle süresi':<{col2}}{llm_results['avg_first_sentence_ms']:<{col3}.0f} ms")
    log(f"{'LLM':<{col1}}{'ort. token/s':<{col2}}{llm_results['avg_tokens_per_sec']:<{col3}.1f}")
    log(f"{'SpeakerOut':<{col1}}{'ort. kesme gecikmesi':<{col2}}{speaker_results['avg_latency_ms']:<{col3}.2f} ms")
    log(f"{'SpeakerOut':<{col1}}{'en kötü kesme gecikmesi':<{col2}}{speaker_results['worst_latency_ms']:<{col3}.2f} ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"benchmark_{machine_slug()}.txt"
    out_path.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
    log(f"\nSonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
