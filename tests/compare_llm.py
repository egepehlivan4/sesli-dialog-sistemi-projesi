"""
Farklı Ollama modellerini aynı sorularla karşılaştırır: ısınma süresi,
her soru için TTFT (ilk token süresi), ilk cümlenin tamamlanma süresi
ve token/saniye.

Modeller config.py'deki LLM_COMPARE_MODELS'ten okunur (varsayılan:
qwen2.5:3b, qwen2.5:7b, gemma3:4b). Son soru bilerek önceki bir
konuşma turuyla (bağlamla) birlikte soruluyor; ayrıca gemma3:4b bir kez
de "bilgi kartı" içeren genişletilmiş bir sistem promptuyla (uydurma
testi) ayrı bir sütun olarak koşuluyor.

Çalıştırmadan önce her model için 'ollama pull <model>' ile modelin
kurulu olduğundan emin olun.

Tüm çıktı (cevaplar + özet tablo) results/llm_comparison.txt dosyasına
da yazılır.

Çalıştırma:
    python tests/compare_llm.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS
from modules.llm import LLM, SENTENCE_END_RE, SYSTEM_PROMPT

QUESTIONS = [
    "Yarın saat kaçta açıksınız?",
    "Randevumu iptal etmek istiyorum.",
    "Bu hafta sonu hava nasıl olacak?",
    "Bana kısa bir tavsiye ver.",
    "Türkiye'nin başkenti neresi?",
]

# Son soru bilerek bağlamla (önceki tur) birlikte soruluyor; modelin
# "pazartesi"yi hatırlayıp düzeltmeyi doğru anlayıp anlamadığını ölçer.
CONTEXT_LABEL = "Dur, yanlış anladın, ben salıyı kastettim."
CONTEXT_MESSAGES = [
    {"role": "user", "content": "Pazartesi saat 10'a randevu almak istiyorum."},
    {"role": "assistant", "content": "Randevunuz pazartesi saat 10:00 için oluşturuldu."},
    {"role": "user", "content": CONTEXT_LABEL},
]

# Uydurma testi: gemma3:4b'yi bilgi kartı içeren genişletilmiş bir sistem
# promptuyla ayrıca koş (bilmediği şeyi uydurmak yerine kabul edip
# etmediğini görmek için).
INFO_CARD_MODEL = "gemma3:4b"
INFO_CARD_LABEL = "gemma3:4b + bilgi kartı"
INFO_CARD_SYSTEM_PROMPT = SYSTEM_PROMPT + " " + (
    "Sen bir randevu asistanısın. Bilgiler: Çalışma saatleri hafta içi "
    "09:00-18:00, hafta sonu kapalı. Randevular telefonla veya buradan "
    "alınır ve iptal edilebilir. Hava durumu ve genel bilgi sorularına "
    "erişimin yok; bu durumda kibarca bunu söyle."
)

RESULTS_PATH = SETTINGS.base_dir / "results" / "llm_comparison.txt"
_log_lines = []


def log(line: str = "") -> None:
    print(line)
    _log_lines.append(line)


def question_entries() -> list:
    """(etiket, mesajlar) çiftleri. İlk 5 soru tek turlu; son soru önceki
    bir konuşma turuyla (çok turlu) soruluyor."""
    entries = [(q, [{"role": "user", "content": q}]) for q in QUESTIONS]
    entries.append((CONTEXT_LABEL, CONTEXT_MESSAGES))
    return entries


def run_question(llm: LLM, messages: list) -> dict:
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
        # stream bittiyse (ör. "...Ankara." ile biten cevap), kalan metin
        # stream bittiği anda fiilen tamamlanmış sayılır. Bu olmadan
        # first_sentence_time hiç set edilmez ve ortalamalar yanlış
        # çıkar (bkz. gemma3 tek cümlelik cevaplar).
        first_sentence_time = end_time

    first_token_ms = (
        (first_token_time - request_start) * 1000 if first_token_time is not None else None
    )
    first_sentence_ms = (
        (first_sentence_time - request_start) * 1000 if first_sentence_time is not None else None
    )
    decode_duration = (end_time - first_token_time) if first_token_time else 0.0
    tokens_per_sec = (
        (token_count - 1) / decode_duration if decode_duration > 0 and token_count > 1 else 0.0
    )

    return {
        "answer": full_text.strip(),
        "first_token_ms": first_token_ms,
        "first_sentence_ms": first_sentence_ms,
        "tokens_per_sec": tokens_per_sec,
        "token_count": token_count,
    }


def run_model(label: str, model: str, entries: list, system_prompt: str = None) -> list:
    """Bir modeli ısıtır, tüm soruları sorar, sonuç listesini döner.
    Isınma başarısız olursa (model kurulu değil vb.) boş liste döner."""
    log(f"=== {label} ===")
    llm = LLM(model=model, system_prompt=system_prompt)

    log("Isınma isteği gönderiliyor...")
    try:
        warmup_elapsed = llm.warmup()
    except RuntimeError as exc:
        log(f"ATLANDI: {exc}\n")
        return []
    log(f"Isınma tamamlandı ({warmup_elapsed * 1000:.0f} ms)\n")

    results = []
    for q_label, messages in entries:
        log(f"Soru: {q_label}")
        try:
            result = run_question(llm, messages)
        except RuntimeError as exc:
            log(f"  HATA: {exc}\n")
            continue
        log(f"  Cevap           : {result['answer']}")
        ttft = f"{result['first_token_ms']:.0f} ms" if result["first_token_ms"] is not None else "-"
        fs = f"{result['first_sentence_ms']:.0f} ms" if result["first_sentence_ms"] is not None else "-"
        log(f"  TTFT            : {ttft}")
        log(f"  İlk cümle süresi: {fs}")
        log(f"  Token/saniye    : {result['tokens_per_sec']:.1f}\n")
        results.append(result)

    log()
    return results


def main() -> None:
    models = SETTINGS.llm_compare_models
    entries = question_entries()
    log(f"Karşılaştırılacak modeller: {', '.join(models)}\n")

    all_results = {}
    for model in models:
        results = run_model(model, model, entries)
        if results:
            all_results[model] = results

    info_card_results = run_model(
        INFO_CARD_LABEL, INFO_CARD_MODEL, entries, system_prompt=INFO_CARD_SYSTEM_PROMPT
    )
    if info_card_results:
        all_results[INFO_CARD_LABEL] = info_card_results

    if not all_results:
        log("Hiçbir model için sonuç alınamadı.")
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
        return

    # --- Özet tablo ---
    col_width = 24
    question_width = 36

    log("=" * (question_width + col_width * len(all_results)))
    log("ÖZET  (ilk cümle süresi / token/saniye)")
    log("=" * (question_width + col_width * len(all_results)))

    header = f"{'Soru':<{question_width}}" + "".join(
        f"{model:<{col_width}}" for model in all_results
    )
    log(header)
    log("-" * len(header))

    for i, (q_label, _messages) in enumerate(entries):
        row = f"{q_label:<{question_width}}"
        for model in all_results:
            if i < len(all_results[model]):
                result = all_results[model][i]
                if result["first_sentence_ms"] is not None:
                    cell = f"{result['first_sentence_ms']:.0f}ms / {result['tokens_per_sec']:.1f}t/s"
                else:
                    cell = "-"
            else:
                cell = "-"
            row += f"{cell:<{col_width}}"
        log(row)

    log("-" * len(header))
    avg_row = f"{'ORTALAMA':<{question_width}}"
    for model in all_results:
        results = all_results[model]
        valid_fs = [r["first_sentence_ms"] for r in results if r["first_sentence_ms"] is not None]
        valid_tps = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]
        if valid_fs:
            avg_tps = sum(valid_tps) / len(valid_tps) if valid_tps else 0.0
            cell = f"{sum(valid_fs) / len(valid_fs):.0f}ms / {avg_tps:.1f}t/s"
        else:
            cell = "-"
        avg_row += f"{cell:<{col_width}}"
    log(avg_row)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
    log(f"\nSonuçlar kaydedildi: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
