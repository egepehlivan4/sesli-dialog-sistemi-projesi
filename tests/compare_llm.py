"""
Farklı Ollama modellerini aynı sorularla karşılaştırır: ısınma süresi,
her soru için ilk cümlenin tamamlanma süresi ve token/saniye.

Modeller config.py'deki LLM_COMPARE_MODELS'ten okunur (varsayılan:
qwen2.5:3b, qwen2.5:7b, gemma3:4b). Çalıştırmadan önce her model için
'ollama pull <model>' ile modelin kurulu olduğundan emin olun.

Çalıştırma:
    python tests/compare_llm.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS
from modules.llm import LLM, SENTENCE_END_RE

QUESTIONS = [
    "Yarın saat kaçta açıksınız?",
    "Randevumu iptal etmek istiyorum.",
    "Bu hafta sonu hava nasıl olacak?",
    "Bana kısa bir tavsiye ver.",
    "Türkiye'nin başkenti neresi?",
]


def run_question(llm: LLM, question: str) -> dict:
    messages = [{"role": "user", "content": question}]

    request_start = time.perf_counter()
    first_token_time = None
    first_sentence_time = None
    token_count = 0
    full_text = ""
    buffer = ""

    for token in llm.stream_tokens(messages):
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        token_count += 1
        full_text += token
        buffer += token
        if first_sentence_time is None and SENTENCE_END_RE.search(buffer):
            first_sentence_time = now

    end_time = time.perf_counter()

    first_sentence_ms = (
        (first_sentence_time - request_start) * 1000 if first_sentence_time is not None else None
    )
    decode_duration = (end_time - first_token_time) if first_token_time else 0.0
    tokens_per_sec = (
        (token_count - 1) / decode_duration if decode_duration > 0 and token_count > 1 else 0.0
    )

    return {
        "question": question,
        "answer": full_text.strip(),
        "first_sentence_ms": first_sentence_ms,
        "tokens_per_sec": tokens_per_sec,
        "token_count": token_count,
    }


def main() -> None:
    models = SETTINGS.llm_compare_models
    print(f"Karşılaştırılacak modeller: {', '.join(models)}\n")

    all_results = {}

    for model in models:
        print(f"=== {model} ===")
        llm = LLM(model=model)

        print("Isınma isteği gönderiliyor...")
        try:
            warmup_elapsed = llm.warmup()
        except RuntimeError as exc:
            print(f"ATLANDI: {exc}\n")
            continue
        print(f"Isınma tamamlandı ({warmup_elapsed * 1000:.0f} ms)\n")

        results = []
        for question in QUESTIONS:
            print(f"Soru: {question}")
            try:
                result = run_question(llm, question)
            except RuntimeError as exc:
                print(f"  HATA: {exc}\n")
                continue
            print(f"  Cevap           : {result['answer']}")
            if result["first_sentence_ms"] is not None:
                print(f"  İlk cümle süresi: {result['first_sentence_ms']:.0f} ms")
            else:
                print("  İlk cümle süresi: -")
            print(f"  Token/saniye    : {result['tokens_per_sec']:.1f}\n")
            results.append(result)

        if results:
            all_results[model] = results
        print()

    if not all_results:
        print("Hiçbir model için sonuç alınamadı.")
        return

    # --- Özet tablo ---
    col_width = 24
    question_width = 36

    print("=" * (question_width + col_width * len(all_results)))
    print("ÖZET  (ilk cümle süresi / token/saniye)")
    print("=" * (question_width + col_width * len(all_results)))

    header = f"{'Soru':<{question_width}}" + "".join(
        f"{model:<{col_width}}" for model in all_results
    )
    print(header)
    print("-" * len(header))

    for i, question in enumerate(QUESTIONS):
        row = f"{question:<{question_width}}"
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
        print(row)

    print("-" * len(header))
    avg_row = f"{'ORTALAMA':<{question_width}}"
    for model in all_results:
        results = all_results[model]
        valid_fs = [r["first_sentence_ms"] for r in results if r["first_sentence_ms"] is not None]
        valid_tps = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]
        avg_fs = sum(valid_fs) / len(valid_fs) if valid_fs else 0.0
        avg_tps = sum(valid_tps) / len(valid_tps) if valid_tps else 0.0
        cell = f"{avg_fs:.0f}ms / {avg_tps:.1f}t/s"
        avg_row += f"{cell:<{col_width}}"
    print(avg_row)


if __name__ == "__main__":
    main()
