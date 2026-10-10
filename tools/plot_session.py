"""
Bir oturumun results/sessions/<zaman>.jsonl dosyasını okuyup zaman
çizelgesi PNG'si üretir: üstte kullanıcı konuşma aralıkları (VAD
START-END, üzerinde final metin), altta asistan konuşma aralıkları
(TTS başla-bitir), kesintiler kırmızı dikey çizgi, arka planda
LISTEN/THINKING/SPEAK durum bantları.

Okunabilirlik için oturum satırlara bölünür (--row-seconds, varsayılan
20 s); satırlar alt alta, tek bir ortak lejant ile. --start/--end
(saniye) ile belirli bir aralığa yakınlaşılabilir.

Varsayılan olarak results/sessions/ içindeki EN SON oturumu çizer
(dosya adları %Y%m%d_%H%M%S olduğu için alfabetik sıralama = zaman
sıralaması). Belirli bir oturum için dosya adı/yolu argüman olarak
verilebilir. Çıktı varsayılan olarak results/sessions/<zaman>.png;
--out ile değiştirilebilir.

Çalıştırma:
    python tools/plot_session.py                              # en son oturum
    python tools/plot_session.py 20261010_173408                # belirli oturum
    python tools/plot_session.py --row-seconds 10
    python tools/plot_session.py --start 55 --end 70 --out results/sessions/20261010_173408_hihi.png
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

STATE_COLORS = {
    "LISTEN": "#9ecae1",
    "THINKING": "#fdd0a2",
    "SPEAK": "#a1d99b",
}
USER_COLOR = "#3182bd"
ASSISTANT_COLOR = "#31a354"
INTERRUPT_COLOR = "#e6194b"

USER_BAND = (2.2, 0.8)  # (y, height) broken_barh için
ASSISTANT_BAND = (0.8, 0.8)
ROW_YLIM = (0.0, USER_BAND[0] + USER_BAND[1] + 0.9)

FIG_WIDTH_IN = 12.0  # A4'e sığacak genişlik
ROW_HEIGHT_IN = 1.5
DPI = 200

CHARS_PER_SECOND = 7  # bar genişliğine göre metin kısaltma kaba tahmini


def resolve_session_path(arg: str = None) -> Path:
    sessions_dir = SETTINGS.base_dir / "results" / "sessions"

    if arg is None:
        candidates = sorted(sessions_dir.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"{sessions_dir} içinde hiç oturum (.jsonl) bulunamadı.")
        return candidates[-1]

    path = Path(arg)
    if path.exists():
        return path
    candidate = sessions_dir / arg
    if candidate.exists():
        return candidate
    candidate = sessions_dir / f"{arg}.jsonl"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Oturum bulunamadı: {arg!r} (aranan: {path}, {candidate})")


def load_events(path: Path) -> list:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def build_timeline(events: list) -> dict:
    """Olay akışını çizim için gereken yapılara ayrıştırır:
    user_intervals, assistant_intervals, interrupts, state_segments,
    end_time."""
    user_intervals = []
    assistant_intervals = []
    interrupts = []
    state_segments = []

    open_user_start = None
    last_user_interval = None
    open_assistant_start = None
    open_assistant_text = None
    cur_state = "LISTEN"
    state_since = 0.0
    end_time = 0.0
    strategy = None

    for rec in events:
        t = rec.get("elapsed_s", 0.0)
        end_time = max(end_time, t)
        event = rec.get("event")

        if event == "session_start":
            strategy = rec.get("strategy")

        elif event == "vad_event":
            if rec.get("kind") == "SPEECH_START":
                open_user_start = t
            elif rec.get("kind") == "SPEECH_END" and open_user_start is not None:
                interval = {"start": open_user_start, "end": t, "text": None}
                user_intervals.append(interval)
                last_user_interval = interval
                open_user_start = None

        elif event == "asr_final" and last_user_interval is not None and last_user_interval["text"] is None:
            last_user_interval["text"] = rec.get("text", "")

        elif event == "tts_start":
            open_assistant_start = t
            open_assistant_text = rec.get("text", "")
        elif event == "tts_end" and open_assistant_start is not None:
            assistant_intervals.append({"start": open_assistant_start, "end": t, "text": open_assistant_text})
            open_assistant_start = None
            open_assistant_text = None

        elif event == "interrupt":
            interrupts.append({"t": t, "cutoff_ms": rec.get("cutoff_ms", 0.0)})

        elif event == "state_change":
            state_segments.append({"start": state_since, "end": t, "state": cur_state})
            cur_state = rec.get("to", cur_state)
            state_since = t

    state_segments.append({"start": state_since, "end": end_time, "state": cur_state})

    return {
        "user_intervals": user_intervals,
        "assistant_intervals": assistant_intervals,
        "interrupts": interrupts,
        "state_segments": state_segments,
        "end_time": end_time,
        "strategy": strategy,
    }


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def _label_for_bar(text: str, duration_s: float) -> str:
    if not text:
        return None
    max_len = int(duration_s * CHARS_PER_SECOND)
    if max_len < 3:
        return None
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def plot_timeline(
    timeline: dict,
    title: str,
    out_path: Path,
    row_seconds: float = 20.0,
    start: float = None,
    end: float = None,
) -> None:
    window_start = 0.0 if start is None else start
    window_end = timeline["end_time"] if end is None else end
    if window_end <= window_start:
        raise ValueError(f"Geçersiz aralık: start={window_start} >= end={window_end}")

    n_rows = max(1, math.ceil((window_end - window_start) / row_seconds))
    rows = [
        (window_start + i * row_seconds, min(window_start + (i + 1) * row_seconds, window_end))
        for i in range(n_rows)
    ]

    fig, axes = plt.subplots(n_rows, 1, figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * n_rows + 0.6))
    if n_rows == 1:
        axes = [axes]

    for ax, (row_start, row_end) in zip(axes, rows):
        for seg in timeline["state_segments"]:
            if _overlaps(seg["start"], seg["end"], row_start, row_end):
                ax.axvspan(
                    seg["start"], seg["end"], color=STATE_COLORS.get(seg["state"], "#dddddd"), alpha=0.35, zorder=0
                )

        row_user = [iv for iv in timeline["user_intervals"] if _overlaps(iv["start"], iv["end"], row_start, row_end)]
        ax.broken_barh(
            [(iv["start"], iv["end"] - iv["start"]) for iv in row_user],
            USER_BAND,
            facecolors=USER_COLOR,
            edgecolors="black",
            linewidth=0.5,
            zorder=2,
        )
        for iv in row_user:
            label = _label_for_bar(iv["text"], iv["end"] - iv["start"])
            if label:
                ax.text(
                    max(iv["start"], row_start),
                    USER_BAND[0] + USER_BAND[1] + 0.08,
                    label,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    zorder=3,
                )

        row_assistant = [
            iv for iv in timeline["assistant_intervals"] if _overlaps(iv["start"], iv["end"], row_start, row_end)
        ]
        ax.broken_barh(
            [(iv["start"], iv["end"] - iv["start"]) for iv in row_assistant],
            ASSISTANT_BAND,
            facecolors=ASSISTANT_COLOR,
            edgecolors="black",
            linewidth=0.5,
            zorder=2,
        )
        for iv in row_assistant:
            label = _label_for_bar(iv.get("text"), iv["end"] - iv["start"])
            if label:
                ax.text(
                    max(iv["start"], row_start),
                    ASSISTANT_BAND[0] + ASSISTANT_BAND[1] + 0.08,
                    label,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    zorder=3,
                )

        for intr in timeline["interrupts"]:
            if row_start <= intr["t"] <= row_end:
                ax.axvline(x=intr["t"], color=INTERRUPT_COLOR, linewidth=1.5, zorder=4)

        ax.set_xlim(row_start, row_end)
        ax.set_ylim(*ROW_YLIM)
        ax.set_yticks([ASSISTANT_BAND[0] + ASSISTANT_BAND[1] / 2, USER_BAND[0] + USER_BAND[1] / 2])
        ax.set_yticklabels(["Asistan", "Kullanıcı"], fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    axes[-1].set_xlabel("Zaman (s)", fontsize=9)
    fig.suptitle(title, fontsize=10)

    legend_handles = [Patch(facecolor=color, alpha=0.35, label=state) for state, color in STATE_COLORS.items()]
    legend_handles.append(Patch(facecolor=USER_COLOR, label="Kullanıcı konuşuyor"))
    legend_handles.append(Patch(facecolor=ASSISTANT_COLOR, label="Asistan konuşuyor"))
    legend_handles.append(Patch(facecolor=INTERRUPT_COLOR, label="Kesinti"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False)

    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _default_title(session_path: Path, strategy: str) -> str:
    try:
        dt = datetime.strptime(session_path.stem, "%Y%m%d_%H%M%S")
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        date_str = session_path.stem
    return f"Oturum {date_str} — strateji: {strategy or '?'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Oturum JSONL'inden zaman çizelgesi PNG'si üretir.")
    parser.add_argument("session", nargs="?", default=None, help="Oturum adı/yolu (verilmezse en son oturum).")
    parser.add_argument("--row-seconds", type=float, default=20.0, help="Satır başına gösterilen süre (s).")
    parser.add_argument("--start", type=float, default=None, help="Çizilecek aralığın başlangıcı (s).")
    parser.add_argument("--end", type=float, default=None, help="Çizilecek aralığın sonu (s).")
    parser.add_argument("--out", type=str, default=None, help="Çıktı PNG yolu (verilmezse <oturum>.png).")
    parser.add_argument(
        "--title", type=str, default=None, help="Grafik başlığı (verilmezse 'Oturum <tarih saat> — strateji: ...')."
    )
    args = parser.parse_args()

    session_path = resolve_session_path(args.session)
    events = load_events(session_path)
    timeline = build_timeline(events)

    out_path = Path(args.out) if args.out else session_path.with_suffix(".png")
    title = args.title or _default_title(session_path, timeline["strategy"])
    plot_timeline(
        timeline,
        title=title,
        out_path=out_path,
        row_seconds=args.row_seconds,
        start=args.start,
        end=args.end,
    )

    print(f"Oturum: {session_path}")
    print(f"Kullanıcı aralığı sayısı : {len(timeline['user_intervals'])}")
    print(f"Asistan aralığı sayısı   : {len(timeline['assistant_intervals'])}")
    print(f"Kesinti sayısı           : {len(timeline['interrupts'])}")
    print(f"Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
