"""
Bir oturumun wav ve jsonl kayıtlarının zaman ekseninde ne kadar hizalı
olduğunu kontrol eder (bkz. modules/controller.py SessionRecorder).

- SAĞ kanal (asistan): her tts_start olayına EN YAKIN ses başlangıcını
  (enerjinin eşiği aştığı an) bulur, farkı (ses - olay) yazar.
- SOL kanal (mikrofon): her SPEECH_START olayından ÖNCEKİ ~0.5 sn
  içindeki ses başlangıcını bulur, farkı (ses - olay) yazar. VAD
  min_speech_ms kadar (varsayılan 250 ms) gecikmeli bildirdiği için ses,
  olaydan biraz ÖNCE (negatif fark) başlamış olmalı.

Sağ kanal: ortalama ve en kötü (mutlak) farkı basar; en kötü fark
100 ms'den büyükse "HİZALAMA SORUNU" der.

Sol kanal (enerji eşiğiyle başlangıç tespiti yapıldığı için ±150 ms
dağınıklık doğaldır): ORTALAMA sapma (beklenen ~-VAD_MIN_SPEECH_MS'ten)
100 ms'yi aşarsa -- yani sabit bir kayma varsa -- "HİZALAMA SORUNU"
der; tekil bir olayın sapması 250 ms'yi aşarsa sadece o olay için uyarı
basar (tüm oturumu sorunlu saymaz). Bir önceki ses bölümünün bitişinden
300 ms'den az sonra başlayan olaylar için (yeni bir başlangıç kenarı
ayırt edilemeyeceğinden) "bulunamadı" yerine "bitişik, atlandı" yazar.

Sadece numpy ve standart kütüphane kullanır (matplotlib'e bağımlı
değildir -- bkz. tools/plot_session.py).

Çalıştırma:
    python tools/check_session_sync.py                   # en son oturum
    python tools/check_session_sync.py 20261010_173408
"""

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

FRAME_MS = 20.0
SEARCH_BEFORE_S = 0.5
SEARCH_AFTER_S = 0.5

ALIGNMENT_THRESHOLD_MS = 100.0  # sağ kanal: en kötü sapma eşiği

LEFT_MEAN_THRESHOLD_MS = 100.0  # sol kanal: ORTALAMA sapma eşiği (sabit kayma şüphesi)
LEFT_EVENT_WARN_MS = 250.0  # sol kanal: tekil olay uyarı eşiği (enerji eşiği dağılma payı)
MIN_ADJACENT_GAP_S = 0.3  # bundan az sonra başlayan olay "bitişik" sayılır


def resolve_session_stem(arg: str = None) -> Path:
    """results/sessions/<zaman> kök yolunu (uzantısız) döner."""
    sessions_dir = SETTINGS.base_dir / "results" / "sessions"

    if arg is None:
        candidates = sorted(sessions_dir.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"{sessions_dir} içinde hiç oturum (.jsonl) bulunamadı.")
        return candidates[-1].with_suffix("")

    for candidate in (Path(arg).with_suffix(""), sessions_dir / Path(arg).with_suffix("").name):
        if candidate.with_suffix(".jsonl").exists():
            return candidate

    raise FileNotFoundError(f"Oturum bulunamadı: {arg!r}")


def load_events(path: Path) -> list:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_stereo_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_channels)
    return data.astype(np.float32) / 32767.0, sample_rate


def frame_rms(channel: np.ndarray, sample_rate: int, frame_ms: float = FRAME_MS):
    """Kısa pencerelerde (varsayılan 20 ms) RMS enerjisi. Onset tespiti
    bu pencereler üzerinden yapılır (ham örnek bazında gürültüye karşı
    daha dayanıklı)."""
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000)))
    n_frames = len(channel) // frame_len
    if n_frames == 0:
        return np.zeros(0), frame_len
    trimmed = channel[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    return rms, frame_len


def adaptive_threshold(rms: np.ndarray) -> float:
    """Medyan (çoğunlukla sessizlik) üzerinden kaba bir eşik -- bu bir
    VAD değil, sadece senkron teşhisi için 'ses var/yok' ayrımı."""
    if len(rms) == 0:
        return 0.01
    noise_floor = float(np.median(rms))
    return max(noise_floor * 4.0, 0.01)


def find_rising_edges(rms: np.ndarray, frame_len: int, sample_rate: int, threshold: float) -> list:
    above = rms > threshold
    if len(above) == 0:
        return []
    edges = [0.0] if above[0] else []
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            edges.append(i * frame_len / sample_rate)
    return edges


def find_falling_edges(rms: np.ndarray, frame_len: int, sample_rate: int, threshold: float) -> list:
    """Enerjinin eşiğin ALTINA düştüğü anlar -- bir ses bölümünün
    bittiği an. "bitişik, atlandı" teşhisi için kullanılır."""
    above = rms > threshold
    edges = []
    for i in range(1, len(above)):
        if not above[i] and above[i - 1]:
            edges.append(i * frame_len / sample_rate)
    return edges


def nearest_edge(edges: list, target_s: float, before_s: float, after_s: float):
    candidates = [e for e in edges if target_s - before_s <= e <= target_s + after_s]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs(e - target_s))


def latest_before(edges: list, target_s: float):
    candidates = [e for e in edges if e <= target_s]
    return max(candidates) if candidates else None


def check_channel(
    title: str,
    edges: list,
    event_times: list,
    before_s: float,
    after_s: float,
    expected_diff_ms: float = 0.0,
    falling_edges: list = None,
    min_gap_s: float = 0.0,
    worst_threshold_ms: float = None,
    mean_threshold_ms: float = None,
    event_warn_ms: float = None,
) -> list:
    """fark = ses_başlangıcı - olay_zamanı. Sol kanalda VAD'ın kendi
    algılama gecikmesi (varsayılan ~250 ms) yüzünden fark normalde ~0
    değil, ~-250 ms'dir -- bu YANLIŞ hizalama değildir. Bu yüzden
    HİZALAMA SORUNU kararı ham farka değil, farkın BEKLENEN değerden
    (expected_diff_ms) SAPMASINA uygulanır.

    İki ayrı mod var:
    - worst_threshold_ms: EN KÖTÜ sapma bunu aşarsa HİZALAMA SORUNU
      (sağ kanal -- tek bir hizalı olmayan olay bile anlamlı).
    - mean_threshold_ms: ORTALAMA sapma bunu aşarsa HİZALAMA SORUNU
      (sol kanal -- enerji eşiğiyle tespitin doğal ±150 ms dağılması
      yüzünden tekil olaylar yanıltıcı olabilir; sabit bir kayma asıl
      aranan şey). event_warn_ms verilmişse, bu eşiği aşan TEKİL
      olaylar da (oturumu sorunlu saymadan) ayrıca işaretlenir.

    falling_edges + min_gap_s verilmişse: bir önceki ses bölümünün
    bitişinden min_gap_s'den az sonra başlayan olaylar (yeni bir
    başlangıç kenarı ayırt edilemeyeceği için) "bulunamadı" yerine
    "bitişik, atlandı" olarak işaretlenir ve soruna sayılmaz."""
    print(f"\n--- {title} ---")
    if expected_diff_ms:
        print(f"  (beklenen fark: ~{expected_diff_ms:+.0f} ms -- VAD algılama gecikmesi)")
    if not event_times:
        print("  (bu oturumda ilgili olay yok)")
        return []

    diffs_ms = []
    not_found = 0
    adjacent = 0

    for t in event_times:
        onset = nearest_edge(edges, t, before_s, after_s)
        if onset is None:
            preceding_end = latest_before(falling_edges, t) if falling_edges else None
            if preceding_end is not None and (t - preceding_end) < min_gap_s:
                adjacent += 1
                gap_ms = (t - preceding_end) * 1000
                print(f"  olay @ {t:7.3f}s : bitişik, atlandı (önceki ses bölümü {gap_ms:.0f} ms önce bitti)")
            else:
                not_found += 1
                print(
                    f"  olay @ {t:7.3f}s : ses başlangıcı bulunamadı (±{before_s:.1f}/{after_s:.1f}s penceresinde)"
                )
            continue

        diff_ms = (onset - t) * 1000
        diffs_ms.append(diff_ms)
        deviation_ms = abs(diff_ms - expected_diff_ms)
        warn = ""
        if event_warn_ms is not None and deviation_ms > event_warn_ms:
            warn = f"   ⚠ uyarı: sapma {deviation_ms:.0f} ms > {event_warn_ms:.0f} ms"
        print(f"  olay @ {t:7.3f}s : ses başlangıcı {onset:7.3f}s   (fark {diff_ms:+7.1f} ms){warn}")

    if not diffs_ms:
        if not_found:
            print(
                f"  HİZALAMA SORUNU: {not_found} olay için pencere içinde ses başlangıcı "
                "bulunamadı (kanal muhtemelen fena hâlde kaymış)"
            )
        else:
            print("  (tüm olaylar bir önceki ses bölümüne bitişikti, karşılaştırma yapılamadı)")
        return diffs_ms

    avg_diff = sum(diffs_ms) / len(diffs_ms)
    worst_diff = max(diffs_ms, key=abs)
    print(f"  Ortalama fark: {avg_diff:+6.1f} ms   En kötü fark: {worst_diff:+6.1f} ms")

    deviations = [abs(d - expected_diff_ms) for d in diffs_ms]
    avg_dev = sum(deviations) / len(deviations)
    worst_dev = max(deviations)
    if expected_diff_ms:
        print(f"  Beklenenden sapma: ortalama {avg_dev:6.1f} ms   en kötü {worst_dev:6.1f} ms")

    if mean_threshold_ms is not None:
        if avg_dev > mean_threshold_ms:
            print(f"  HİZALAMA SORUNU: ortalama sapma {avg_dev:.1f} ms > {mean_threshold_ms:.0f} ms (sabit kayma)")
    elif worst_threshold_ms is not None and worst_dev > worst_threshold_ms:
        print(f"  HİZALAMA SORUNU: en kötü sapma {worst_dev:.1f} ms > {worst_threshold_ms:.0f} ms eşiği")

    return diffs_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Oturum wav/jsonl hizalamasını kontrol eder.")
    parser.add_argument("session", nargs="?", default=None, help="Oturum adı/yolu (verilmezse en son oturum).")
    args = parser.parse_args()

    stem = resolve_session_stem(args.session)
    jsonl_path = stem.with_suffix(".jsonl")
    wav_path = stem.with_suffix(".wav")

    if not wav_path.exists():
        raise FileNotFoundError(f"wav bulunamadı: {wav_path} (oturum ses kaydı henüz yok mu?)")

    events = load_events(jsonl_path)
    audio, sample_rate = load_stereo_wav(wav_path)
    left, right = audio[:, 0], audio[:, 1]

    tts_start_times = [e["elapsed_s"] for e in events if e.get("event") == "tts_start"]
    speech_start_times = [
        e["elapsed_s"] for e in events if e.get("event") == "vad_event" and e.get("kind") == "SPEECH_START"
    ]

    left_rms, left_frame_len = frame_rms(left, sample_rate)
    right_rms, right_frame_len = frame_rms(right, sample_rate)
    left_threshold = adaptive_threshold(left_rms)
    right_threshold = adaptive_threshold(right_rms)
    left_edges = find_rising_edges(left_rms, left_frame_len, sample_rate, left_threshold)
    right_edges = find_rising_edges(right_rms, right_frame_len, sample_rate, right_threshold)
    left_falling_edges = find_falling_edges(left_rms, left_frame_len, sample_rate, left_threshold)

    print(f"Oturum          : {stem.name}")
    print(f"Örnekleme hızı  : {sample_rate} Hz")
    print(f"Kayıt süresi    : {len(left) / sample_rate:.1f} s")
    print(f"Eşik (sol/sağ)  : {left_threshold:.4f} / {right_threshold:.4f}")

    check_channel(
        "SAĞ KANAL: tts_start -> asistan sesinin fiilen başlaması",
        right_edges,
        tts_start_times,
        before_s=SEARCH_BEFORE_S,
        after_s=SEARCH_AFTER_S,
        expected_diff_ms=0.0,
        worst_threshold_ms=ALIGNMENT_THRESHOLD_MS,
    )
    check_channel(
        "SOL KANAL: SPEECH_START -> mikrofondaki konuşmanın başlaması (olaydan önce beklenir)",
        left_edges,
        speech_start_times,
        before_s=SEARCH_BEFORE_S,
        after_s=0.0,
        expected_diff_ms=-float(SETTINGS.vad_min_speech_ms),
        falling_edges=left_falling_edges,
        min_gap_s=MIN_ADJACENT_GAP_S,
        mean_threshold_ms=LEFT_MEAN_THRESHOLD_MS,
        event_warn_ms=LEFT_EVENT_WARN_MS,
    )


if __name__ == "__main__":
    main()
