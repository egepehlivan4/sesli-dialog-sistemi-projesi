"""
Gerekli model dosyalarını models/ klasörüne indirir.

models/ klasörü .gitignore'da olduğu için git'e dahil değildir; bu
script, projeyi klonlayan herkesin aynı model dosyalarını kendi
makinesine çekmesini sağlar.

Zaten indirilmiş (ve boyutu doğru) dosyaları atlar.

Çalıştırma:
    python scripts/download_models.py
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# (indirilecek dosya, kaynak URL, beklenen minimum boyut [byte] -- yarım inen
#  dosyaları yeniden indirmek için kaba bir kontrol)
MODELS = [
    (
        f"{SETTINGS.tts_voice}.onnx",
        f"{PIPER_VOICES_BASE}/tr/tr_TR/dfki/medium/{SETTINGS.tts_voice}.onnx",
        1_000_000,
    ),
    (
        f"{SETTINGS.tts_voice}.onnx.json",
        f"{PIPER_VOICES_BASE}/tr/tr_TR/dfki/medium/{SETTINGS.tts_voice}.onnx.json",
        100,
    ),
]


def download(url: str, dest: Path) -> None:
    print(f"İndiriliyor: {url}")
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"Kaydedildi: {dest} ({dest.stat().st_size / 1_000_000:.1f} MB)")


def main() -> None:
    SETTINGS.models_dir.mkdir(parents=True, exist_ok=True)

    for filename, url, min_size in MODELS:
        dest = SETTINGS.models_dir / filename
        if dest.exists() and dest.stat().st_size >= min_size:
            print(f"Zaten mevcut, atlanıyor: {dest}")
            continue
        download(url, dest)

    print("Tüm model dosyaları hazır.")


if __name__ == "__main__":
    main()
