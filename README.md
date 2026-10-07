# Türkçe Full-Duplex Sesli Diyalog Sistemi

TÜBİTAK 2209-A başvurusu için prototip. Üç farklı kesinti (barge-in)
stratejisini karşılaştırmak için ölçülebilir bir deney.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # gerekirse kendi makinene göre düzenle
python scripts/download_models.py   # TTS ses modelini indirir
```

`.env` dosyası ve `models/` klasörü git'e dahil değildir (bkz.
`.gitignore`); her geliştirici kurulumdan sonra
`scripts/download_models.py`'yi çalıştırmalı.

## Donanım testi

```bash
python tests/test_mic.py
```

## Modüller

Tek başına çalıştırılabilir modüller:

```bash
python modules/tts.py "Merhaba, nasılsın?"
```
