# Türkçe Full-Duplex Sesli Diyalog Sistemi

TÜBİTAK 2209-A başvurusu için prototip. Amaç çalışan bir ürün değil,
ölçülebilir bir deney: üç farklı kesinti (barge-in) stratejisini
karşılaştırmak.

## Ekip ve platform
- Geliştirici 1: macOS, Apple Silicon (M4)
- Geliştirici 2: Windows
- Kod ZORUNLU olarak çapraz platform olmalı

## Kesin kurallar
- mlx, mlx-whisper, afplay gibi macOS'a özel şeyler KULLANMA
- Mutlak yol yazma; her zaman pathlib.Path kullan
- Platforma/makineye bağlı her ayar config.py içinde, ortam
  değişkeninden okunacak. Kodun geri kalanı platformu bilmeyecek
- Ses çalma: sounddevice.play() (subprocess ile sistem komutu değil)

## Stack
- ASR: faster-whisper (model "small", compute_type "int8")
- LLM: Ollama HTTP API, http://localhost:11434, model qwen2.5:3b
- TTS: piper-tts >= 1.4, Türkçe ses tr_TR-dfki-medium
- VAD: silero-vad
- Ses I/O: sounddevice, 16000 Hz, mono, float32
- Eşzamanlılık: asyncio

## Mimari
mikrofon -> VAD -> ASR (streaming) -> controller -> LLM -> TTS -> hoparlör

Controller üç durumlu bir FSM:
- LISTEN: sessiz, kullanıcıyı bekliyor
- SPEAK: TTS çalıyor, mikrofon AÇIK kalıyor
- INTERRUPTED: kullanıcı araya girdi, TTS durduruldu

Her modül aynı arayüzü uygular: start(), feed(chunk), stop()

## Çalışma tarzı
- Türkçe açıkla
- Küçük adımlarla ilerle; bir seferde bir dosya
- Her modül tek başına çalıştırılabilir olsun (if __name__ == "__main__")
- Yeni bağımlılık eklemeden önce sor
- Kodu yazdıktan sonra NE YAPTIĞINI kısaca anlat