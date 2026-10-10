# Türkçe Full-Duplex Sesli Diyalog Sistemi

TÜBİTAK 2209-A başvurusu için fizibilite çalışması. Amaç satışa hazır bir
ürün değil, **ölçülebilir bir deney**: tamamen yerel (Ollama + faster-whisper
+ piper-tts) bir Türkçe sesli diyalog hattı kurup, kullanıcı asistanın
sözünü kestiğinde (barge-in) farklı kesinti stratejilerinin davranışını
ve gecikmesini ölçmek. Sistem; mikrofon → VAD → iki kademeli ASR →
LLM → TTS → hoparlör hattını tek bir durum makinesiyle (`Controller`)
birleştirir ve her oturumu (ses kaydı + olay günlüğü) analiz edilebilir
şekilde diske yazar. Şu an tek bir strateji (`vad_only`) uygulanmış
durumda; mimari ileride `duration`/`semantic` gibi başka stratejileri
karşılaştırmaya göre tasarlandı.

## 1. Mimari

```
mikrofon ──▶ VAD ──▶ StreamingASR ──▶ Controller ──▶ LLM ──▶ TTS ──▶ hoparlör
(MicStream)         (partial: tiny      (LISTEN /                  (SpeakerOut)
                      final: small)      THINKING /
                                         SPEAK)
                           ▲                 │
                           └── barge-in (kesinti kararı) ──┘
```

| Modül | Dosya | Tek cümlelik görevi |
|---|---|---|
| Ses I/O | `modules/audio_io.py` | `MicStream` sürekli mikrofon okur, `SpeakerOut` anında kesilebilir şekilde hoparlöre çalar (stream program boyunca açık kalır). |
| VAD | `modules/vad.py` | silero-vad ile ses chunk'larını işleyip `SPEECH_START` / `SPEECH_END` olayları üretir. |
| ASR | `modules/asr.py` | `ASR` faster-whisper ile tek seferlik transcribe yapar; `StreamingASR` konuşma sürerken hızlı bir modelle (partial) ara metin, konuşma bitince daha kaliteli bir modelle (final) kesin metin üretir; pre-roll tamponu ve teşhis için `results/debug/utt_<n>.wav` kaydı içerir. |
| LLM | `modules/llm.py` | Ollama `/api/chat` ile streaming sohbet; sistem promptunu `prompts/system_tr.txt`'ten okur, cümle cümle (`stream_sentences`) üretir, `cancel()` ile anında durdurulabilir. |
| TTS | `modules/tts.py` | piper-tts ile Türkçe metni sese çevirir. |
| Strateji | `modules/strategies.py` | Barge-in kararını verir (`INTERRUPT`/`WAIT`/`IGNORE`); şu an tek uygulama `VadOnlyStrategy` (konuşma başlar başlamaz keser). |
| Controller | `modules/controller.py` | `LISTEN`/`THINKING`/`SPEAK` durum makinesi, konuşma geçmişi, `SessionLogger` (olay günlüğü) ve `SessionRecorder` (iki kanallı ses kaydı). |

Ana döngü (mikrofon → VAD → ASR → durum kararları) tek bir thread'de
çalışır; LLM → TTS → hoparlör cevabı ayrı bir worker thread'de üretilir
(büyük bir asyncio dönüşümü yok, mevcut thread tabanlı modüller
kullanılıyor). Barge-in olduğunda worker beklenmeden hoparlör susturulur,
LLM isteği iptal edilir ve kullanıcının yeni konuşması (pre-roll dahil)
kaybolmadan ASR'a akmaya devam eder.

## 2. Gereksinimler

- **Python 3.10+** (geliştirme ve testler Python 3.13 ile yapıldı).
- **[Ollama](https://ollama.com)** yerel olarak çalışıyor olmalı
  (`http://localhost:11434`) ve `gemma3:4b` modeli kurulu olmalı
  (varsayılan `LLM_MODEL`).
- **Kulaklık zorunlu.** Sistemde yankı giderme (echo cancellation) yok;
  hoparlörden çıkan ses mikrofona karışırsa VAD bunu kullanıcı konuşması
  sanıp yanlışlıkla asistanın sözünü keser.
- Donanım: geliştirme ve ölçümler **macOS, Apple Silicon (M4), CPU**
  üzerinde yapıldı (`ASR_DEVICE=cpu`, `ASR_COMPUTE_TYPE=int8`).
  Windows'ta NVIDIA GPU varsa `.env`'de `ASR_DEVICE=cuda`,
  `ASR_COMPUTE_TYPE=float16` denenebilir (bkz. `.env.example`).

## 3. Kurulum

`models/` klasörü ve `.env` dosyası git'e dahil değildir (bkz.
`.gitignore`); kurulumdan sonra model indirme adımı herkes için gerekir.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # gerekirse değerleri düzenleyin

python scripts/download_models.py   # piper TTS sesini models/ altına indirir

# Ollama (bir kere): brew install ollama  ya da https://ollama.com/download
ollama pull gemma3:4b
```

Mikrofon izni: **System Settings → Privacy & Security → Microphone**
bölümünden terminal/IDE uygulamasına izin verin (bkz. `tests/test_mic.py`).

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env

python scripts\download_models.py

# Ollama: https://ollama.com/download adresinden kurun
ollama pull gemma3:4b
```

Mikrofon izni: **Ayarlar → Gizlilik ve güvenlik → Mikrofon** bölümünden
uygulamanın mikrofona erişimine izin verin.

### Her iki platformda ortak notlar

- `requirements.txt` içinde `python-dotenv` **yoktur**; kuruluysa
  `config.py` `.env` dosyasını otomatik okur, kurulu değilse ortam
  değişkenlerini kendiniz export etmeniz gerekir (ya da
  `pip install python-dotenv`).
- `scripts/download_models.py` yalnızca **piper TTS sesini** indirir.
  faster-whisper modelleri (`tiny`, `small`) ilk kullanımda Hugging Face
  Hub'dan otomatik indirilip yerel önbelleğe kaydedilir; ilk
  `python main.py` çalıştırması bu yüzden internet ister ve biraz
  yavaş başlar.
- Kurulumu doğrulamak için: `python tests/test_mic.py` (mikrofon),
  `python modules/tts.py "Merhaba"` (piper sesi).

## 4. Çalıştırma

```bash
python main.py
```

Açılışta modeller yüklenir ve ısıtılır; ısınma süreleri ekrana yazılır
(`ASR ısınma (partial/tiny)`, `ASR ısınma (final/small)`, `LLM ısınma`).
Sonrasında canlı diyalog döngüsü başlar ve ekranda sade bir akış görünür:

- `[LISTEN]` / `[THINKING]` / `[SPEAK]` — durum değişimleri
- `Sen: ...` — ASR'ın ürettiği final metin
- `Asistan: ...` — gerçekten çalınan her cümle
- `⚡ KESİLDİ (xx ms)` — bir barge-in anında, hoparlörün susma gecikmesiyle

**Ctrl+C** ile çıkış temizdir: worker iptal edilir, ASR/hoparlör/mikrofon
kapatılır, oturum ses kaydı diske yazılır ve bir özet basılır (tur
sayısı, kesinti sayısı, yanıt gecikmesi ve kesinti gecikmesi listeleri,
log/ses kaydı yolları).

`modules/controller.py` doğrudan da çalıştırılabilir (`python
modules/controller.py`) — `main.py` ile aynı işi yapar.

## 5. Oturum çıktıları

Her oturum için `results/sessions/<YYYYAAGG_SSDDss>` kökünde iki dosya
oluşur:

- **`<zaman>.jsonl`**: her olay zaman damgasıyla (`elapsed_s`) tek satır
  JSON. Olay tipleri: `session_start`, `state_change`, `vad_event`,
  `barge_in_decision`, `speech_end`, `asr_final`, `interrupt`,
  `llm_start`, `llm_sentence`, `response_latency`, `tts_start`,
  `tts_end`, `llm_error`, `turn_end`, `mic_overflow`, `session_end`.
- **`<zaman>.wav`**: stereo, 16 kHz. **Sol kanal mikrofon** (kullanıcı,
  sürekli), **sağ kanal hoparlöre giden ses** (asistan, 16 kHz'e
  resample edilmiş); bir kesinti olduysa o cümlenin sesi kayıtta da tam
  kesinti anında kesilir.

Ayrıca `StreamingASR`, her konuşmanın ASR'a verilen tam sesini teşhis
amacıyla `results/debug/utt_<n>.wav` olarak kaydeder.

### `tools/plot_session.py` — zaman çizelgesi PNG'si

```bash
python tools/plot_session.py                      # en son oturum
python tools/plot_session.py 20261010_173408       # belirli bir oturum
python tools/plot_session.py --row-seconds 10      # satır başına 10 s
python tools/plot_session.py --start 55 --end 70 --out results/sessions/ornek_hihi.png
python tools/plot_session.py --title "Özel başlık"
```

Üst şerit kullanıcı konuşma aralıkları (final metin etiketiyle), alt
şerit asistan (TTS) konuşma aralıkları, kırmızı dikey çizgi kesintiler,
arka plan bandı LISTEN/THINKING/SPEAK durumları. Çıktı varsayılan olarak
`results/sessions/<zaman>.png`.

### `tools/check_session_sync.py` — ses/log hizalama kontrolü

```bash
python tools/check_session_sync.py                 # en son oturum
python tools/check_session_sync.py 20261010_173408
```

`<zaman>.wav` ile `<zaman>.jsonl`'i karşılaştırır: sağ kanalda her
`tts_start`'a en yakın ses başlangıcını, sol kanalda her
`SPEECH_START`'tan önceki sesi bulur. Sol kanalda sabit bir kayma
(işaretli ortalama sapma 100 ms'yi aşarsa) "HİZALAMA SORUNU" basar;
tekil olaylardaki dağılma (≤250 ms, enerji eşiği tespitinin doğal payı)
sadece uyarı olarak işaretlenir. Sadece `numpy` ve standart kütüphane
kullanır.

## 6. Testler ve ölçüm araçları

| Script | Ne ölçer | Çalıştırma | Çıktı |
|---|---|---|---|
| `tests/test_mic.py` | Mikrofon izni/çalışırlığı (3 sn kayıt, maks. genlik) | `python tests/test_mic.py` | `test_mic.wav` (repo kökü) |
| `tests/test_vad_guided.py` | VAD'ı yönlendirmeli senaryolarla (sessizlik, normal cümle, arada duraklama, "hı hı", vuruş sesi) test eder | `python tests/test_vad_guided.py` | `results/vad_test.txt` |
| `tests/test_asr_offline.py` | Eski (streaming olmayan) TTS→ASR tek seferlik transcribe, RTF | `python tests/test_asr_offline.py` | sadece konsol |
| `tests/asr_latency.py` | `tiny`/`base`/`small` whisper modellerinin 3 farklı ses uzunluğunda (~0.5/1.5/4 s) transcribe gecikmesi, sabit maliyet hipotezi | `python tests/asr_latency.py` | `results/asr_latency.txt` |
| `tests/compare_llm.py` | `LLM_COMPARE_MODELS` modellerini aynı sorularla karşılaştırır (TTFT, ilk cümle süresi, token/s); gemma3:4b'yi ayrıca "bilgi kartı" promptuyla dener | `python tests/compare_llm.py` | `results/llm_comparison.txt` |
| `tests/benchmark.py` | TTS, ASR, LLM, SpeakerOut'un tamamını tek seferde ölçer (RTF, WER, TTFT, kesinti gecikmesi), makine bilgisiyle | `python tests/benchmark.py` | `results/benchmark_<makine>.txt` |

## 7. Ayarlar (`.env`)

`cp .env.example .env` ile kopyalanıp düzenlenir; `config.py` her
değişkeni okuyup `SETTINGS` nesnesine toplar (varsayılanlar Mac/CPU
içindir). Öne çıkan değişkenler:

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Ses I/O örnekleme hızı (Hz). |
| `AUDIO_CHUNK_MS` | `20` | Mikrofon okuma / hoparlör çalma chunk boyutu (ms). |
| `VAD_THRESHOLD` | `0.5` | silero-vad konuşma olasılığı eşiği. |
| `VAD_MIN_SPEECH_MS` | `250` | Konuşma olarak sayılması için gereken minimum süre (SPEECH_START gecikmesi). |
| `VAD_MIN_SILENCE_MS` | `500` | SPEECH_END için gereken minimum sessizlik süresi. |
| `ASR_MODEL_SIZE` | `small` | `StreamingASR.finalize()` (kesin metin) için whisper modeli. |
| `ASR_PARTIAL_MODEL` | `tiny` | Konuşma sürerken ara metin için kullanılan (daha hızlı) whisper modeli. |
| `ASR_DEVICE` | `cpu` | `cpu` (Mac) / `cuda` (Windows + NVIDIA GPU). |
| `ASR_COMPUTE_TYPE` | `int8` | `int8` (CPU) / `float16` (GPU). |
| `ASR_CPU_THREADS` | `4` | Whisper modeline verilen CPU thread sayısı. |
| `ASR_PARTIAL_INTERVAL_MS` | `500` | Konuşma sürerken kaç ms'de bir yeni ara metin üretilir. |
| `ASR_PREROLL_MS` | `500` | VAD'ın geç bildirdiği konuşma başlarını telafi eden sürekli ses tamponu. |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama sunucu adresi. |
| `LLM_MODEL` | `gemma3:4b` | Canlı diyalogda kullanılan model. |
| `LLM_TEMPERATURE` | `0.1` | Düşük sıcaklık = daha az uydurma, daha tutarlı cevap. |
| `LLM_TIMEOUT_S` | `30` | Ollama isteği zaman aşımı (sn). |
| `LLM_COMPARE_MODELS` | `qwen2.5:3b,qwen2.5:7b,gemma3:4b` | `tests/compare_llm.py`'nin karşılaştırdığı modeller. |
| `TTS_VOICE` | `tr_TR-dfki-medium` | piper Türkçe ses modeli adı. |
| `BARGE_IN_STRATEGY` | `vad_only` | Kesinti stratejisi (şu an tek seçenek). |
| `INPUT_DEVICE` / `OUTPUT_DEVICE` | *(boş)* | Belirli bir ses cihazı seçmek için (boşsa sistem varsayılanı). |

(Tam liste için `.env.example` dosyasına bakın; `python config.py` tüm
çözümlenmiş ayarları ekrana basar.)

## 8. Sınırlılıklar

- **Kulaklık zorunlu** — yankı giderme yok (bkz. Gereksinimler).
- Şu an yalnızca **`vad_only`** stratejisi uygulanmış durumda (konuşma
  başlar başlamaz keser); `duration`/`semantic` stratejiler için
  arayüz (`modules/strategies.py`) hazır ama henüz yazılmadı.
- **Küçük LLM** (`gemma3:4b`) bazen "tek kısa cümle" kuralını bozabilir
  ya da yanlış/uydurma bilgi verebilir; `prompts/system_tr.txt` bunu
  azaltmaya çalışır ama garanti etmez.
- **ASR kısa/alçak seslerde zayıf**: 300 ms'den kısa ses hiç transcribe
  edilmez (halüsinasyonu önlemek için), "hı hı" gibi kısa ünlemler çoğu
  zaman anlamlı metne dönüşmez.
- **Yanıt gecikmesi** (SPEECH_END → asistanın ilk sesi) gerçek
  oturumlarda tipik olarak **~2.5–3.5 sn** civarında ölçüldü (bkz.
  `results/sessions/*.jsonl` içindeki `response_latency` olayları).

## 9. Sık karşılaşılan sorunlar

- **Mikrofon genliği 0 / "MİKROFON İZNİ YOK" uyarısı**: `tests/test_mic.py`
  bunu tespit eder. macOS'ta System Settings → Privacy & Security →
  Microphone; Windows'ta Ayarlar → Gizlilik ve güvenlik → Mikrofon
  bölümünden izin verin.
- **"Ollama'ya bağlanılamadı" hatası**: Ollama sunucusu çalışmıyor ya da
  model kurulu değil. `ollama serve` ile başlatın, `ollama pull
  gemma3:4b` ile modeli indirin (bkz. `modules/llm.py`).
- **piper-tts sürüm hatası**: `piper-tts` 1.4'ten eskiyse (özellikle
  macOS'ta derleme hatası verir) `modules/tts.py` açık bir `RuntimeError`
  fırlatır. Çözüm: `pip install --upgrade "piper-tts>=1.4"`.
- **Ctrl+C programı hemen kapatmıyor**: Whisper'ın gerçek bir transcribe
  çağrısı sürerken (saniyeler sürebilir) Python sinyali hemen işleyemez;
  çağrı bitince (`StreamingASR`/`Controller` zaten `daemon=True` thread'ler
  ve `join(timeout=2.0)` ile) birkaç saniye içinde temiz kapanır.
- **`HF_TOKEN` / Hugging Face uyarısı**: faster-whisper modelleri ilk kez
  indirirken `huggingface_hub` bir token uyarısı basabilir; genel
  (public) modeller için zararsızdır, işlevi etkilemez.
- **`torch.jit.load` / `FutureWarning` uyarısı**: `modules/vad.py`'nin
  yüklediği silero-vad modelinden gelir (kütüphane içi, zararsız).

---

## Kontrol listesi (bu README için doğrulanan komutlar/dosyalar)

- `python -m venv .venv`, `source .venv/bin/activate`, `.venv\Scripts\activate`, `pip install -r requirements.txt` — standart venv adımları.
- `cp .env.example .env` / `copy .env.example .env` — dosya gerçekten var (`.env.example`).
- `python scripts/download_models.py` — dosya var, sadece piper sesini indirdiği doğrulandı (`scripts/download_models.py` okundu).
- `ollama pull gemma3:4b` — `config.py`'deki `LLM_MODEL` varsayılanıyla (`gemma3:4b`) eşleşiyor.
- `python tests/test_mic.py` — dosya var, çıktı yolu (`test_mic.wav`, repo kökü) kodundan doğrulandı.
- `python modules/tts.py "Merhaba"` — dosyanın docstring'indeki örnekle birebir.
- `python main.py` — dosya var, `modules/controller.py`'yi çağırdığı doğrulandı.
- `python modules/controller.py` — dosyanın kendi `if __name__ == "__main__"` bloğu var.
- Ekrandaki `[LISTEN]/[THINKING]/[SPEAK]`, `Sen: `, `Asistan: `, `⚡ KESİLDİ (... ms)`, ısınma satırları ve özet başlıkları — `modules/controller.py` içindeki `self._ui(...)`/`print(...)` çağrılarından birebir alındı.
- `results/sessions/<zaman>.jsonl` olay adları — `modules/controller.py`'deki tüm `self.logger.log(...)` çağrıları tek tek grep'lendi.
- `results/sessions/<zaman>.wav` kanal düzeni (sol=mikrofon, sağ=asistan, 16 kHz) — `SessionRecorder` sınıfının docstring'i ve kodu okundu.
- `results/debug/utt_<n>.wav` — `modules/asr.py`'deki `StreamingASR._save_debug_audio` ve `debug_dir` varsayılanı doğrulandı.
- `python tools/plot_session.py` ve tüm argümanları (`--row-seconds`, `--start`, `--end`, `--out`, `--title`, pozisyonel `session`) — `argparse` tanımları dosyadan birebir okundu.
- `python tools/check_session_sync.py` ve pozisyonel `session` argümanı — `argparse` tanımı dosyadan okundu.
- Her `tests/*.py` dosyasının çalıştırma komutu ve çıktı yolu (`results/vad_test.txt`, `results/asr_latency.txt`, `results/llm_comparison.txt`, `results/benchmark_<makine>.txt`, `test_mic.wav`, ya da çıktı yok) — her dosyanın docstring'i ve `RESULTS_PATH`/`OUTPUT_PATH` tanımı tek tek grep'lenip okundu.
- Ayarlar tablosundaki tüm değişken adı ve varsayılan değer — `config.py`'deki `_load()` fonksiyonundan birebir kopyalandı (uydurulan değer yok).
- `requirements.txt` içeriği (`python-dotenv` dahil olmadığı) ve `.gitignore`'daki `*.wav`/`models/`/`.env` kuralları — ilgili dosyalar doğrudan okundu.
- Python sürümü gereksinimi (3.10+) — `config.py`/`modules/audio_io.py`'deki `X | None` tip ipucu sözdiziminin (PEP 604, `from __future__ import annotations` olmadan) minimum sürüm gereksinimi olduğu kontrol edildi; geliştirme ortamı `.venv/bin/python3 --version` ile `3.13.4` olarak doğrulandı.
- TTS örnekleme hızı (22050 Hz) ve `SessionRecorder`'ın hedef hızı (16000 Hz) — `modules/audio_io.py` (`SpeakerOut.DEFAULT_SAMPLE_RATE`) ve `models/tr_TR-dfki-medium.onnx.json`'daki `audio.sample_rate` alanı okunarak doğrulandı.
