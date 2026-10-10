"""
ASR modülü: faster-whisper ile Türkçe konuşmayı metne çevirir.

Model boyutu, cihaz, hesaplama tipi ve CPU thread sayısı config.py'den
okunur (varsayılan: small / cpu / int8 / 4 thread). Whisper 16000 Hz ses
bekler; farklı bir örnekleme hızıyla gelen ses transcribe() içinde
otomatik yeniden örneklenir.

StreamingASR, konuşma sürerken belirli aralıklarla (varsayılan 500 ms,
ASR_PARTIAL_INTERVAL_MS) o ana kadar biriken sesi transcribe ederek
"kısmi metin" üretir; ayrı bir worker thread'de çalıştığı için mikrofon
akışını bloklamaz. VAD'ın SPEECH_START'ı ~250 ms gecikmeyle bildirmesi
yüzünden kaybolan cümle başlarını telafi etmek için sürekli bir pre-roll
halka tamponu tutar (ASR_PREROLL_MS). Her konuşmanın ASR'a verilen tam
sesi, teşhis amacıyla results/debug/utt_<n>.wav olarak kaydedilir.

Canlı test (MicStream + VAD + StreamingASR):
    python modules/asr.py

Eski, streaming olmayan TTS -> ASR offline entegrasyon testi
tests/test_asr_offline.py'e taşındı.
"""

import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

WHISPER_SAMPLE_RATE = 16000
LANGUAGE = "tr"

# Bundan daha kısa ses hiç transcribe edilmez: whisper çok kısa/gürültü
# parçalarında sıkça halüsinasyon üretir ("Altyazı M.K." gibi).
MIN_TRANSCRIBE_AUDIO_MS = 300


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Basit lineer interpolasyonla yeniden örnekleme (ek bağımlılık gerektirmez)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    orig_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
    resampled = np.interp(target_times, orig_times, audio)
    return resampled.astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """float32 [-1, 1] sesi 16-bit PCM WAV olarak kaydeder (ek bağımlılık
    gerektirmez, stdlib wave modülü yeterli)."""
    pcm16 = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


@dataclass
class TranscriptionResult:
    text: str
    no_speech_prob: float
    avg_logprob: float


class ASR:
    """faster-whisper tabanlı konuşma tanıma. Model ilk kullanımda
    yüklenir (lazy), her transcribe() çağrısında yeniden yüklenmez."""

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None,
        cpu_threads: int = None,
    ):
        self.model_size = model_size or SETTINGS.asr_model_size
        self.device = device or SETTINGS.asr_device
        self.compute_type = compute_type or SETTINGS.asr_compute_type
        self.cpu_threads = cpu_threads if cpu_threads is not None else SETTINGS.asr_cpu_threads
        self._model: WhisperModel | None = None

    def load(self) -> WhisperModel:
        """Model zaten yüklenmediyse yükler. Yükleme süresini yazdırır."""
        if self._model is None:
            print(
                f"Whisper modeli yükleniyor: {self.model_size} "
                f"({self.device}, {self.compute_type}, cpu_threads={self.cpu_threads})..."
            )
            start = time.perf_counter()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            print(f"Model yüklendi ({time.perf_counter() - start:.3f} s).")
        return self._model

    def transcribe_detailed(
        self, audio: np.ndarray, sample_rate: int, beam_size: int = None
    ) -> TranscriptionResult:
        """transcribe()'ın detaylı hali: metnin yanında no_speech_prob ve
        avg_logprob'u da döner (segmentler üzerinden ortalanmış; ileride
        güven skoru olarak kullanılabilir).

        - condition_on_previous_text=False: whisper'ın önceki (hayali)
          metne bakarak halüsinasyona devam etmesini engeller.
        - beam_size=None bırakılırsa whisper'ın kendi varsayılanı
          kullanılır (final transcribe için); partial'lar için çağıran
          taraf beam_size=1 vererek hızlandırabilir.
        - MIN_TRANSCRIBE_AUDIO_MS'den kısa ses hiç modele verilmez.
        """
        audio = np.asarray(audio, dtype=np.float32)
        duration_ms = len(audio) / sample_rate * 1000 if sample_rate else 0.0
        if duration_ms < MIN_TRANSCRIBE_AUDIO_MS:
            return TranscriptionResult(text="", no_speech_prob=1.0, avg_logprob=0.0)

        model = self.load()
        audio_16k = _resample(audio, sample_rate, WHISPER_SAMPLE_RATE)

        kwargs = {"language": LANGUAGE, "condition_on_previous_text": False}
        if beam_size is not None:
            kwargs["beam_size"] = beam_size

        segments_gen, _info = model.transcribe(audio_16k, **kwargs)
        segments = list(segments_gen)
        text = " ".join(segment.text.strip() for segment in segments).strip()

        if segments:
            no_speech_prob = sum(s.no_speech_prob for s in segments) / len(segments)
            avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        else:
            no_speech_prob = 1.0
            avg_logprob = 0.0

        return TranscriptionResult(text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)

    def transcribe(self, audio: np.ndarray, sample_rate: int, beam_size: int = None) -> str:
        return self.transcribe_detailed(audio, sample_rate, beam_size=beam_size).text

    def transcribe_file(self, path: Path) -> str:
        model = self.load()
        segments, _info = model.transcribe(
            str(path), language=LANGUAGE, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class StreamingASR:
    """Akışlı ASR: ses chunk'larını (ör. MicStream'den 20 ms'lik
    parçalar) biriktirir; arka planda çalışan bir worker thread belirli
    aralıklarla (varsayılan 500 ms) o ana kadar biriken TÜM sesi
    faster-whisper'a vererek "kısmi metin" (partial) üretir.

    İki kademeli model: partial'lar hızlı ama daha kaba bir modelle
    (varsayılan ASR_PARTIAL_MODEL="tiny", beam_size=1), finalize() ise
    daha kaliteli asıl modelle (ASR_MODEL_SIZE="small", varsayılan beam)
    çalışır -- tests/asr_latency.py ölçümlerine göre.

    feed() çağrısı sadece numpy concatenate + lock içerdiği için ucuzdur
    ve ses thread'ini bloklamaz; asıl ağır iş (whisper transcribe) her
    zaman worker thread'inde çalışır. finalize() çağıran thread'i
    bloklar (konuşma bittiğinde bir kereliğine tam transcribe yapar) ama
    worker'da sürmekte olan bir partial'ı BEKLEMEZ.

    feed_preroll(), konuşma durumundan bağımsız sürekli çağrılmalıdır;
    reset() çağrıldığında o ana kadar biriken pre-roll sesi yeni
    konuşmanın başına eklenir -- VAD'ın SPEECH_START'ı geç bildirmesi
    yüzünden kaybolan cümle başlarını telafi eder.
    """

    def __init__(
        self,
        model_size: str = None,
        partial_model_size: str = None,
        device: str = None,
        compute_type: str = None,
        partial_interval_ms: int = None,
        preroll_ms: int = None,
        sample_rate: int = None,
        debug_dir: Path = None,
    ):
        # İki kademeli ASR: partial'lar hızlı ama daha kaba bir modelle
        # (varsayılan "tiny"), finalize() asr_latency.txt ölçümlerinde
        # daha kaliteli çıkan asıl modelle (varsayılan "small") çalışır.
        self._final_asr = ASR(model_size=model_size, device=device, compute_type=compute_type)
        self._partial_asr = ASR(
            model_size=partial_model_size if partial_model_size is not None else SETTINGS.asr_partial_model,
            device=device,
            compute_type=compute_type,
        )
        self.partial_interval_s = (
            partial_interval_ms if partial_interval_ms is not None else SETTINGS.asr_partial_interval_ms
        ) / 1000.0
        self.preroll_ms = preroll_ms if preroll_ms is not None else SETTINGS.asr_preroll_ms
        self.sample_rate = sample_rate or SETTINGS.sample_rate
        self._preroll_max_samples = int(self.sample_rate * self.preroll_ms / 1000)
        self.debug_dir = Path(debug_dir) if debug_dir else SETTINGS.base_dir / "results" / "debug"

        self._lock = threading.Lock()
        self._transcribe_lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._preroll_buffer = np.zeros(0, dtype=np.float32)
        self._speech_start_time = None  # time.perf_counter() değeri
        self._partial_text = ""
        self._partials = []  # [(elapsed_ms, text, transcribe_ms, no_speech_prob, avg_logprob), ...]
        self._active = False  # reset() çağrıldı, finalize() henüz çağrılmadı
        self._utt_index = 0

        self._stop_event = threading.Event()
        # daemon=True: Ctrl+C'de ana thread çıkınca worker'ın bir sonraki
        # transcribe turunu beklemeden program kapanabilsin.
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def load(self) -> None:
        """Her iki whisper modelini de önceden yükler (konuşma
        başlamadan önce çağrılırsa, model yükleme süresi ilk partial'a
        ya da FINAL'e karışmaz)."""
        self._final_asr.load()
        self._partial_asr.load()

    def warmup(self) -> dict:
        """Her iki modeli de (partial: tiny, final: small) 1 saniyelik
        sessizlikle GERÇEK bir transcribe çağrısıyla ısıtır -- ilk
        kullanıcı cümlesi soğuk modele denk gelmesin diye. Isınma
        süresini saniye olarak döner: {"partial": ..., "final": ...}."""
        silence = np.zeros(self.sample_rate, dtype=np.float32)  # 1 s

        start = time.perf_counter()
        self._partial_asr.transcribe_detailed(silence, self.sample_rate, beam_size=1)
        partial_s = time.perf_counter() - start

        start = time.perf_counter()
        self._final_asr.transcribe_detailed(silence, self.sample_rate)
        final_s = time.perf_counter() - start

        return {"partial": partial_s, "final": final_s}

    def feed_preroll(self, chunk: np.ndarray) -> None:
        """Konuşma durumundan BAĞIMSIZ, her mikrofon chunk'ında çağrılır;
        son preroll_ms kadar sesi sürekli tutar. reset() bu tamponu yeni
        konuşmanın başlangıcına ekler."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            self._preroll_buffer = np.concatenate([self._preroll_buffer, chunk])
            if len(self._preroll_buffer) > self._preroll_max_samples:
                self._preroll_buffer = self._preroll_buffer[-self._preroll_max_samples :]

    def feed(self, chunk: np.ndarray) -> None:
        """Ses chunk'ını biriktirir. reset()'ten sonra çağrılmalıdır."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            self._audio = np.concatenate([self._audio, chunk])

    def get_partial(self) -> str:
        """O ana kadarki en güncel (kısmi) metin."""
        with self._lock:
            return self._partial_text

    @property
    def partials(self) -> list:
        """Şimdiye kadar üretilen tüm partial'lar:
        [(elapsed_ms, text, transcribe_ms, no_speech_prob, avg_logprob), ...].
        elapsed_ms, konuşma başlangıcından (pre-roll dahil) bu partial'ın
        üretildiği ana kadar geçen süre; transcribe_ms o partial için
        whisper çağrısının kendisinin ne kadar sürdüğüdür."""
        with self._lock:
            return list(self._partials)

    def finalize(self) -> TranscriptionResult:
        """Konuşma bitince çağrılır: biriken TÜM sesi transcribe edip
        kesin sonucu döner, ASR'a verilen sesi debug/utt_<n>.wav'a
        kaydeder. Worker'ın bu konuşma turu için yeni partial üretmesini
        durdurur ama worker'da hâlâ süren bir partial'ı BEKLEMEZ; o
        partial sonunda kendi sonucunu (aktif konuşma artık bitti diye)
        yok sayar."""
        with self._lock:
            audio = self._audio.copy()
            self._active = False

        self._save_debug_audio(audio)

        got_lock = self._transcribe_lock.acquire(blocking=False)
        try:
            result = self._final_asr.transcribe_detailed(audio, self.sample_rate)
        finally:
            if got_lock:
                self._transcribe_lock.release()

        with self._lock:
            self._partial_text = result.text
        return result

    def reset(self) -> None:
        """Yeni bir konuşma turu başlatır: tampon, o ana kadar biriken
        pre-roll sesiyle başlatılır (VAD'ın SPEECH_START'ı geç bildirmesi
        yüzünden kaybolan cümle başlarını telafi eder)."""
        with self._lock:
            preroll = self._preroll_buffer.copy()
            preroll_ms = len(preroll) / self.sample_rate * 1000 if self.sample_rate else 0.0

            self._utt_index += 1
            self._audio = preroll
            self._partial_text = ""
            self._partials = []
            self._active = True
            self._speech_start_time = time.perf_counter() - preroll_ms / 1000.0

    def close(self) -> None:
        """Worker thread'i durdurur (program sonunda çağrılmalı)."""
        self._stop_event.set()
        self._worker.join(timeout=2.0)

    def _save_debug_audio(self, audio: np.ndarray) -> None:
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            path = self.debug_dir / f"utt_{self._utt_index}.wav"
            _write_wav(path, audio, self.sample_rate)
        except OSError as exc:
            print(f"[StreamingASR] debug wav yazılamadı: {exc}", file=sys.stderr)

    def _worker_loop(self) -> None:
        # wait(timeout) hem periyodik bekleme hem de close()'da anında
        # uyanma sağlar: event set edilirse True döner ve döngü biter.
        while not self._stop_event.wait(timeout=self.partial_interval_s):
            with self._lock:
                active = self._active
                audio = self._audio.copy()
                start_time = self._speech_start_time

            if not active or len(audio) == 0:
                continue

            # Aynı anda yalnızca bir transcribe çalışsın: lock tutulamazsa
            # (ör. finalize() o an transcribe ediyor), bu turu atla --
            # bekleme, işlemci çakışmasına girme.
            if not self._transcribe_lock.acquire(blocking=False):
                continue

            try:
                transcribe_start = time.perf_counter()
                result = self._partial_asr.transcribe_detailed(audio, self.sample_rate, beam_size=1)
                transcribe_ms = (time.perf_counter() - transcribe_start) * 1000
            finally:
                self._transcribe_lock.release()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            threshold_ms = self.partial_interval_s * 1000
            if transcribe_ms > threshold_ms:
                print(
                    f"[StreamingASR] partial geride kaldı: transcribe "
                    f"{transcribe_ms:.0f} ms sürdü (eşik {threshold_ms:.0f} ms)",
                    file=sys.stderr,
                )

            with self._lock:
                # Transcribe sürerken finalize() çağrılmış ya da yeni bir
                # konuşma başlamış olabilir; öyleyse bu partial'ı atla.
                if not self._active or self._speech_start_time != start_time:
                    continue
                self._partial_text = result.text
                self._partials.append(
                    (elapsed_ms, result.text, transcribe_ms, result.no_speech_prob, result.avg_logprob)
                )


def main() -> None:
    """MicStream + VAD + StreamingASR'ı birlikte çalıştırır: VAD konuşma
    başlangıcını algılayınca StreamingASR beslenmeye başlar, konuşma
    bitince finalize() çağrılır. Her partial, konuşma başlangıcına göre
    zaman damgasıyla ekrana basılır. Konuşma bitince SPEECH_END'den
    FINAL'e kadar geçen süre ve ilk partial'ın konuşma başlangıcından ne
    kadar sonra geldiği ayrıca özetlenir (eski -streaming olmayan-
    yöntemle, bkz. tests/test_asr_offline.py, karşılaştırmak için)."""
    from modules.audio_io import MicStream
    from modules.vad import VAD

    print("StreamingASR canlı testi. Konuşun; VAD konuşmayı algılayınca")
    print("ASR beslenmeye başlayacak, konuşma bitince FINAL basılacak.")
    print(f"Partial modeli: {SETTINGS.asr_partial_model}  |  Final modeli: {SETTINGS.asr_model_size}")
    print(f"Partial aralığı: {SETTINGS.asr_partial_interval_ms} ms.")
    print(f"Pre-roll: {SETTINGS.asr_preroll_ms} ms. Çıkmak için Ctrl+C.\n")

    mic = MicStream()
    vad = VAD()
    streaming_asr = StreamingASR()
    streaming_asr.load()  # model yüklemesi ilk konuşmadan önce bitsin

    mic.start()
    in_speech = False
    printed_partials = 0

    try:
        while True:
            chunk = mic.read_chunk(timeout=0.1)
            if chunk is None:
                continue

            # Konuşma durumundan bağımsız: VAD'ın geç haber verdiği
            # cümle başlarını telafi etmek için sürekli biriktir.
            streaming_asr.feed_preroll(chunk)
            vad.feed(chunk)

            just_started = False
            for event in vad.events:
                if event.kind == "SPEECH_START":
                    streaming_asr.reset()  # pre-roll sesiyle başlar
                    in_speech = True
                    just_started = True
                    printed_partials = 0
                    print("--- konuşma başladı ---")
                elif event.kind == "SPEECH_END":
                    in_speech = False
                    finalize_start = time.perf_counter()
                    result = streaming_asr.finalize()
                    finalize_ms = (time.perf_counter() - finalize_start) * 1000

                    partials_this_turn = streaming_asr.partials
                    first_partial_ms = partials_this_turn[0][0] if partials_this_turn else None

                    print(f'FINAL:   "{result.text}"')
                    print(
                        f"  no_speech_prob={result.no_speech_prob:.2f}  "
                        f"avg_logprob={result.avg_logprob:.2f}"
                    )
                    print(f"SPEECH_END → FINAL: {finalize_ms:.0f} ms")
                    if first_partial_ms is not None:
                        print(f"İlk partial        : {first_partial_ms:.0f} ms")
                    else:
                        print("İlk partial        : (partial üretilmedi)")
                    print()

            # Bu chunk SPEECH_START'ı tetiklediyse zaten reset() ile
            # pre-roll üzerinden eklendi; tekrar eklenmesin.
            if in_speech and not just_started:
                streaming_asr.feed(chunk)

            new_partials = streaming_asr.partials
            for elapsed_ms, text, _transcribe_ms, no_speech_prob, avg_logprob in new_partials[
                printed_partials:
            ]:
                print(
                    f'[{elapsed_ms:6.0f} ms] partial: "{text}"  '
                    f"(no_speech={no_speech_prob:.2f} logprob={avg_logprob:.2f})"
                )
            printed_partials = len(new_partials)
    except KeyboardInterrupt:
        pass
    finally:
        # Her adım kendi try/except'inde: biri patlarsa diğeri yine de
        # çalışıp mikrofonu/stream'i açık bırakmasın.
        try:
            streaming_asr.close()
        except Exception as exc:
            print(f"StreamingASR kapatılırken hata: {exc}", file=sys.stderr)
        try:
            mic.stop()
        except Exception as exc:
            print(f"MicStream kapatılırken hata: {exc}", file=sys.stderr)
        print("\nTest bitti.")


if __name__ == "__main__":
    main()
