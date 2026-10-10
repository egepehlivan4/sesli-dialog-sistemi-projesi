"""
Controller: mikrofon -> VAD -> StreamingASR -> LLM -> TTS -> hoparlör
hattını bir durum makinesiyle (LISTEN/THINKING/SPEAK) birleştirir ve
barge-in (kesinti) stratejisini uygular.

Mimari (büyük bir asyncio dönüşümü YOK, mevcut thread tabanlı modüller
kullanılıyor):
  - ANA thread (run()): mikrofon okur, VAD'a besler, StreamingASR'a
    besler (pre-roll dahil, state'ten BAĞIMSIZ). state sadece burada
    VAD olaylarına göre değiştirilir (bir Lock ile korunur).
  - WORKER thread (_response_worker, her turda yeniden oluşturulur):
    LLM.stream_sentences -> her cümle için TTS.synthesize ->
    SpeakerOut.play/wait. Döngünün başında HER cümle için kendi
    cancel_event'ini kontrol eder (iptal edilebilir worker).

Barge-in: THINKING/SPEAK sırasında SPEECH_START gelirse strateji
sorulur. INTERRUPT ise: SpeakerOut.stop(), LLM.cancel(), worker'ın
cancel_event'i set edilir, state hemen LISTEN'a döner (worker burada
BEKLENMEZ ki kullanıcının yeni konuşması ASR'a gecikmeden akmaya devam
etsin). Bir sonraki SPEECH_END'de (finalize()'dan hemen önce, zaten
doğal bir bekleme noktası) yeni worker spawn edilmeden önce eskisi
join() ile toparlanır -- böylece iki worker aynı anda SpeakerOut'a
yazmaz.

Konuşma geçmişine yalnızca GERÇEKTEN çalınan cümleler eklenir; kesilen
cümlenin sonuna " [kesildi]" eklenir.

Oturum ses kaydı (SessionRecorder): mikrofon (sol kanal) ve hoparlöre
giden ses (sağ kanal, asistan) aynı perf_counter() eksenine göre
hizalanıp oturum sonunda tek bir stereo WAV'a yazılır -- bkz.
SessionRecorder docstring'i için hizalama açıklaması.

Tek başına çalıştırma (canlı diyalog):
    python modules/controller.py
"""

import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS
from modules.asr import StreamingASR
from modules.audio_io import MicStream, SpeakerOut
from modules.llm import LLM
from modules.strategies import INTERRUPT, create_strategy
from modules.tts import TTS
from modules.vad import VAD

LISTEN = "LISTEN"
THINKING = "THINKING"
SPEAK = "SPEAK"


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Basit lineer interpolasyonla yeniden örnekleme (ek bağımlılık
    gerektirmez; modules/asr.py ve modules/audio_io.py'deki kopyalarla
    aynı -- bu modüller kendi kendine yeterli kalsın diye paylaşılmıyor)."""
    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    orig_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_times, orig_times, audio).astype(np.float32)


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, sample_rate: int) -> None:
    """İki mono float32 [-1, 1] kanalı interleave edip 16-bit PCM
    stereo WAV olarak yazar (stdlib wave, ek bağımlılık yok)."""
    n = max(len(left), len(right))
    left = np.pad(left, (0, n - len(left)))
    right = np.pad(right, (0, n - len(right)))
    stereo = np.clip(np.stack([left, right], axis=1), -1.0, 1.0)
    pcm16 = (stereo * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


class SessionRecorder:
    """Oturumun mikrofon (sol kanal) ve hoparlöre giden (sağ kanal,
    asistan) seslerini tek bir stereo WAV'da birleştirir.

    Hizalama: sol kanal, mikrofon chunk'larının VARIŞ SIRASINA göre art
    arda eklenmesiyle oluşur -- mikrofon akışı sürekli/boşluksuz
    olduğu için bu, oturum başlangıcından itibaren geçen gerçek süreyi
    doğru yansıtır (örnek sayısı / örnekleme hızı = zaman). Sağ kanal
    için her cümlenin SpeakerOut'ta GERÇEKTEN çalmaya başladığı an
    (last_sound_at; ses callback thread'i tarafından yazılır, bkz.
    Controller._poll_actual_sound_start) ile aynı oturum başlangıcı
    (__init__'te alınan perf_counter()) arasındaki fark örnek indeksine
    çevrilip ses o noktaya yerleştirilir. İki kanal da AYNI
    perf_counter() epoğuna bağlı olduğu için ±1 ses bloğu (chunk_ms,
    varsayılan 20 ms) hassasiyetinde hizalanmış olur -- hedeflenen
    ±20 ms budur.

    Ses callback'lerinde İŞ YAPILMAZ: add_mic_chunk/add_speaker_audio
    sadece listeye ekler (append); resample/karıştırma/WAV yazma hep
    oturum sonunda save() içinde yapılır.

    Sol kanal başlangıç ofseti: __init__'teki t0 ile ilk mikrofon
    parçasının FİİLEN geldiği an arasında (model yükleme/ısınma süresi
    yüzünden) saniyeler mertebesinde boşluk olabilir. Bu boşluk
    kaydedilmezse sol kanal o kadar ERKEN kaymış olur (sağ kanal zaten
    play anına göre offsetlendiği için bu sorunu yaşamaz). Bunu telafi
    etmek için ilk parçanın varış zamanı tutulur; save() sol kanalın
    başına (varış_zamanı - t0 - ilk_parça_süresi) kadar sessizlik ekler,
    sonraki parçalar yine art arda (boşluksuz) eklenir.

    Kesinti: mark_interrupt(), SpeakerOut'un fiilen sustuğu anı bildirir.
    Bir cümlenin audio'su add_speaker_audio() ile TAMAMI kaydedilmiş
    olsa bile (worker kesintiyi o anda henüz fark etmemiş olabilir),
    save() o cümlenin oynatma penceresine denk gelen bir kesinti varsa
    sesi tam o noktada keser -- kayıt gerçekten DUYULAN sesi yansıtsın
    diye. (Konuşma geçmişindeki "[kesildi]" işaretlemesi bundan
    bağımsız, ayrı bir mekanizma.)"""

    def __init__(self, sample_rate: int = None):
        self.sample_rate = sample_rate or SETTINGS.sample_rate
        self._session_start = time.perf_counter()
        self._first_mic_arrival = None  # ilk add_mic_chunk() çağrısının perf_counter() anı
        self._mic_chunks = []  # [(np.ndarray, sample_rate), ...] varış sırasına göre
        self._speaker_events = []  # [(offset_s, np.ndarray, sample_rate), ...]
        self._interrupts = []  # [offset_s, ...]
        self._lock = threading.Lock()

    def add_mic_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            if self._first_mic_arrival is None:
                self._first_mic_arrival = time.perf_counter()
            self._mic_chunks.append((chunk, sample_rate))

    def add_speaker_audio(self, played_at: float, audio: np.ndarray, sample_rate: int) -> None:
        offset_s = played_at - self._session_start
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        with self._lock:
            self._speaker_events.append((offset_s, audio, sample_rate))

    def mark_interrupt(self, interrupted_at: float) -> None:
        offset_s = interrupted_at - self._session_start
        with self._lock:
            self._interrupts.append(offset_s)

    def save(self, path: Path) -> None:
        with self._lock:
            mic_chunks = list(self._mic_chunks)
            speaker_events = list(self._speaker_events)
            interrupts = sorted(self._interrupts)
            first_mic_arrival = self._first_mic_arrival

        left_parts = [_resample(chunk, sr, self.sample_rate) for chunk, sr in mic_chunks]

        if left_parts and first_mic_arrival is not None:
            first_chunk_duration_s = len(left_parts[0]) / self.sample_rate
            silence_s = (first_mic_arrival - self._session_start) - first_chunk_duration_s
            silence_samples = max(0, int(round(silence_s * self.sample_rate)))
            left_parts.insert(0, np.zeros(silence_samples, dtype=np.float32))

        left = np.concatenate(left_parts) if left_parts else np.zeros(0, dtype=np.float32)

        right = np.zeros(len(left), dtype=np.float32)
        for offset_s, audio, sr in speaker_events:
            audio_16k = _resample(audio, sr, self.sample_rate)

            # Bu cümle çalarken bir kesinti olduysa, kayıtta da tam o
            # anda kes -- gerçekte duyulmayan ses kayıtta görünmesin.
            natural_end_s = offset_s + len(audio_16k) / self.sample_rate
            cutoffs = [t for t in interrupts if offset_s < t < natural_end_s]
            if cutoffs:
                keep_samples = max(0, int(round((min(cutoffs) - offset_s) * self.sample_rate)))
                audio_16k = audio_16k[:keep_samples]

            start_sample = max(0, int(round(offset_s * self.sample_rate)))
            end_sample = start_sample + len(audio_16k)
            if end_sample > len(right):
                right = np.pad(right, (0, end_sample - len(right)))
            right[start_sample:end_sample] += audio_16k  # nadiren çakışırsa topla

        path.parent.mkdir(parents=True, exist_ok=True)
        _write_stereo_wav(path, left, right, self.sample_rate)


class SessionLogger:
    """Her olayı zaman damgasıyla results/sessions/<zaman>.jsonl'a
    yazar. Hem ana thread hem worker thread'den çağrılır; kendi lock'u
    var."""

    def __init__(self, session_id: str, sessions_dir: Path = None):
        self.dir = Path(sessions_dir) if sessions_dir else SETTINGS.base_dir / "results" / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{session_id}.jsonl"
        self._file = open(self.path, "a", encoding="utf-8")
        self._start = time.perf_counter()
        self._lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        record = {"elapsed_s": round(time.perf_counter() - self._start, 3), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()


class Controller:
    def __init__(self, ui: callable = None):
        self._ui = ui or print

        self.mic = MicStream()
        self.vad = VAD()
        self.asr = StreamingASR()
        self.llm = LLM()
        self.tts = TTS()
        self.speaker = SpeakerOut()
        self.strategy = create_strategy()

        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.logger = SessionLogger(self.session_id)
        self.recorder = SessionRecorder()

        self._state_lock = threading.Lock()
        self.state = LISTEN
        self.history = []  # [{"role": "user"/"assistant", "content": ...}]

        self._worker_thread: threading.Thread | None = None
        self._worker_cancel = threading.Event()
        self._last_partial_count = 0
        self._last_overflow_count = 0

        self._turn_count = 0
        self._interrupt_count = 0
        self._response_latencies_ms = []
        self._interrupt_latencies_ms = []

    # --- kurulum -----------------------------------------------------

    def load(self) -> None:
        """Açılışta modelleri yükler ve LLM'i ısıtır (ilk tur bu
        yüzden yavaş olmasın diye)."""
        self.asr.load()
        self.llm.warmup()

    # --- state yardımcıları -------------------------------------------

    def _get_state(self) -> str:
        with self._state_lock:
            return self.state

    def _set_state(self, new_state: str) -> None:
        with self._state_lock:
            old_state = self.state
            self.state = new_state
        if old_state != new_state:
            self.logger.log("state_change", frm=old_state, to=new_state)
            self._ui(f"[{new_state}]")

    # --- ana döngü ------------------------------------------------------

    def run(self) -> None:
        self.mic.start()
        self.logger.log("session_start", strategy=self.strategy.name)
        self._ui(f"[{self.state}]")

        try:
            while True:
                chunk = self.mic.read_chunk(timeout=0.1)
                if chunk is None:
                    continue

                overflow_count = self.mic.overflow_count
                if overflow_count > self._last_overflow_count:
                    self.logger.log(
                        "mic_overflow", new=overflow_count - self._last_overflow_count, total=overflow_count
                    )
                    self._last_overflow_count = overflow_count

                self.recorder.add_mic_chunk(chunk, self.mic.sample_rate)

                # Her zaman, state'ten bağımsız: pre-roll telafi ve VAD.
                self.asr.feed_preroll(chunk)
                self.vad.feed(chunk)

                just_started = False
                for event in self.vad.events:
                    self.logger.log("vad_event", kind=event.kind)
                    if event.kind == "SPEECH_START":
                        self._handle_speech_start()
                        just_started = True
                    elif event.kind == "SPEECH_END":
                        self._handle_speech_end()

                if self.vad.in_speech and not just_started:
                    self.asr.feed(chunk)

                self._check_pending_partials()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _handle_speech_start(self) -> None:
        self._last_partial_count = 0
        state = self._get_state()

        if state == LISTEN:
            self.asr.reset()
            return

        # THINKING ya da SPEAK: asistan konuşuyor/düşünüyor, kullanıcı
        # araya girdi -- stratejiye sor.
        decision = self.strategy.on_user_speech_start()
        self.logger.log("barge_in_decision", hook="on_user_speech_start", decision=decision)

        # Karar ne olursa olsun kullanıcının konuşması KAYBOLMASIN diye
        # ASR'ı bu yeni konuşma için hemen resetleyip (pre-roll dahil)
        # beslemeye başlıyoruz.
        self.asr.reset()

        if decision == INTERRUPT:
            self._do_interrupt()

    def _handle_speech_end(self) -> None:
        self.logger.log("speech_end")
        state = self._get_state()

        if state != LISTEN:
            decision = self.strategy.on_user_speech_end()
            self.logger.log("barge_in_decision", hook="on_user_speech_end", decision=decision)
            if decision != INTERRUPT:
                # Asistan konuşmaya devam ediyor; bu konuşmayı yok say.
                return
            self._do_interrupt()

        speech_end_at = time.perf_counter()
        result = self.asr.finalize()
        self.logger.log(
            "asr_final", text=result.text, no_speech_prob=result.no_speech_prob, avg_logprob=result.avg_logprob
        )

        text = result.text.strip()
        if not text:
            self._ui("(konuşma algılanamadı, dinlemeye devam)")
            return

        self._ui(f"Sen: {text}")

        # Yeni worker'ı başlatmadan önce eskisinin (varsa, bir kesinti
        # sonrası) toparlanmasını bekle -- aynı anda iki worker
        # SpeakerOut/LLM'e dokunmasın.
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

        messages = list(self.history) + [{"role": "user", "content": text}]
        self._turn_count += 1
        self._set_state(THINKING)

        self._worker_cancel = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._response_worker,
            args=(text, messages, speech_end_at, self._worker_cancel),
            daemon=True,
        )
        self._worker_thread.start()

    def _check_pending_partials(self) -> None:
        """THINKING/SPEAK sırasında (henüz kesinti kararı verilmemiş bir
        konuşma varsa) yeni partial geldikçe stratejiye sorar. VadOnlyStrategy
        speech_start'ta zaten karar verdiği için bu yol pratikte hiç
        çalışmaz; ileride WAIT dönebilecek stratejiler için hazır bekler."""
        if self._get_state() == LISTEN or not self.vad.in_speech:
            return

        partials = self.asr.partials
        if len(partials) <= self._last_partial_count:
            return
        self._last_partial_count = len(partials)

        latest_text = partials[-1][1]
        decision = self.strategy.on_partial(latest_text)
        self.logger.log("barge_in_decision", hook="on_partial", decision=decision)
        if decision == INTERRUPT:
            self._do_interrupt()

    def _do_interrupt(self) -> None:
        interrupt_at = time.perf_counter()
        self._worker_cancel.set()
        self.llm.cancel()
        self.speaker.stop()
        self._set_state(LISTEN)

        deadline = interrupt_at + 0.5
        while (
            self.speaker.last_silence_at is None or self.speaker.last_silence_at < interrupt_at
        ) and time.perf_counter() < deadline:
            time.sleep(0.001)

        if self.speaker.last_silence_at is not None and self.speaker.last_silence_at >= interrupt_at:
            silence_at = self.speaker.last_silence_at
            cutoff_ms = (silence_at - interrupt_at) * 1000
        else:
            silence_at = time.perf_counter()
            cutoff_ms = (silence_at - interrupt_at) * 1000

        # Oturum kaydı: o an çalmakta olan cümle, kayıtta da (gerçekte
        # duyulduğu gibi) tam bu noktada kesilsin.
        self.recorder.mark_interrupt(silence_at)

        self._interrupt_count += 1
        self._interrupt_latencies_ms.append(cutoff_ms)
        self.logger.log("interrupt", cutoff_ms=cutoff_ms)
        self._ui(f"⚡ KESİLDİ ({cutoff_ms:.0f} ms)")

    # --- worker (LLM -> TTS -> SpeakerOut) --------------------------------

    def _response_worker(
        self, user_text: str, messages: list, speech_end_at: float, cancel_event: threading.Event
    ) -> None:
        self.logger.log("llm_start")
        played_sentences = []
        interrupted = False
        first_sentence = True

        try:
            for sentence in self.llm.stream_sentences(messages):
                if cancel_event.is_set():
                    interrupted = True
                    break
                self.logger.log("llm_sentence", text=sentence)

                audio, sample_rate = self.tts.synthesize(sentence)
                if cancel_event.is_set():
                    interrupted = True
                    break

                play_call_time = time.perf_counter()
                self.speaker.play(audio, sample_rate)
                actual_start = self._poll_actual_sound_start(play_call_time)
                self.recorder.add_speaker_audio(actual_start, audio, sample_rate)

                if first_sentence:
                    first_sentence = False
                    response_latency_ms = (actual_start - speech_end_at) * 1000
                    self._response_latencies_ms.append(response_latency_ms)
                    self.logger.log("response_latency", ms=response_latency_ms)
                    self._set_state(SPEAK)

                self.logger.log("tts_start", text=sentence)
                self.speaker.wait()
                self.logger.log("tts_end", text=sentence)

                if cancel_event.is_set():
                    played_sentences.append(sentence + " [kesildi]")
                    interrupted = True
                    break

                played_sentences.append(sentence)
                self._ui(f"Asistan: {sentence}")
        except RuntimeError as exc:
            self.logger.log("llm_error", error=str(exc))
            self._ui(f"[LLM hata: {exc}]")
        finally:
            self.history.append({"role": "user", "content": user_text})
            if played_sentences:
                self.history.append({"role": "assistant", "content": " ".join(played_sentences)})
            self.logger.log("turn_end", interrupted=interrupted, played=len(played_sentences))
            if not interrupted:
                self._set_state(LISTEN)

    def _poll_actual_sound_start(self, play_call_time: float) -> float:
        """play() çağrıldıktan sonra sesin ses callback thread'inde
        FİİLEN başladığı anı (last_sound_at) bekler -- hem yanıt
        gecikmesi ölçümü hem de oturum kaydındaki hizalama için
        kullanılır."""
        deadline = play_call_time + 0.5
        while (
            self.speaker.last_sound_at is None or self.speaker.last_sound_at < play_call_time
        ) and time.perf_counter() < deadline:
            time.sleep(0.001)
        return self.speaker.last_sound_at if self.speaker.last_sound_at else time.perf_counter()

    # --- kapanış -------------------------------------------------------

    def _shutdown(self) -> None:
        try:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_cancel.set()
                self.llm.cancel()
                self.speaker.stop()
                self._worker_thread.join(timeout=2.0)
        except Exception as exc:
            print(f"Worker kapatılırken hata: {exc}", file=sys.stderr)

        for label, closer in (
            ("StreamingASR", self.asr.close),
            ("SpeakerOut", self.speaker.close),
            ("MicStream", self.mic.stop),
        ):
            try:
                closer()
            except Exception as exc:
                print(f"{label} kapatılırken hata: {exc}", file=sys.stderr)

        wav_path = self.logger.dir / f"{self.session_id}.wav"
        try:
            self.recorder.save(wav_path)
        except Exception as exc:
            print(f"Oturum ses kaydı yazılamadı: {exc}", file=sys.stderr)
            wav_path = None

        self.logger.log("session_end", turns=self._turn_count, interrupts=self._interrupt_count)
        self.logger.close()
        self._print_summary(wav_path)

    def _print_summary(self, wav_path: Path = None) -> None:
        print("\n=== Oturum özeti ===")
        print(f"Strateji       : {self.strategy.name}")
        print(f"Tur sayısı     : {self._turn_count}")
        print(f"Kesinti sayısı : {self._interrupt_count}")

        if self._response_latencies_ms:
            avg = sum(self._response_latencies_ms) / len(self._response_latencies_ms)
            print(f"\nYanıt gecikmesi (SPEECH_END -> ilk ses), ortalama {avg:.0f} ms:")
            for i, ms in enumerate(self._response_latencies_ms, 1):
                print(f"  tur {i}: {ms:.0f} ms")

        if self._interrupt_latencies_ms:
            avg = sum(self._interrupt_latencies_ms) / len(self._interrupt_latencies_ms)
            print(f"\nKesinti gecikmesi (SPEECH_START -> sessizlik), ortalama {avg:.0f} ms:")
            for i, ms in enumerate(self._interrupt_latencies_ms, 1):
                print(f"  kesinti {i}: {ms:.0f} ms")

        print(f"\nOturum log kaydı: {self.logger.path}")
        if wav_path is not None:
            print(f"Oturum ses kaydı: {wav_path}")


def main() -> None:
    controller = Controller()
    print("Modeller yükleniyor ve LLM ısıtılıyor...")
    controller.load()
    print("Hazır. Konuşmaya başlayabilirsiniz (çıkmak için Ctrl+C).\n")
    controller.run()


if __name__ == "__main__":
    main()
