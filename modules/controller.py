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

Tek başına çalıştırma (canlı diyalog):
    python modules/controller.py
"""

import json
import sys
import threading
import time
from pathlib import Path

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


class SessionLogger:
    """Her olayı zaman damgasıyla results/sessions/<zaman>.jsonl'a
    yazar. Hem ana thread hem worker thread'den çağrılır; kendi lock'u
    var."""

    def __init__(self, sessions_dir: Path = None):
        self.dir = Path(sessions_dir) if sessions_dir else SETTINGS.base_dir / "results" / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
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
        self.logger = SessionLogger()

        self._state_lock = threading.Lock()
        self.state = LISTEN
        self.history = []  # [{"role": "user"/"assistant", "content": ...}]

        self._worker_thread: threading.Thread | None = None
        self._worker_cancel = threading.Event()
        self._last_partial_count = 0

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
            cutoff_ms = (self.speaker.last_silence_at - interrupt_at) * 1000
        else:
            cutoff_ms = (time.perf_counter() - interrupt_at) * 1000

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

                if first_sentence:
                    first_sentence = False
                    self._mark_first_sound(play_call_time, speech_end_at)
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

    def _mark_first_sound(self, play_call_time: float, speech_end_at: float) -> None:
        """İlk cümlenin fiilen ne zaman duyulmaya başladığını ölçer
        (SpeakerOut'un ses callback thread'inde güncellediği
        last_sound_at üzerinden) ve yanıt gecikmesini kaydeder."""
        deadline = play_call_time + 0.5
        while (
            self.speaker.last_sound_at is None or self.speaker.last_sound_at < play_call_time
        ) and time.perf_counter() < deadline:
            time.sleep(0.001)

        actual_start = self.speaker.last_sound_at if self.speaker.last_sound_at else time.perf_counter()
        response_latency_ms = (actual_start - speech_end_at) * 1000
        self._response_latencies_ms.append(response_latency_ms)
        self.logger.log("response_latency", ms=response_latency_ms)

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

        self.logger.log("session_end", turns=self._turn_count, interrupts=self._interrupt_count)
        self.logger.close()
        self._print_summary()

    def _print_summary(self) -> None:
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

        print(f"\nOturum kaydı: {self.logger.path}")


def main() -> None:
    controller = Controller()
    print("Modeller yükleniyor ve LLM ısıtılıyor...")
    controller.load()
    print("Hazır. Konuşmaya başlayabilirsiniz (çıkmak için Ctrl+C).\n")
    controller.run()


if __name__ == "__main__":
    main()
