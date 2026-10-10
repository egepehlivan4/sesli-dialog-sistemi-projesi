"""
Ses I/O modülü: mikrofondan sürekli okuma (MicStream) ve hoparlöre
anında kesilebilir çalma (SpeakerOut).

sounddevice callback'leri kendi ses thread'inde çalışır. Callback'ler
burada SADECE queue.Queue'ya veri koyar/alır; ağır iş (VAD, ASR, ...)
read_chunk() ile alan tarafta yapılmalı, yoksa ses kopar.

Tek başına çalıştırma (entegrasyon testi):
    python modules/audio_io.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS


class MicStream:
    """sounddevice.InputStream ile sürekli mikrofon okuma.

    Callback gelen chunk'ı kopyalayıp queue'ya koyar; tüketici tarafı
    read_chunk() ile çeker.
    """

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        dtype: str = None,
        device: str = None,
        chunk_ms: int = None,
    ):
        self.sample_rate = sample_rate or SETTINGS.sample_rate
        self.channels = channels or SETTINGS.channels
        self.dtype = dtype or SETTINGS.audio_dtype
        self.device = device if device is not None else SETTINGS.input_device
        self.chunk_ms = chunk_ms or SETTINGS.audio_chunk_ms
        self.blocksize = int(self.sample_rate * self.chunk_ms / 1000)

        self._queue = queue.Queue()
        self._stream = None

        # input_overflow: PortAudio'nun bizim tüketemediğimiz için attığı
        # örnekler -- SessionRecorder'ın sol kanalının neden kaymış
        # olabileceğini teşhis etmek için sayılır (bkz. Controller.run()).
        self.overflow_count = 0
        self._overflow_lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"MicStream durumu: {status}", file=sys.stderr)
            if getattr(status, "input_overflow", False):
                with self._overflow_lock:
                    self.overflow_count += 1
        self._queue.put(indata.copy())

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device,
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None

    def read_chunk(self, timeout: float = None):
        """Kuyruktan bir chunk alır. timeout saniye içinde veri gelmezse
        None döner."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Basit lineer interpolasyonla yeniden örnekleme (ek bağımlılık gerektirmez)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    orig_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_times, orig_times, audio).astype(np.float32)


class SpeakerOut:
    """sounddevice.OutputStream'i program boyunca açık tutan, anında
    susturulabilir ses çalma.

    Stream __init__ içinde BİR KEZ açılır ve close() çağrılana kadar
    sürekli çalışır. Kuyruk boşken callback sessizlik (sıfır) yazar.
    Bu sayede:
      - stop() stream'i abort/kapat etmez, sadece kuyruğu temizler ->
        ses anında kesilir (barge-in) ama stream açık kalır.
      - stop()'tan sonraki play() çağrısı stream'i yeniden açmak
        zorunda kalmadığı için gecikmesiz başlar.
      - close() stream'i gerçekten durdurup kapatan ayrı bir metottur
        (program sonunda çağrılmalı).
    """

    # piper TTS'in varsayılan Türkçe sesinin kendi örnekleme hızı.
    DEFAULT_SAMPLE_RATE = 22050

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        dtype: str = None,
        device: str = None,
        chunk_ms: int = None,
    ):
        self.sample_rate = sample_rate or self.DEFAULT_SAMPLE_RATE
        self.channels = channels or SETTINGS.channels
        self.dtype = dtype or SETTINGS.audio_dtype
        self.device = device if device is not None else SETTINGS.output_device
        self.chunk_ms = chunk_ms or SETTINGS.audio_chunk_ms
        self.blocksize = max(1, int(self.sample_rate * self.chunk_ms / 1000))

        self._queue = queue.Queue()
        self._finished = threading.Event()
        self._finished.set()  # başta kuyruk boş -> "bitmiş" durumda
        self.last_silence_at: float | None = None
        self.last_sound_at: float | None = None

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device,
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    @property
    def queued_chunks(self) -> int:
        return self._queue.qsize()

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(f"SpeakerOut durumu: {status}", file=sys.stderr)
        try:
            block = self._queue.get_nowait()
        except queue.Empty:
            outdata.fill(0)
            self._finished.set()
            self.last_silence_at = time.perf_counter()
            return

        self.last_sound_at = time.perf_counter()
        n = len(block)
        if n < frames:
            outdata[:n, :] = block.reshape(-1, 1)
            outdata[n:, :] = 0
        else:
            outdata[:, :] = block[:frames].reshape(-1, 1)

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Sesi (gerekirse stream'in sabit hızına yeniden örnekleyip)
        chunk_ms'lik bloklara bölüp kuyruğa koyar. Bloklamaz; stream
        zaten açık olduğu için hemen çalmaya başlar."""
        self.stop()  # olası eski kuyruk kalıntısını temizle

        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.sample_rate:
            audio = _resample(audio, sample_rate, self.sample_rate)

        self._finished.clear()
        for start_idx in range(0, len(audio), self.blocksize):
            self._queue.put(audio[start_idx : start_idx + self.blocksize])

    def wait(self) -> None:
        """Çalmanın doğal olarak bitmesini bekler (bloklar)."""
        self._finished.wait()

    def stop(self) -> None:
        """Kuyruğu temizler; ses ANINDA kesilir. Stream açık kalır."""
        with self._queue.mutex:
            self._queue.queue.clear()
        self._finished.set()

    def close(self) -> None:
        """Stream'i gerçekten durdurur ve kapatır (program sonu)."""
        self.stop()
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None


def main() -> None:
    print("Dinliyorum, konuşun (5 saniye)...")
    mic = MicStream()
    mic.start()

    start = time.perf_counter()
    last_reported_second = 0
    peak_this_second = 0.0
    while time.perf_counter() - start < 5.0:
        chunk = mic.read_chunk(timeout=0.1)
        if chunk is not None:
            peak_this_second = max(peak_this_second, float(np.max(np.abs(chunk))))

        elapsed = time.perf_counter() - start
        if int(elapsed) > last_reported_second:
            last_reported_second = int(elapsed)
            print(f"  {last_reported_second}. saniye - maksimum genlik: {peak_this_second:.4f}")
            peak_this_second = 0.0

    mic.stop()
    print("Mikrofon testi bitti.\n")

    print("TTS ile uzun bir cümle üretiliyor...")
    from modules.tts import TTS

    tts = TTS()

    text1 = (
        "Bu cümle bilerek oldukça uzun yazıldı, çünkü oynatmanın birinci "
        "saniyesinde durdurma çağrısı yapacağız ve sesin gerçekten anında "
        "kesilip kesilmediğini bu şekilde gözlemleyebileceğiz."
    )
    audio1, sr1 = tts.synthesize(text1)
    duration1 = len(audio1) / sr1
    print(f"1. cümle üretildi, süresi: {duration1:.2f} s")

    speaker = SpeakerOut()
    print(f"SpeakerOut stream açıldı ({speaker.sample_rate} Hz, kalıcı).")
    print("Çalma başlıyor...")
    speaker.play(audio1, sr1)

    time.sleep(1.0)
    pending = speaker.queued_chunks
    print(f"1. saniyede stop() çağrılıyor (barge-in simülasyonu), kuyrukta {pending} blok vardı...")

    stop_call_time = time.perf_counter()
    speaker.stop()
    stop_call_elapsed = time.perf_counter() - stop_call_time

    # Ses callback thread'inin fiilen sessizliğe geçtiği anı bekle (poll).
    while speaker.last_silence_at is None or speaker.last_silence_at < stop_call_time:
        time.sleep(0.001)
    silence_latency = speaker.last_silence_at - stop_call_time

    print(f"stop() çağrısının kendisi {stop_call_elapsed * 1000:.2f} ms sürdü.")
    print(f"stop() -> sesin fiilen kesilmesi (ses callback thread'inde ölçüldü): {silence_latency * 1000:.2f} ms")
    print(f"Kuyrukta kalan blok sayısı: {speaker.queued_chunks} (0 olmalı -> anında kesildi)")
    print(f"Çalınmadan kesilen ses: ~{max(duration1 - 1.0, 0.0):.2f} s\n")

    text2 = "İkinci cümle hemen şimdi, gecikmesiz başlıyor."
    audio2, sr2 = tts.synthesize(text2)
    duration2 = len(audio2) / sr2
    print(f"2. cümle üretildi, süresi: {duration2:.2f} s")

    play_call_time = time.perf_counter()
    speaker.play(audio2, sr2)
    play_call_elapsed = time.perf_counter() - play_call_time

    # Ses callback thread'inin fiilen sesli çıkışa geçtiği anı bekle (poll).
    while speaker.last_sound_at is None or speaker.last_sound_at < play_call_time:
        time.sleep(0.001)
    start_latency = speaker.last_sound_at - play_call_time

    print(f"play() çağrısının kendisi {play_call_elapsed * 1000:.2f} ms sürdü.")
    print(
        f"play() -> sesin fiilen başlaması (ses callback thread'inde ölçüldü): "
        f"{start_latency * 1000:.2f} ms (stream zaten açıktı, yeniden açma gecikmesi yok)."
    )

    speaker.wait()
    print("2. cümle çalması tamamlandı.")

    speaker.close()
    print("SpeakerOut stream kapatıldı.")


if __name__ == "__main__":
    main()
