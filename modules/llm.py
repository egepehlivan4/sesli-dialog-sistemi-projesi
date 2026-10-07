"""
LLM modülü: Ollama /api/chat endpoint'ini streaming modda kullanır.

URL ve model config.py'den okunur (varsayılan: http://localhost:11434,
qwen2.5:3b). Sistem promptu Türkçe ve cevapları kısa tutmaya zorlar
(sesli asistanda uzun cevap = uzun gecikme).

Tek başına çalıştırma (entegrasyon testi):
    python modules/llm.py
"""

import json
import re
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

SYSTEM_PROMPT = (
    "Sen Türkçe konuşan sesli bir asistansın. Yalnızca TEK kısa cümleyle "
    "cevap ver. Emin olmadığın bilgiyi uydurma."
)

# Düşük sıcaklık = daha az uydurma (hallucination), daha tutarlı cevap.
DEFAULT_TEMPERATURE = 0.3

# Bir cümlenin bittiğini kabul etmek için: nokta/ünlem/soru işareti(leri)
# ve ardından boşluk. Boşluk şartı, "3.5" gibi ondalık sayılarda veya daha
# token gelmeden erken bölünmeyi önler.
SENTENCE_END_RE = re.compile(r"[.!?]+\s+")


class LLM:
    """Ollama /api/chat üzerinden streaming sohbet."""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        system_prompt: str = None,
        temperature: float = None,
    ):
        self.base_url = (base_url or SETTINGS.llm_base_url).rstrip("/")
        self.model = model or SETTINGS.llm_model
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.timeout = SETTINGS.llm_timeout_s

        self._session = requests.Session()
        self._response = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    def _post(self, payload: dict, stream: bool):
        """/api/chat'e POST atar; bağlantı/HTTP hatalarını açık mesajlı
        RuntimeError'a çevirir. stream_tokens() ve warmup() tarafından
        paylaşılır."""
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat", json=payload, stream=stream, timeout=self.timeout
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            raise RuntimeError(
                f"Ollama'ya bağlanılamadı ({self.base_url}): {exc.__class__.__name__}. "
                "Önce 'ollama serve' ile sunucuyu başlatın ve modelin kurulu "
                f"olduğundan emin olun ('ollama pull {self.model}')."
            ) from exc

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            response.close()
            raise RuntimeError(
                f"Ollama isteği başarısız oldu ({exc}). "
                f"'{self.model}' modelinin kurulu olduğundan emin olun: "
                f"'ollama pull {self.model}'"
            ) from exc

        return response

    def warmup(self) -> float:
        """Kısa, tek token'lık bir istek göndererek modeli belleğe/GPU'ya
        yükler. Isınma süresini saniye olarak döndürür. Ölçümlerden ÖNCE
        çağrılmalı, yoksa soğuk başlangıç (model yükleme) süresi gerçek
        üretim ölçümlerine karışır."""
        start = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "merhaba"},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 1},
        }
        response = self._post(payload, stream=False)
        response.close()
        return time.perf_counter() - start

    def stream_tokens(self, messages: list):
        """messages: [{"role": "user"/"assistant", "content": "..."}].
        Sistem promptu otomatik eklenir. Token'ları tek tek üretir."""
        self._cancel_event.clear()

        full_messages = [{"role": "system", "content": self.system_prompt}] + list(messages)
        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }

        response = self._post(payload, stream=True)

        with self._lock:
            self._response = response

        try:
            for line in response.iter_lines():
                if self._cancel_event.is_set():
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
        except requests.exceptions.RequestException:
            if not self._cancel_event.is_set():
                raise
        finally:
            response.close()
            with self._lock:
                self._response = None

    def stream_sentences(self, messages: list):
        """Token'ları biriktirir, tamamlanan cümleleri (.!? ile biten) tek
        tek üretir. Amaç: ilk cümle biter bitmez TTS'e verilebilmesi."""
        buffer = ""
        for token in self.stream_tokens(messages):
            buffer += token
            while True:
                match = SENTENCE_END_RE.search(buffer)
                if not match:
                    break
                end = match.end()
                sentence = buffer[:end].strip()
                buffer = buffer[end:]
                if sentence:
                    yield sentence

        tail = buffer.strip()
        if tail:
            yield tail

    def cancel(self) -> None:
        """Üretimi anında durdurur ve HTTP bağlantısını kapatır.
        Barge-in olduğunda LLM'in boşuna üretmeye devam etmemesi içindir."""
        self._cancel_event.set()
        with self._lock:
            if self._response is not None:
                self._response.close()


def main() -> None:
    llm = LLM()

    print("Isınma isteği gönderiliyor (model yükleniyor)...")
    warmup_elapsed = llm.warmup()
    print(f"Isınma tamamlandı ({warmup_elapsed * 1000:.0f} ms)\n")

    question = "Türkiye'nin başkenti neresidir ve kısaca bir özelliğini söyler misin?"
    print(f"Soru: {question}\n")

    messages = [{"role": "user", "content": question}]

    print("--- stream_tokens (canlı token akışı) ---")
    request_start = time.perf_counter()
    first_token_time = None
    first_sentence_time = None
    token_count = 0
    buffer = ""
    full_text = ""

    for token in llm.stream_tokens(messages):
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        token_count += 1
        full_text += token
        buffer += token
        print(token, end="", flush=True)

        if first_sentence_time is None and SENTENCE_END_RE.search(buffer):
            first_sentence_time = now

    end_time = time.perf_counter()
    print("\n")

    ttft = first_token_time - request_start
    first_sentence_elapsed = (
        first_sentence_time - request_start if first_sentence_time is not None else None
    )
    decode_duration = end_time - first_token_time if first_token_time else 0.0
    tokens_per_sec = (token_count - 1) / decode_duration if decode_duration > 0 and token_count > 1 else 0.0

    print(f"TTFT (ilk token'a kadar)       : {ttft * 1000:.0f} ms")
    if first_sentence_elapsed is not None:
        print(f"İlk cümlenin tamamlanma süresi  : {first_sentence_elapsed * 1000:.0f} ms")
    print(f"Toplam token sayısı             : {token_count}")
    print(f"Token/saniye (TTFT sonrası)     : {tokens_per_sec:.1f}")

    print("\n--- stream_sentences + TTS + SpeakerOut (cümle cümle seslendirme) ---")
    from modules.audio_io import SpeakerOut
    from modules.tts import TTS

    tts = TTS()
    speaker = SpeakerOut()
    try:
        stream_start = time.perf_counter()
        first_speech_start = None
        sentence_no = 0

        for sentence in llm.stream_sentences(messages):
            sentence_no += 1
            arrived_at = time.perf_counter() - stream_start
            print(f"[{arrived_at * 1000:.0f} ms] Cümle {sentence_no}: {sentence}")

            audio, sample_rate = tts.synthesize(sentence)
            if first_speech_start is None:
                first_speech_start = time.perf_counter() - stream_start
                print(f"  -> konuşma {first_speech_start * 1000:.0f} ms'de başladı (cevabın tamamı beklenmedi)")

            speaker.play(audio, sample_rate)
            speaker.wait()

        total_elapsed = time.perf_counter() - stream_start
        print(f"\nToplam cevap süresi: {total_elapsed * 1000:.0f} ms")
        if first_speech_start is not None:
            print(f"Konuşma, cevabın tamamından {(total_elapsed - first_speech_start) * 1000:.0f} ms önce başladı.")
    finally:
        # Stream __init__'te açılıp kalıcı kaldığı için (bkz. audio_io.py),
        # burada kapatılmazsa süreç PortAudio stream'i ile birlikte asılı
        # kalabilir -- hata/kesinti olsa da close() garanti çalışsın.
        speaker.close()


if __name__ == "__main__":
    main()
