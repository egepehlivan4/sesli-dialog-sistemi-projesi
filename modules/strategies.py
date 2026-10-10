"""
Barge-in stratejileri: asistan THINKING/SPEAK durumundayken kullanıcı
konuşmaya başlarsa, kesip kesmeyeceğimize bu sınıflar karar verir.

BargeInStrategy üç hook tanımlar; her biri şu üç değerden birini döner:
    INTERRUPT - hemen kes (SpeakerOut durur, LLM iptal edilir, LISTEN'a geçilir)
    WAIT      - henüz karar verme, daha fazla bilgi (partial/konuşma sonu) bekle
    IGNORE    - bu konuşmayı kesinti sayma, asistan konuşmaya devam etsin

Şimdilik tek strateji var: VadOnlyStrategy (konuşma başlar başlamaz
keser, metni ya da süreyi hiç dikkate almaz). İleride duration/semantic
stratejiler eklenecek; strateji config'teki BARGE_IN_STRATEGY ile seçilir.

Tek başına çalıştırma:
    python modules/strategies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

INTERRUPT = "INTERRUPT"
WAIT = "WAIT"
IGNORE = "IGNORE"


class BargeInStrategy:
    """Alt sınıflar bu üç metottan ilgili olanı override eder.
    Override edilmeyenler varsayılan olarak WAIT döner (karar erteleme)."""

    name = "base"

    def on_user_speech_start(self) -> str:
        return WAIT

    def on_partial(self, text: str) -> str:
        return WAIT

    def on_user_speech_end(self) -> str:
        return WAIT


class VadOnlyStrategy(BargeInStrategy):
    """En basit strateji: kullanıcı konuşmaya başladığı an kes."""

    name = "vad_only"

    def on_user_speech_start(self) -> str:
        return INTERRUPT


_STRATEGIES = {
    "vad_only": VadOnlyStrategy,
}


def create_strategy(name: str = None) -> BargeInStrategy:
    key = (name or SETTINGS.barge_in_strategy).strip().lower()
    try:
        cls = _STRATEGIES[key]
    except KeyError:
        raise ValueError(
            f"Bilinmeyen BARGE_IN_STRATEGY: {key!r}. Geçerli seçenekler: "
            f"{', '.join(_STRATEGIES)}"
        ) from None
    return cls()


def main() -> None:
    strategy = create_strategy()
    print(f"Aktif strateji (BARGE_IN_STRATEGY): {strategy.name}")
    print(f"on_user_speech_start() -> {strategy.on_user_speech_start()}")
    print(f"on_partial('merhaba')  -> {strategy.on_partial('merhaba')}")
    print(f"on_user_speech_end()   -> {strategy.on_user_speech_end()}")


if __name__ == "__main__":
    main()
