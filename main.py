"""
Giriş noktası: Controller'ı başlatır (mikrofon -> VAD -> StreamingASR ->
LLM -> TTS -> hoparlör, barge-in dahil). Açılışta modelleri yükler ve
LLM'i ısıtır, sonra canlı diyaloğu çalıştırır. Ctrl+C ile temiz kapanır
ve oturum özetini basar (bkz. modules/controller.py).

Çalıştırma:
    python main.py
"""

from modules.controller import Controller


def main() -> None:
    controller = Controller()

    print("Modeller yükleniyor ve LLM ısıtılıyor...")
    controller.load()
    print("Hazır. Konuşmaya başlayabilirsiniz (çıkmak için Ctrl+C).\n")

    controller.run()


if __name__ == "__main__":
    main()
