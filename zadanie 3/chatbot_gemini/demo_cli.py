"""Interaktywne demo chatbota w terminalu.

Uruchomienie:
    python demo_cli.py

Komendy w trakcie rozmowy:
    /reset  — czyści historię rozmowy
    /tokeny — pokazuje aktualną liczbę tokenów kontekstu
    /koniec — kończy program
"""
from __future__ import annotations

from rdzen import (
    ChatbotGemini,
    BladPolaczenia,
    BladOdpowiedziModelu,
    LimitKontekstuPrzekroczony,
)


def main() -> None:
    print("=" * 60)
    print(" GeoBot — chatbot geoinformatyczny (Gemini 2.5 Flash)")
    print(" Wpisz /koniec aby wyjść, /reset aby zacząć od nowa.")
    print("=" * 60)

    try:
        bot = ChatbotGemini()
    except RuntimeError as e:
        print(f"[BŁĄD KONFIGURACJI] {e}")
        return

    while True:
        try:
            pytanie = input("\nTy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDo zobaczenia!")
            break

        if not pytanie:
            continue
        if pytanie == "/koniec":
            print("Do zobaczenia!")
            break
        if pytanie == "/reset":
            bot.wyczysc_historie()
            print("[Historia wyczyszczona]")
            continue
        if pytanie == "/tokeny":
            print(f"[Kontekst: {bot.historia.liczba_tokenow()} tokenów]")
            continue

        try:
            odpowiedz = bot.odpowiedz(pytanie)
            print(f"\nGeoBot: {odpowiedz}")
        except LimitKontekstuPrzekroczony:
            print("\n[!] Twoja wiadomość jest zbyt długa. Skróć ją i spróbuj ponownie.")
        except BladPolaczenia:
            print("\n[!] Problem z połączeniem z modelem. Spróbuj ponownie za chwilę.")
        except BladOdpowiedziModelu:
            print("\n[!] Model nie zwrócił poprawnej odpowiedzi (możliwa blokada treści).")


if __name__ == "__main__":
    main()
