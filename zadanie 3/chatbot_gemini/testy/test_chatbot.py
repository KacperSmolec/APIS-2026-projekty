"""Testy jednostkowe rdzenia chatbota.

Testy NIE łączą się z siecią — używają atrap (fake) klienta Gemini, dzięki
czemu są szybkie, deterministyczne i działają bez klucza API.

Uruchomienie:
    python -m pytest testy/ -v
albo bez pytest:
    python testy/test_chatbot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Pozwala uruchomić plik bezpośrednio (dodaje katalog repo do ścieżki importów).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdzen.historia import HistoriaRozmowy
from rdzen.wyjatki import LimitKontekstuPrzekroczony, BladPolaczenia, BladOdpowiedziModelu


# Prosty licznik tokenów: 1 token = 1 słowo. Wystarczy do testów logiki.
def licznik_slow(tekst: str) -> int:
    return len(tekst.split())


def test_kontrola_kontekstu_usuwa_najstarsze():
    h = HistoriaRozmowy(prompt_systemowy="systemowy prompt", budzet_tokenow=10, licznik_tokenow=licznik_slow)
    for i in range(6):
        h.dodaj("user", f"wiadomosc numer {i}")  # 3 tokeny każda
    usuniete = h.przytnij_do_budzetu()
    assert usuniete > 0
    assert h.liczba_tokenow() <= 10
    # Prompt systemowy nigdy nie znika.
    assert h.prompt_systemowy == "systemowy prompt"


def test_za_dluga_pojedyncza_wiadomosc_zglasza_wyjatek():
    h = HistoriaRozmowy(prompt_systemowy="x", budzet_tokenow=5, licznik_tokenow=licznik_slow)
    h.dodaj("user", "to jest zdecydowanie zbyt dluga pojedyncza wiadomosc uzytkownika")
    try:
        h.przytnij_do_budzetu()
        assert False, "powinien polecieć wyjątek"
    except LimitKontekstuPrzekroczony:
        pass


def test_format_gemini():
    h = HistoriaRozmowy(prompt_systemowy="x", budzet_tokenow=100, licznik_tokenow=licznik_slow)
    h.dodaj("user", "czesc")
    h.dodaj("model", "witaj")
    fmt = h.do_formatu_gemini()
    assert fmt == [
        {"role": "user", "parts": [{"text": "czesc"}]},
        {"role": "model", "parts": [{"text": "witaj"}]},
    ]


# --- Testy obsługi błędów z atrapą klienta Gemini ---

class FakeOdpowiedz:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, tryb):
        self.tryb = tryb
        self.licznik_wywolan = 0

    def count_tokens(self, model, contents):
        class W:  # noqa: D401
            total_tokens = max(1, len(contents.split()))
        return W()

    def generate_content(self, model, contents, config):
        self.licznik_wywolan += 1
        if self.tryb == "ok":
            return FakeOdpowiedz("To jest odpowiedź testowa.")
        if self.tryb == "pusta":
            return FakeOdpowiedz("")
        if self.tryb == "wyjatek":
            raise ConnectionError("symulacja braku sieci")
        raise RuntimeError("nieznany tryb")


class FakeKlient:
    def __init__(self, tryb):
        self.models = FakeModels(tryb)


def _zbuduj_bota(tryb):
    from rdzen.konfiguracja import Ustawienia
    from rdzen.chatbot import ChatbotGemini
    ust = Ustawienia(klucz_api="testowy", liczba_prob=2, opoznienie_ponowienia_s=0.0)
    return ChatbotGemini(ustawienia=ust, klient=FakeKlient(tryb))


def test_poprawna_odpowiedz_zapisuje_historie():
    bot = _zbuduj_bota("ok")
    odp = bot.odpowiedz("Czym jest GIS?")
    assert "testowa" in odp
    assert len(bot.historia.wiadomosci) == 2  # user + model


def test_pusta_odpowiedz_modelu():
    bot = _zbuduj_bota("pusta")
    try:
        bot.odpowiedz("pytanie")
        assert False
    except BladOdpowiedziModelu:
        # Po błędzie wiadomość użytkownika jest wycofana — historia spójna.
        assert len(bot.historia.wiadomosci) == 0


def test_blad_polaczenia_po_wyczerpaniu_prob():
    bot = _zbuduj_bota("wyjatek")
    try:
        bot.odpowiedz("pytanie")
        assert False
    except BladPolaczenia:
        assert bot.klient.models.licznik_wywolan == 2  # liczba_prob = 2
        assert len(bot.historia.wiadomosci) == 0


if __name__ == "__main__":
    funkcje = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bledy = 0
    for f in funkcje:
        try:
            f()
            print(f"[OK]   {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bledy += 1
            print(f"[FAIL] {f.__name__}: {e}")
    print(f"\nZaliczone: {len(funkcje) - bledy}/{len(funkcje)}")
    sys.exit(1 if bledy else 0)
