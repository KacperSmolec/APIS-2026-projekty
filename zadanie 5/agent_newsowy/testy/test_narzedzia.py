"""Testy jednostkowe narzędzi agenta (offline, bez sieci ani LLM).

Uruchomienie:
    python -m pytest testy/ -v
    python testy/test_narzedzia.py
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdzen import narzedzia


@dataclass
class FakeUstawienia:
    """Minimalny kontekst do testów narzędzi nieużywających LLM."""
    katalog_raportow: Path
    liczba_artykulow: int = 5
    klucz_newsapi: str = ""
    model_llm: str = "gemini-2.5-flash"
    temperatura: float = 0.3


def test_nazwa_pliku_pdf_ma_wymagany_format():
    katalog = Path(tempfile.mkdtemp())
    narzedzia.ustaw_kontekst(klient=None, ustawienia=FakeUstawienia(katalog_raportow=katalog))

    wynik = narzedzia.zapisz_raport_pdf("# Test\n\nTreść raportu.", "Sztuczna Inteligencja", "wysoka")
    assert wynik.startswith("Sukces")

    pliki = list(katalog.glob("*.pdf"))
    assert len(pliki) == 1
    nazwa = pliki[0].name
    # format: <data>_<temat>_<istotnosc>.pdf
    assert nazwa.endswith("_wysoka.pdf")
    assert "sztuczna_inteligencja" in nazwa
    # zaczyna się od daty RRRR-MM-DD
    assert nazwa[:4].isdigit() and nazwa[4] == "-"


def test_rss_parsowanie(monkeypatch=None):
    # Podstawiamy atrapę urlopen zwracającą przykładowy XML RSS.
    przykladowy_xml = b"""<?xml version="1.0"?>
    <rss><channel>
      <item><title>Tytul A</title><link>http://a</link>
            <pubDate>Mon, 01 Jan 2026</pubDate><description>&lt;b&gt;Opis A&lt;/b&gt;</description></item>
      <item><title>Tytul B</title><link>http://b</link>
            <pubDate>Tue, 02 Jan 2026</pubDate><description>Opis B</description></item>
    </channel></rss>"""

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return przykladowy_xml

    oryginalny = narzedzia.urlopen
    narzedzia.urlopen = lambda *a, **k: FakeResp()
    try:
        wynik = narzedzia._google_news_rss("cokolwiek", 5)
    finally:
        narzedzia.urlopen = oryginalny

    assert len(wynik) == 2
    assert wynik[0]["tytul"] == "Tytul A"
    assert wynik[0]["url"] == "http://a"
    assert "<b>" not in wynik[0]["opis"]   # tagi HTML usunięte


def test_dyspozytor_ma_cztery_narzedzia():
    # AgentNewsowy buduje mapę nazwa -> funkcja z dokładnie 4 narzędziami.
    from rdzen.agent import AgentNewsowy
    agent = AgentNewsowy()   # tworzy klienta, ale bez wywołań sieciowych
    assert set(agent._dyspozytor) == {
        "szukaj_wiadomosci", "podsumuj_wiadomosci", "ocen_istotnosc", "zapisz_raport_pdf"
    }


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
