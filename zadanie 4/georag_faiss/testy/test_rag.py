"""Testy jednostkowe komponentów RAG (offline, bez sieci).

Sprawdzają logikę dzielenia tekstu oraz bazy wektorowej FAISS na sztucznych
danych. Uruchomienie:
    python -m pytest testy/ -v
    python testy/test_rag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rdzen.wczytywanie import StronaDokumentu
from rdzen.dzielenie import DzielnikTekstu, Fragment
from rdzen.baza_wektorowa import BazaWektorowaFAISS
from rdzen.wyjatki import BrakIndeksu


def test_dzielnik_pakuje_zdania_w_fragmenty():
    tekst = " ".join(f"To jest zdanie numer {i} w tym akapicie." for i in range(40))
    strona = StronaDokumentu(tekst=tekst, zrodlo="plik.pdf", strona=1)
    dzielnik = DzielnikTekstu(docelowa_dlugosc=30, zakladka_zdan=1)
    fragmenty = dzielnik.podziel([strona])
    assert len(fragmenty) > 1                     # długi tekst -> wiele fragmentów
    assert all(isinstance(f, Fragment) for f in fragmenty)
    assert all(f.zrodlo == "plik.pdf" for f in fragmenty)


def test_dzielnik_zachowuje_zakladke():
    # Dwa fragmenty powinny współdzielić co najmniej jedno zdanie (overlap).
    zdania = [f"Zdanie {i} jakąś treścią wypełnione tutaj." for i in range(20)]
    strona = StronaDokumentu(tekst=" ".join(zdania), zrodlo="a.pdf", strona=2)
    dzielnik = DzielnikTekstu(docelowa_dlugosc=25, zakladka_zdan=2)
    fr = dzielnik.podziel([strona])
    if len(fr) >= 2:
        koniec = set(fr[0].tekst.split(".")[-3:])
        poczatek = set(fr[1].tekst.split(".")[:3])
        assert koniec & poczatek                  # istnieje wspólny fragment zdania


def test_faiss_zwraca_najblizszy():
    baza = BazaWektorowaFAISS(wymiar=3)
    fragmenty = [
        Fragment("czerwony", "p.pdf", 1, 0),
        Fragment("zielony", "p.pdf", 1, 1),
        Fragment("niebieski", "p.pdf", 1, 2),
    ]
    wektory = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    baza.zbuduj(wektory, fragmenty)
    zapytanie = np.array([0.9, 0.1, 0.0], dtype="float32")
    wyniki = baza.szukaj(zapytanie, k=1)
    assert wyniki[0][0].tekst == "czerwony"       # najbliższy wektorowi [1,0,0]
    assert wyniki[0][1] >= 0.85                      # wysokie podobieństwo


def test_zapis_i_wczytanie_indeksu(tmp_path=None):
    import tempfile
    katalog = Path(tempfile.mkdtemp())
    baza = BazaWektorowaFAISS(wymiar=2)
    fr = [Fragment("a", "x.pdf", 1, 0), Fragment("b", "x.pdf", 2, 1)]
    baza.zbuduj(np.array([[1, 0], [0, 1]], dtype="float32"), fr)
    baza.zapisz(katalog)

    baza2 = BazaWektorowaFAISS()
    baza2.wczytaj(katalog)
    assert len(baza2.fragmenty) == 2
    assert baza2.indeks.ntotal == 2


def test_brak_indeksu_zglasza_wyjatek():
    baza = BazaWektorowaFAISS(wymiar=2)
    try:
        baza.szukaj(np.array([1.0, 0.0], dtype="float32"), k=1)
        assert False
    except BrakIndeksu:
        pass


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
