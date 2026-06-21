"""Baza wektorowa oparta o FAISS.

FAISS to biblioteka do szybkiego wyszukiwania najbliższych sąsiadów w przestrzeni
wektorowej. Używamy indeksu ``IndexFlatIP`` (iloczyn skalarny). Ponieważ wektory
są wcześniej znormalizowane do długości 1, iloczyn skalarny równa się
**podobieństwu kosinusowemu** — im wynik bliższy 1, tym fragment bardziej pasuje
do zapytania.

FAISS sam nie przechowuje treści ani metadanych, dlatego trzymamy je równolegle
w liście Pythona, a przy zapisie serializujemy do pliku JSON. Indeks i metadane
zapisujemy obok siebie, co pozwala odtworzyć bazę bez ponownej wektoryzacji.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .dzielenie import Fragment
from .wyjatki import BrakIndeksu

logger = logging.getLogger("rag.baza")

NAZWA_INDEKSU = "indeks.faiss"
NAZWA_METADANYCH = "fragmenty.json"


class BazaWektorowaFAISS:
    """Buduje, zapisuje, wczytuje i przeszukuje indeks FAISS."""

    def __init__(self, wymiar: int = 768):
        self.wymiar = wymiar
        self.indeks: faiss.Index | None = None
        self.fragmenty: List[Fragment] = []

    # --------------------------- budowa / zapis --------------------------- #
    def zbuduj(self, embeddingi: np.ndarray, fragmenty: List[Fragment]) -> None:
        if embeddingi.shape[0] != len(fragmenty):
            raise ValueError("Liczba embeddingów nie zgadza się z liczbą fragmentów.")
        self.indeks = faiss.IndexFlatIP(self.wymiar)   # IP + wektory znormalizowane = kosinus
        self.indeks.add(embeddingi)
        self.fragmenty = fragmenty
        logger.info("Zbudowano indeks FAISS: %d wektorów, wymiar %d.", self.indeks.ntotal, self.wymiar)

    def zapisz(self, katalog: Path) -> None:
        katalog = Path(katalog)
        katalog.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.indeks, str(katalog / NAZWA_INDEKSU))
        with open(katalog / NAZWA_METADANYCH, "w", encoding="utf-8") as f:
            json.dump([asdict(fr) for fr in self.fragmenty], f, ensure_ascii=False)
        logger.info("Zapisano bazę wektorową do %s.", katalog)

    # ------------------------------ wczytanie ----------------------------- #
    def wczytaj(self, katalog: Path) -> None:
        katalog = Path(katalog)
        plik_indeksu = katalog / NAZWA_INDEKSU
        plik_meta = katalog / NAZWA_METADANYCH
        if not plik_indeksu.exists() or not plik_meta.exists():
            raise BrakIndeksu(f"Brak zapisanego indeksu w {katalog}. Najpierw zbuduj bazę.")
        self.indeks = faiss.read_index(str(plik_indeksu))
        with open(plik_meta, encoding="utf-8") as f:
            self.fragmenty = [Fragment(**d) for d in json.load(f)]
        self.wymiar = self.indeks.d
        logger.info("Wczytano bazę: %d wektorów.", self.indeks.ntotal)

    # ----------------------------- wyszukiwanie --------------------------- #
    def szukaj(self, wektor_zapytania: np.ndarray, k: int = 4) -> List[Tuple[Fragment, float]]:
        """Zwraca k najbardziej podobnych fragmentów wraz z wynikiem podobieństwa."""
        if self.indeks is None:
            raise BrakIndeksu("Indeks nie został zbudowany ani wczytany.")
        wektor = wektor_zapytania.reshape(1, -1).astype("float32")
        podobienstwa, indeksy = self.indeks.search(wektor, k)
        wyniki: List[Tuple[Fragment, float]] = []
        for idx, sim in zip(indeksy[0], podobienstwa[0]):
            if idx == -1:
                continue
            wyniki.append((self.fragmenty[idx], float(sim)))
        return wyniki
