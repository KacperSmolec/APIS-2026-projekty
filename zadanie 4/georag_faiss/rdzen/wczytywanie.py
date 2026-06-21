"""Wczytywanie dokumentów PDF.

Korzystamy bezpośrednio z biblioteki ``pypdf`` (bez nakładek frameworkowych),
dzięki czemu mamy pełną kontrolę nad ekstrakcją i czyszczeniem tekstu. Każda
strona staje się osobnym rekordem z metadanymi (plik źródłowy, numer strony),
co później pozwala wskazać dokładne źródło odpowiedzi.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

from .wyjatki import BladWczytywania

logger = logging.getLogger("rag.wczytywanie")


@dataclass
class StronaDokumentu:
    """Tekst pojedynczej strony PDF wraz z metadanymi."""

    tekst: str
    zrodlo: str   # nazwa pliku
    strona: int   # numer strony (od 1)


def _wyczysc_tekst(surowy: str) -> str:
    """Porządkuje tekst wyekstrahowany z PDF.

    PDF-y naukowe często łamią słowa myślnikiem na końcu wiersza oraz wstawiają
    przypadkowe znaki nowej linii w środku zdań. Sklejamy podział wyrazów i
    normalizujemy białe znaki do pojedynczych spacji.
    """
    # Sklejanie wyrazów przeniesionych myślnikiem na końcu wiersza.
    surowy = re.sub(r"-\s*\n\s*", "", surowy)
    # Wszystkie znaki nowej linii traktujemy jak spacje (zdania bywają łamane w PDF).
    surowy = re.sub(r"\s*\n\s*", " ", surowy)
    # Redukcja wielokrotnych białych znaków do pojedynczej spacji.
    surowy = re.sub(r"\s{2,}", " ", surowy)
    return surowy.strip()


class WczytywaczPDF:
    """Wczytuje pliki PDF z katalogu i zwraca listę stron z tekstem."""

    def __init__(self, katalog: Path):
        self.katalog = Path(katalog)

    def wczytaj(self) -> List[StronaDokumentu]:
        if not self.katalog.exists():
            raise BladWczytywania(f"Katalog z danymi nie istnieje: {self.katalog}")

        pliki = sorted(self.katalog.glob("*.pdf"))
        if not pliki:
            raise BladWczytywania(f"Brak plików PDF w katalogu: {self.katalog}")

        strony: List[StronaDokumentu] = []
        for plik in pliki:
            try:
                reader = PdfReader(str(plik))
            except Exception as e:  # uszkodzony/zaszyfrowany plik
                logger.warning("Pomijam plik %s (nie da się otworzyć): %s", plik.name, e)
                continue

            liczba_dodanych = 0
            for nr, strona in enumerate(reader.pages, start=1):
                tekst = _wyczysc_tekst(strona.extract_text() or "")
                if len(tekst) < 50:   # pomijamy puste strony / same nagłówki
                    continue
                strony.append(StronaDokumentu(tekst=tekst, zrodlo=plik.name, strona=nr))
                liczba_dodanych += 1
            logger.info("Wczytano %s: %d stron z tekstem", plik.name, liczba_dodanych)

        if not strony:
            raise BladWczytywania("Nie udało się wyekstrahować tekstu z żadnego PDF.")

        logger.info("Łącznie wczytano %d stron z %d plików.", len(strony), len(pliki))
        return strony
