"""Główny silnik RAG — spina wszystkie komponenty w jeden system.

Przepływ:
  PDF -> WczytywaczPDF -> DzielnikTekstu -> ModelEmbeddingow -> BazaWektorowaFAISS
a przy zapytaniu:
  pytanie -> embedding -> wyszukiwanie w FAISS -> budowa promptu z kontekstem -> Gemini
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from google import genai
from google.genai import types

from .baza_wektorowa import BazaWektorowaFAISS
from .dzielenie import DzielnikTekstu, Fragment
from .embeddingi import ModelEmbeddingow
from .konfiguracja import Ustawienia
from .wczytywanie import WczytywaczPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rag.silnik")


@dataclass
class Odpowiedz:
    """Wynik zapytania do systemu RAG."""

    pytanie: str
    odpowiedz: str
    zrodla: List[Tuple[Fragment, float]]   # (fragment, podobieństwo)


class SilnikRAG:
    """System RAG: budowanie bazy z PDF i odpowiadanie na pytania z kontekstu."""

    def __init__(self, ustawienia: Optional[Ustawienia] = None, klient: Optional[genai.Client] = None):
        self.ust = ustawienia or Ustawienia()
        self.klient = klient or genai.Client(api_key=self.ust.klucz_api)

        self.embeddingi = ModelEmbeddingow(
            klient=self.klient,
            nazwa_modelu=self.ust.model_embeddingow,
            wymiar=self.ust.wymiar_embeddingu,
        )
        self.dzielnik = DzielnikTekstu(
            docelowa_dlugosc=self.ust.docelowa_dlugosc_fragmentu,
            zakladka_zdan=self.ust.zakladka_zdan,
        )
        self.baza = BazaWektorowaFAISS(wymiar=self.ust.wymiar_embeddingu)

    # ----------------------------- budowanie ----------------------------- #
    def zbuduj_indeks(self) -> int:
        """Buduje bazę wektorową z PDF-ów w katalogu danych i zapisuje na dysk.

        :return: liczba zaindeksowanych fragmentów.
        """
        logger.info("== Budowanie indeksu z katalogu: %s ==", self.ust.katalog_danych)
        strony = WczytywaczPDF(self.ust.katalog_danych).wczytaj()
        fragmenty = self.dzielnik.podziel(strony)

        wektory = self.embeddingi.osadz_dokumenty([fr.tekst for fr in fragmenty])
        self.baza.zbuduj(wektory, fragmenty)
        self.baza.zapisz(self.ust.katalog_indeksu)
        return len(fragmenty)

    def wczytaj_indeks(self) -> None:
        """Wczytuje wcześniej zbudowaną bazę wektorową z dysku."""
        self.baza.wczytaj(self.ust.katalog_indeksu)

    # ------------------------------ zapytanie ----------------------------- #
    def zapytaj(self, pytanie: str) -> Odpowiedz:
        """Zadaje pytanie do systemu: retrieval + generacja odpowiedzi z kontekstu."""
        wektor = self.embeddingi.osadz_zapytanie(pytanie)
        trafienia = self.baza.szukaj(wektor, k=self.ust.liczba_fragmentow)

        kontekst = self._zbuduj_kontekst(trafienia)
        odpowiedz = self._generuj(pytanie, kontekst)
        return Odpowiedz(pytanie=pytanie, odpowiedz=odpowiedz, zrodla=trafienia)

    @staticmethod
    def _zbuduj_kontekst(trafienia: List[Tuple[Fragment, float]]) -> str:
        """Skleja znalezione fragmenty w ponumerowany kontekst dla modelu."""
        bloki = []
        for i, (fr, _sim) in enumerate(trafienia, start=1):
            bloki.append(f"[{i}] (źródło: {fr.zrodlo}, str. {fr.strona})\n{fr.tekst}")
        return "\n\n".join(bloki)

    def _generuj(self, pytanie: str, kontekst: str) -> str:
        tresc = f"KONTEKST:\n{kontekst}\n\nPYTANIE: {pytanie}"
        konfiguracja = types.GenerateContentConfig(
            system_instruction=self.ust.prompt_systemowy,
            temperature=self.ust.temperatura,
            max_output_tokens=self.ust.maks_tokenow_odpowiedzi,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        odp = self.klient.models.generate_content(
            model=self.ust.model_llm, contents=tresc, config=konfiguracja
        )
        return (odp.text or "").strip()
