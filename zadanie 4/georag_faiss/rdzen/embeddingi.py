"""Generowanie embeddingów przy użyciu modelu Gemini Embedding.

Model ``gemini-embedding-001`` udostępnia mechanizm ``task_type``, który pozwala
**asymetrycznie** kodować dokumenty i zapytania:
* ``RETRIEVAL_DOCUMENT`` — dla fragmentów zapisywanych do bazy,
* ``RETRIEVAL_QUERY``    — dla pytania użytkownika.

Model jest świadomy tej różnicy (dokument vs zapytanie), co poprawia trafność
wyszukiwania w porównaniu z kodowaniem obu tym samym trybem. Wektory dodatkowo
normalizujemy do długości 1, dzięki czemu iloczyn skalarny w FAISS jest wprost
podobieństwem kosinusowym.
"""
from __future__ import annotations

import logging
import re
import time
from typing import List

import numpy as np
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from .wyjatki import BladEmbeddingu

logger = logging.getLogger("rag.embeddingi")


class ModelEmbeddingow:
    """Nakładka na API embeddingów Gemini z normalizacją, wsadowaniem,
    ograniczaniem tempa (rate limiting) i ponawianiem po przekroczeniu limitu.

    Darmowy plan liczy każdy osadzany fragment jako osobne zapytanie
    (limit ~100/min), dlatego pilnujemy okna minutowego i w razie błędu 429
    czekamy zalecany czas i ponawiamy paczkę.
    """

    def __init__(self, klient: genai.Client, nazwa_modelu: str, wymiar: int = 768,
                 rozmiar_wsadu: int = 50, limit_na_minute: int = 95):
        self.klient = klient
        self.nazwa_modelu = nazwa_modelu
        self.wymiar = wymiar
        self.rozmiar_wsadu = rozmiar_wsadu
        self.limit_na_minute = limit_na_minute

    def _osadz(self, teksty: List[str], typ_zadania: str) -> np.ndarray:
        wektory: List[List[float]] = []
        okno_start = time.monotonic()
        w_oknie = 0   # ile fragmentów osadzono w bieżącym oknie minutowym

        for i in range(0, len(teksty), self.rozmiar_wsadu):
            paczka = teksty[i:i + self.rozmiar_wsadu]

            # Rate limiting: jeśli paczka przekroczyłaby limit w bieżącym oknie,
            # poczekaj do końca minuty i otwórz nowe okno.
            if w_oknie + len(paczka) > self.limit_na_minute:
                pozostalo = 60 - (time.monotonic() - okno_start)
                if pozostalo > 0:
                    logger.info("Limit tempa — czekam %.0f s przed kolejną paczką...", pozostalo)
                    time.sleep(pozostalo)
                okno_start = time.monotonic()
                w_oknie = 0

            odp = self._wyslij_paczke(paczka, typ_zadania)
            wektory.extend(e.values for e in odp.embeddings)
            w_oknie += len(paczka)
            if len(teksty) > self.rozmiar_wsadu:
                logger.info("Osadzono %d/%d fragmentów...", min(i + self.rozmiar_wsadu, len(teksty)), len(teksty))

        macierz = np.asarray(wektory, dtype="float32")
        return self._normalizuj(macierz)

    def _wyslij_paczke(self, paczka: List[str], typ_zadania: str, maks_prob: int = 4):
        """Wysyła jedną paczkę; po błędzie 429 czeka zalecany czas i ponawia."""
        for proba in range(1, maks_prob + 1):
            try:
                return self.klient.models.embed_content(
                    model=self.nazwa_modelu,
                    contents=paczka,
                    config=types.EmbedContentConfig(
                        task_type=typ_zadania,
                        output_dimensionality=self.wymiar,
                    ),
                )
            except genai_errors.APIError as e:
                if getattr(e, "code", None) == 429 and proba < maks_prob:
                    czekaj = self._czas_oczekiwania(str(e))
                    logger.info("Przekroczono limit (429). Czekam %d s i ponawiam (proba %d).", czekaj, proba)
                    time.sleep(czekaj)
                    continue
                raise BladEmbeddingu(f"Błąd generowania embeddingów: {e}") from e
            except Exception as e:
                raise BladEmbeddingu(f"Błąd generowania embeddingów: {e}") from e
        raise BladEmbeddingu("Nie udało się osadzić paczki po wielu próbach.")

    @staticmethod
    def _czas_oczekiwania(komunikat: str, domyslny: int = 46) -> int:
        m = re.search(r"retry in (\d+)", komunikat) or re.search(r"retryDelay.*?(\d+)s", komunikat)
        return int(m.group(1)) + 2 if m else domyslny

    @staticmethod
    def _normalizuj(macierz: np.ndarray) -> np.ndarray:
        normy = np.linalg.norm(macierz, axis=1, keepdims=True)
        normy[normy == 0] = 1.0
        return macierz / normy

    def osadz_dokumenty(self, teksty: List[str]) -> np.ndarray:
        """Embeddingi fragmentów dokumentów (tryb RETRIEVAL_DOCUMENT)."""
        return self._osadz(teksty, "RETRIEVAL_DOCUMENT")

    def osadz_zapytanie(self, pytanie: str) -> np.ndarray:
        """Embedding pojedynczego zapytania (tryb RETRIEVAL_QUERY)."""
        return self._osadz([pytanie], "RETRIEVAL_QUERY")[0]
