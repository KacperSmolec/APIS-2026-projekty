"""Wspólny pomocnik wywołań modelu z ponawianiem po przekroczeniu limitu (429).

Agent wykonuje wiele zapytań do modelu (pętla decyzyjna + narzędzia korzystające
z LLM), a darmowy plan Gemini ogranicza liczbę zapytań na minutę. Ten helper
przechwytuje błąd 429, odczekuje zalecany czas i ponawia — dzięki czemu cała
sesja agenta kończy się powodzeniem mimo limitów.
"""
from __future__ import annotations

import logging
import re
import time

from google.genai import errors as genai_errors
from google.genai import types

logger = logging.getLogger("agent.llm")


def myslenie(model: str):
    """Zwraca konfigurację 'myślenia' (wyłączonego) dla modeli z rodziny 2.5.

    Wyłączamy myślenie dla przewidywalnego kosztu i pełnej kontroli nad długością
    odpowiedzi. Modele spoza rodziny 2.5 nie wspierają ThinkingConfig — None.
    """
    return types.ThinkingConfig(thinking_budget=0) if "2.5" in model else None


def _czas_oczekiwania(komunikat: str, domyslny: int = 30) -> int:
    m = re.search(r"retry in (\d+)", komunikat) or re.search(r"retryDelay.*?(\d+)s", komunikat)
    return int(m.group(1)) + 2 if m else domyslny


def generuj(klient, *, model, contents, config, maks_prob: int = 5):
    """Wywołuje generate_content; po błędzie 429 czeka i ponawia."""
    for proba in range(1, maks_prob + 1):
        try:
            return klient.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 429 and proba < maks_prob:
                czekaj = _czas_oczekiwania(str(e))
                logger.info("Limit zapytań (429) — czekam %d s i ponawiam (próba %d/%d).", czekaj, proba, maks_prob)
                time.sleep(czekaj)
                continue
            raise
    raise RuntimeError("Nie udało się uzyskać odpowiedzi modelu po wielu próbach.")
