"""Konfiguracja agenta newsowego."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

KATALOG_GLOWNY = Path(__file__).resolve().parents[1]


def _wczytaj_klucz_gemini() -> str:
    klucz = os.getenv("GEMINI_API_KEY", "").strip()
    if klucz:
        return klucz
    for kandydat in (KATALOG_GLOWNY / "api_key.txt", KATALOG_GLOWNY.parents[1] / "api_key.txt"):
        if kandydat.exists():
            return kandydat.read_text(encoding="utf-8").strip()
    return ""


def _wczytaj_klucz_newsapi() -> str:
    """Klucz NewsAPI ze zmiennej środowiskowej lub pliku newsapi_key.txt.

    Brak klucza nie jest błędem — narzędzie wyszukiwania ma wtedy fallback
    na darmowy kanał RSS (Google News).
    """
    klucz = os.getenv("NEWSAPI_KEY", "").strip()
    if klucz:
        return klucz
    plik = KATALOG_GLOWNY / "newsapi_key.txt"
    if plik.exists():
        return plik.read_text(encoding="utf-8").strip()
    return ""


@dataclass
class Ustawienia:
    """Parametry agenta."""

    klucz_gemini: str = field(default_factory=_wczytaj_klucz_gemini)
    klucz_newsapi: str = field(default_factory=_wczytaj_klucz_newsapi)

    model_llm: str = field(default_factory=lambda: os.getenv("MODEL_LLM", "gemini-2.5-flash"))
    temperatura: float = field(default_factory=lambda: float(os.getenv("TEMPERATURA", "0.3")))

    liczba_artykulow: int = field(default_factory=lambda: int(os.getenv("LICZBA_ARTYKULOW", "5")))
    maks_iteracji_agenta: int = field(default_factory=lambda: int(os.getenv("MAKS_ITERACJI", "8")))

    katalog_raportow: Path = field(default_factory=lambda: KATALOG_GLOWNY / "raporty")

    def __post_init__(self) -> None:
        if not self.klucz_gemini:
            raise RuntimeError("Brak klucza API Gemini. Ustaw GEMINI_API_KEY lub dodaj api_key.txt.")
