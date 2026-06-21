"""Konfiguracja systemu RAG.

Wszystkie nastawy w jednym miejscu; można je nadpisać zmiennymi środowiskowymi
(plik .env), żeby ten sam kod działał bez zmian na różnych maszynach.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

KATALOG_GLOWNY = Path(__file__).resolve().parents[1]


def _wczytaj_klucz() -> str:
    """Zwraca klucz API Gemini ze zmiennej środowiskowej lub pliku api_key.txt."""
    klucz = os.getenv("GEMINI_API_KEY", "").strip()
    if klucz:
        return klucz
    for kandydat in (KATALOG_GLOWNY / "api_key.txt", KATALOG_GLOWNY.parents[1] / "api_key.txt"):
        if kandydat.exists():
            return kandydat.read_text(encoding="utf-8").strip()
    return ""


# Prompt systemowy generatora — wymusza odpowiadanie WYŁĄCZNIE z kontekstu.
PROMPT_SYSTEMOWY = (
    "Jesteś ekspertem-asystentem w dziedzinie teledetekcji i klasyfikacji pokrycia "
    "terenu. Odpowiadasz na pytania WYŁĄCZNIE na podstawie fragmentów publikacji "
    "naukowych podanych w sekcji KONTEKST. Zasady:\n"
    "1. Jeśli w kontekście nie ma odpowiedzi, napisz dokładnie: "
    "'Na podstawie dostarczonych publikacji nie potrafię odpowiedzieć na to pytanie.' "
    "Nie korzystaj z wiedzy spoza kontekstu i niczego nie zmyślaj.\n"
    "2. Odpowiadaj rzeczowo, po polsku, w 2-5 zdaniach.\n"
    "3. Na końcu odpowiedzi w nawiasie kwadratowym podaj numery wykorzystanych "
    "fragmentów, np. [1, 3]."
)


@dataclass
class Ustawienia:
    """Parametry sterujące pracą systemu RAG."""

    # --- Dostęp do modeli ---
    klucz_api: str = field(default_factory=_wczytaj_klucz)
    model_llm: str = field(default_factory=lambda: os.getenv("MODEL_LLM", "gemini-2.5-flash"))
    model_embeddingow: str = field(default_factory=lambda: os.getenv("MODEL_EMB", "gemini-embedding-001"))
    wymiar_embeddingu: int = field(default_factory=lambda: int(os.getenv("WYMIAR_EMB", "768")))

    # --- Chunking (dzielenie na fragmenty) — w SŁOWACH ---
    docelowa_dlugosc_fragmentu: int = field(default_factory=lambda: int(os.getenv("DLUGOSC_FRAGMENTU", "180")))
    zakladka_zdan: int = field(default_factory=lambda: int(os.getenv("ZAKLADKA_ZDAN", "2")))

    # --- Retrieval ---
    liczba_fragmentow: int = field(default_factory=lambda: int(os.getenv("LICZBA_FRAGMENTOW", "4")))

    # --- Generacja ---
    temperatura: float = field(default_factory=lambda: float(os.getenv("TEMPERATURA", "0.2")))
    maks_tokenow_odpowiedzi: int = field(default_factory=lambda: int(os.getenv("MAKS_TOKENOW", "800")))

    # --- Ścieżki ---
    katalog_danych: Path = field(default_factory=lambda: KATALOG_GLOWNY / "dane")
    katalog_indeksu: Path = field(default_factory=lambda: KATALOG_GLOWNY / "indeks")

    prompt_systemowy: str = PROMPT_SYSTEMOWY

    def __post_init__(self) -> None:
        if not self.klucz_api:
            raise RuntimeError(
                "Brak klucza API Gemini. Ustaw GEMINI_API_KEY lub dodaj plik api_key.txt."
            )
