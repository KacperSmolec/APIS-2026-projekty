"""Konfiguracja chatbota — wszystkie nastawy w jednym miejscu.

Wartości można nadpisać zmiennymi środowiskowymi (plik .env), dzięki czemu
ten sam kod działa lokalnie i na serwerze bez modyfikacji źródeł.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Katalog główny repozytorium (dwa poziomy w górę od tego pliku).
KATALOG_GLOWNY = Path(__file__).resolve().parents[1]


def _wczytaj_klucz() -> str:
    """Zwraca klucz API Gemini.

    Kolejność szukania:
    1. zmienna środowiskowa GEMINI_API_KEY,
    2. plik api_key.txt w katalogu nadrzędnym repozytorium (wygodne na zajęciach).
    """
    klucz = os.getenv("GEMINI_API_KEY", "").strip()
    if klucz:
        return klucz

    # Awaryjnie — wspólny plik z kluczem leżący obok folderów zadań.
    for kandydat in (
        KATALOG_GLOWNY / "api_key.txt",
        KATALOG_GLOWNY.parents[1] / "api_key.txt",
    ):
        if kandydat.exists():
            return kandydat.read_text(encoding="utf-8").strip()
    return ""


DOMYSLNY_PROMPT_SYSTEMOWY = (
    "Jesteś asystentem geoinformatycznym o imieniu GeoBot. "
    "Pomagasz studentom kierunku Geoinformatyka w zagadnieniach z zakresu GIS, "
    "teledetekcji, analizy przestrzennej oraz programowania w Pythonie. "
    "Odpowiadasz wyłącznie po polsku, rzeczowo i konkretnie. "
    "Jeśli czegoś nie wiesz lub pytanie wykracza poza Twoją wiedzę, otwarcie się "
    "do tego przyznajesz zamiast zmyślać. Gdy to pomocne, podajesz krótki przykład "
    "kodu lub wzór."
)


@dataclass
class Ustawienia:
    """Zbiór parametrów sterujących pracą chatbota."""

    # --- Model i dostęp ---
    nazwa_modelu: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    klucz_api: str = field(default_factory=_wczytaj_klucz)

    # --- Parametry generacji ---
    temperatura: float = field(default_factory=lambda: float(os.getenv("TEMPERATURA", "0.4")))
    top_p: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.9")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "40")))
    maks_tokenow_odpowiedzi: int = field(default_factory=lambda: int(os.getenv("MAKS_TOKENOW", "1024")))
    # Budżet tokenów "myślenia" modelu (Gemini 2.5). 0 = wyłączone — dla chatbota
    # konwersacyjnego nie potrzebujemy rozszerzonego rozumowania, a zyskujemy
    # przewidywalny koszt, krótszą latencję i pełną kontrolę nad max_output_tokens.
    budzet_myslenia: int = field(default_factory=lambda: int(os.getenv("BUDZET_MYSLENIA", "0")))

    # --- Kontrola kontekstu (liczona w TOKENACH, nie w wiadomościach) ---
    budzet_tokenow_kontekstu: int = field(default_factory=lambda: int(os.getenv("BUDZET_KONTEKSTU", "3000")))

    # --- Sieć / odporność ---
    liczba_prob: int = field(default_factory=lambda: int(os.getenv("LICZBA_PROB", "3")))
    opoznienie_ponowienia_s: float = field(default_factory=lambda: float(os.getenv("OPOZNIENIE_S", "1.5")))

    # --- Prompt systemowy ---
    prompt_systemowy: str = DOMYSLNY_PROMPT_SYSTEMOWY

    # --- Logowanie ---
    katalog_logow: Path = field(default_factory=lambda: KATALOG_GLOWNY / "logi")

    def __post_init__(self) -> None:
        if not self.klucz_api:
            raise RuntimeError(
                "Brak klucza API Gemini. Ustaw zmienną GEMINI_API_KEY "
                "albo umieść plik api_key.txt w katalogu repozytorium."
            )
