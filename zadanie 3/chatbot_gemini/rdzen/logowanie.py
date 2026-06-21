"""Konfiguracja logowania.

Świadomie rozdzielamy dwa strumienie logów:
* ``zapytania`` — historia zapytań użytkownika i metadanych generacji
  (czas odpowiedzi, liczba tokenów). Przydatne do analityki i rozliczeń.
* ``bledy``     — wyłącznie błędy i ostrzeżenia. Krótki plik, który łatwo
  przeglądać podczas diagnozowania awarii.

Każdy logger pisze do osobnego pliku, a błędy dodatkowo trafiają na konsolę.
"""
from __future__ import annotations

import logging
from pathlib import Path

_skonfigurowane: set[str] = set()


def _utworz_logger(nazwa: str, plik: Path, na_konsole: bool) -> logging.Logger:
    logger = logging.getLogger(nazwa)
    if nazwa in _skonfigurowane:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    format_ = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    plik.parent.mkdir(parents=True, exist_ok=True)
    uchwyt_pliku = logging.FileHandler(plik, encoding="utf-8")
    uchwyt_pliku.setFormatter(format_)
    logger.addHandler(uchwyt_pliku)

    if na_konsole:
        uchwyt_konsoli = logging.StreamHandler()
        uchwyt_konsoli.setFormatter(format_)
        uchwyt_konsoli.setLevel(logging.WARNING)
        logger.addHandler(uchwyt_konsoli)

    _skonfigurowane.add(nazwa)
    return logger


def loggery(katalog_logow: Path) -> tuple[logging.Logger, logging.Logger]:
    """Zwraca parę (logger_zapytan, logger_bledow)."""
    log_zapytan = _utworz_logger("chatbot.zapytania", katalog_logow / "zapytania.log", na_konsole=False)
    log_bledow = _utworz_logger("chatbot.bledy", katalog_logow / "bledy.log", na_konsole=True)
    return log_zapytan, log_bledow
