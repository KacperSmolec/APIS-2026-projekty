"""Własna hierarchia wyjątków systemu RAG."""
from __future__ import annotations


class BladRAG(Exception):
    """Bazowy wyjątek systemu RAG."""


class BladWczytywania(BladRAG):
    """Nie udało się wczytać lub odczytać dokumentów PDF."""


class BladEmbeddingu(BladRAG):
    """Błąd podczas generowania embeddingów (np. problem z API)."""


class BrakIndeksu(BladRAG):
    """Próba zapytania zanim zbudowano/wczytano bazę wektorową."""
