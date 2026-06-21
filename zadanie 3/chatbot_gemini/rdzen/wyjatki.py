"""Własna hierarchia wyjątków chatbota.

Dzięki osobnym typom wyjątków warstwa wywołująca (CLI, notebook, ewentualne
API) może różnie reagować na różne klasy błędów — np. ponowić zapytanie przy
błędzie sieci, ale od razu poinformować użytkownika przy przekroczeniu limitu
kontekstu.
"""
from __future__ import annotations


class BladChatbota(Exception):
    """Bazowy wyjątek — wszystkie pozostałe po nim dziedziczą."""


class BladPolaczenia(BladChatbota):
    """Problem z komunikacją sieciową z API modelu (timeout, brak sieci, 5xx)."""


class BladOdpowiedziModelu(BladChatbota):
    """Model nie zwrócił poprawnej odpowiedzi.

    Dotyczy np. odpowiedzi zablokowanej przez filtry bezpieczeństwa, pustej
    treści albo niespodziewanego formatu zwracanych danych.
    """


class LimitKontekstuPrzekroczony(BladChatbota):
    """Pojedyncza wiadomość nie mieści się w budżecie tokenów kontekstu."""
