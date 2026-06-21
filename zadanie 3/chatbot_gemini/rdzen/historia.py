"""Zarządzanie historią rozmowy oraz kontrola długości kontekstu.

W odróżnieniu od popularnego podejścia "przesuwnego okna" liczonego w liczbie
wiadomości, tutaj kontekst kontrolujemy w **tokenach**. Jest to dokładniejsze,
bo jedna wiadomość może mieć 5 tokenów, a inna 500 — to właśnie tokeny, a nie
liczba wpisów, decydują o tym, czy zmieścimy się w oknie modelu i ile zapłacimy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .wyjatki import LimitKontekstuPrzekroczony


@dataclass
class Wiadomosc:
    """Pojedynczy wpis w historii rozmowy."""

    rola: str  # "user" albo "model" (konwencja Gemini)
    tresc: str


@dataclass
class HistoriaRozmowy:
    """Przechowuje wiadomości i pilnuje budżetu tokenów kontekstu.

    :param prompt_systemowy: instrukcja, która NIGDY nie jest usuwana z kontekstu.
    :param budzet_tokenow:   maksymalna liczba tokenów (prompt systemowy + dialog).
    :param licznik_tokenow:  funkcja licząca tokeny dla podanego tekstu.
                             Wstrzykiwana z zewnątrz, dzięki czemu w testach można
                             podać prostą atrapę bez wywołań sieciowych.
    """

    prompt_systemowy: str
    budzet_tokenow: int
    licznik_tokenow: Callable[[str], int]
    wiadomosci: list[Wiadomosc] = field(default_factory=list)

    def dodaj(self, rola: str, tresc: str) -> None:
        self.wiadomosci.append(Wiadomosc(rola=rola, tresc=tresc))

    def cofnij_ostatnia(self) -> None:
        """Usuwa ostatnią wiadomość — używane przy wycofywaniu po błędzie."""
        if self.wiadomosci:
            self.wiadomosci.pop()

    def liczba_tokenow(self) -> int:
        """Szacuje łączną liczbę tokenów: prompt systemowy + cały dialog."""
        razem = self.licznik_tokenow(self.prompt_systemowy)
        for w in self.wiadomosci:
            razem += self.licznik_tokenow(w.tresc)
        return razem

    def przytnij_do_budzetu(self) -> int:
        """Usuwa najstarsze wiadomości dialogu aż kontekst zmieści się w budżecie.

        Prompt systemowy jest chroniony — zostaje zawsze. Jeśli po usunięciu
        całego dialogu poza ostatnią wiadomością nadal przekraczamy budżet,
        oznacza to, że pojedyncza wiadomość jest za długa -> zgłaszamy wyjątek.

        :return: liczba usuniętych (zapomnianych) wiadomości.
        """
        usuniete = 0
        while self.liczba_tokenow() > self.budzet_tokenow and len(self.wiadomosci) > 1:
            self.wiadomosci.pop(0)
            usuniete += 1

        if self.liczba_tokenow() > self.budzet_tokenow:
            raise LimitKontekstuPrzekroczony(
                "Pojedyncza wiadomość przekracza budżet tokenów kontekstu "
                f"({self.budzet_tokenow}). Skróć zapytanie."
            )
        return usuniete

    def do_formatu_gemini(self) -> list[dict]:
        """Zwraca dialog w strukturze oczekiwanej przez API Gemini.

        Prompt systemowy przekazywany jest osobno (system_instruction), więc
        tutaj zwracamy wyłącznie wymianę user/model.
        """
        return [
            {"role": w.rola, "parts": [{"text": w.tresc}]}
            for w in self.wiadomosci
        ]
