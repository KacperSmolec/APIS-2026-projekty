"""Dzielenie tekstu na fragmenty (chunking).

Zamiast gotowego splittera znakowego zaimplementowano **własny dzielnik
zdaniowy**. Tekst najpierw rozbijany jest na zdania, a następnie zdania są
pakowane w fragmenty o docelowej długości liczonej w SŁOWACH. Dzięki temu
fragmenty nigdy nie urywają się w środku zdania, a granica jest semantycznie
sensowna. Zakładka (overlap) realizowana jest przez powtórzenie ostatnich
kilku zdań poprzedniego fragmentu na początku następnego — utrzymuje to
ciągłość kontekstu między sąsiednimi fragmentami.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from .wczytywanie import StronaDokumentu

logger = logging.getLogger("rag.dzielenie")

# Podział na zdania: kropka/wykrzyknik/pytajnik + spacja + wielka litera/cyfra.
_WZORZEC_ZDANIA = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9])")


@dataclass
class Fragment:
    """Pojedynczy fragment tekstu gotowy do wektoryzacji."""

    tekst: str
    zrodlo: str
    strona: int
    nr_fragmentu: int


def _na_zdania(tekst: str) -> List[str]:
    zdania = _WZORZEC_ZDANIA.split(tekst.replace("\n\n", " "))
    return [z.strip() for z in zdania if z.strip()]


class DzielnikTekstu:
    """Dzieli strony dokumentów na fragmenty zdaniowe ze zakładką."""

    def __init__(self, docelowa_dlugosc: int = 180, zakladka_zdan: int = 2):
        """
        :param docelowa_dlugosc: docelowa liczba słów w jednym fragmencie.
        :param zakladka_zdan:    ile ostatnich zdań poprzedniego fragmentu
                                 powtórzyć na początku kolejnego.
        """
        self.docelowa_dlugosc = docelowa_dlugosc
        self.zakladka_zdan = zakladka_zdan

    def podziel(self, strony: List[StronaDokumentu]) -> List[Fragment]:
        fragmenty: List[Fragment] = []
        licznik = 0

        for strona in strony:
            zdania = _na_zdania(strona.tekst)
            biezace: List[str] = []
            liczba_slow = 0

            for zdanie in zdania:
                biezace.append(zdanie)
                liczba_slow += len(zdanie.split())

                if liczba_slow >= self.docelowa_dlugosc:
                    fragmenty.append(self._zbuduj(biezace, strona, licznik))
                    licznik += 1
                    # Zostawiamy ostatnie N zdań jako zakładkę do następnego fragmentu.
                    biezace = biezace[-self.zakladka_zdan:] if self.zakladka_zdan else []
                    liczba_slow = sum(len(z.split()) for z in biezace)

            # Resztka zdań krótsza niż próg — też tworzy fragment, o ile nie jest śladowa.
            if biezace and len(" ".join(biezace).split()) > 20:
                fragmenty.append(self._zbuduj(biezace, strona, licznik))
                licznik += 1

        logger.info("Utworzono %d fragmentów z %d stron.", len(fragmenty), len(strony))
        return fragmenty

    @staticmethod
    def _zbuduj(zdania: List[str], strona: StronaDokumentu, nr: int) -> Fragment:
        return Fragment(
            tekst=" ".join(zdania),
            zrodlo=strona.zrodlo,
            strona=strona.strona,
            nr_fragmentu=nr,
        )
