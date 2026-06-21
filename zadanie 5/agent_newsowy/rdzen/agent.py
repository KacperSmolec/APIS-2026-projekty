"""Agent newsowy — własna pętla wywoływania narzędzi (function calling).

Zamiast korzystać z gotowego frameworka agentowego, implementujemy pętlę ręcznie
na natywnym function-calling modelu Gemini. Pętla działa tak:

  1. wysyłamy do modelu wiadomość + deklaracje narzędzi,
  2. model zwraca albo finalny tekst, albo żądanie wywołania narzędzia,
  3. jeśli to wywołanie — uruchamiamy odpowiednią funkcję i ZWRACAMY wynik do
     modelu jako "obserwację",
  4. powtarzamy, aż model uzna cel za osiągnięty.

Dzięki ręcznej pętli mamy pełną kontrolę i możemy logować każdą decyzję agenta
(jakie narzędzie wybrał, z jakimi argumentami) oraz obsłużyć błędy narzędzi,
przekazując je modelowi do samodzielnej korekty.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from google import genai
from google.genai import types

from .konfiguracja import Ustawienia
from .llm import generuj, myslenie
from .narzedzia import NARZEDZIA, ustaw_kontekst
from .prompty import PROMPT_SYSTEMOWY

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("agent.petla")


@dataclass
class KrokAgenta:
    """Pojedynczy krok śladu działania agenta."""

    narzedzie: str
    argumenty: dict
    obserwacja: str


@dataclass
class WynikAgenta:
    """Końcowy wynik pracy agenta wraz ze śladem decyzji."""

    odpowiedz: str
    slad: List[KrokAgenta] = field(default_factory=list)


class AgentNewsowy:
    """Autonomiczny agent tworzący raport prasowy na zadany temat."""

    def __init__(self, ustawienia: Optional[Ustawienia] = None, klient: Optional[genai.Client] = None):
        self.ust = ustawienia or Ustawienia()
        self.klient = klient or genai.Client(api_key=self.ust.klucz_gemini)
        # Wstrzykujemy kontekst do narzędzi (klient + ustawienia).
        ustaw_kontekst(self.klient, self.ust)
        # Mapa nazwa -> funkcja, do ręcznego dyspozytora wywołań.
        self._dyspozytor = {f.__name__: f for f in NARZEDZIA}

    def _konfiguracja(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=PROMPT_SYSTEMOWY,
            tools=NARZEDZIA,
            temperature=self.ust.temperatura,
            # Wyłączamy automatyczne wywoływanie — chcemy sami sterować pętlą.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=myslenie(self.ust.model_llm),
        )

    def uruchom(self, temat: str) -> WynikAgenta:
        """Uruchamia agenta dla zadanego tematu i zwraca wynik ze śladem decyzji."""
        polecenie = (
            f"Przygotuj raport PDF podsumowujący najważniejsze ostatnie wydarzenia "
            f"na temat: {temat}."
        )
        contents: List[Any] = [types.Content(role="user", parts=[types.Part(text=polecenie)])]
        konfiguracja = self._konfiguracja()
        slad: List[KrokAgenta] = []

        for iteracja in range(1, self.ust.maks_iteracji_agenta + 1):
            odp = generuj(
                self.klient, model=self.ust.model_llm, contents=contents, config=konfiguracja
            )
            kandydat = odp.candidates[0]
            contents.append(kandydat.content)   # dopisujemy odpowiedź modelu do historii

            wywolania = odp.function_calls or []
            if not wywolania:
                # Brak wywołań narzędzi = agent uznał cel za osiągnięty.
                logger.info("Agent zakończył po %d iteracjach.", iteracja)
                return WynikAgenta(odpowiedz=(odp.text or "").strip(), slad=slad)

            # Wykonujemy wszystkie zażądane wywołania i zwracamy obserwacje.
            czesci_odpowiedzi = []
            for wc in wywolania:
                wynik = self._wywolaj_narzedzie(wc.name, dict(wc.args or {}))
                slad.append(KrokAgenta(narzedzie=wc.name, argumenty=dict(wc.args or {}), obserwacja=wynik))
                czesci_odpowiedzi.append(
                    types.Part.from_function_response(name=wc.name, response={"wynik": wynik})
                )
            contents.append(types.Content(role="user", parts=czesci_odpowiedzi))

        logger.warning("Przekroczono limit iteracji (%d).", self.ust.maks_iteracji_agenta)
        return WynikAgenta(
            odpowiedz="Przekroczono limit kroków agenta przed ukończeniem zadania.", slad=slad
        )

    def _wywolaj_narzedzie(self, nazwa: str, argumenty: dict) -> str:
        """Uruchamia narzędzie po nazwie; błąd zamienia na obserwację dla modelu."""
        logger.info("Agent wybrał narzędzie: %s | argumenty: %s", nazwa, list(argumenty))
        funkcja = self._dyspozytor.get(nazwa)
        if funkcja is None:
            return f"Błąd: nieznane narzędzie '{nazwa}'."
        try:
            wynik = funkcja(**argumenty)
            return str(wynik)
        except Exception as e:
            # Zamiast wywracać program, oddajemy błąd modelowi — niech sam skoryguje.
            logger.warning("Narzędzie %s zgłosiło błąd: %s", nazwa, e)
            return f"Błąd narzędzia {nazwa}: {e}"
