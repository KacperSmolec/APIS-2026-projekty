"""Główna klasa chatbota oparta o model Gemini.

Łączy w sobie: integrację z modelem, prompt systemowy, historię rozmowy,
kontrolę kontekstu (w tokenach), parametry generacji, obsługę błędów oraz
logowanie zapytań i błędów.
"""
from __future__ import annotations

import time
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from .konfiguracja import Ustawienia
from .historia import HistoriaRozmowy
from .logowanie import loggery
from .wyjatki import (
    BladOdpowiedziModelu,
    BladPolaczenia,
    LimitKontekstuPrzekroczony,
)


class ChatbotGemini:
    """Production-ready chatbot na modelu Gemini.

    Przykład:
        >>> bot = ChatbotGemini()
        >>> bot.odpowiedz("Czym jest NDVI?")
    """

    def __init__(self, ustawienia: Optional[Ustawienia] = None, klient: Optional[genai.Client] = None):
        self.ust = ustawienia or Ustawienia()
        # Klient można wstrzyknąć (np. atrapę w testach); domyślnie tworzymy realny.
        self.klient = klient or genai.Client(api_key=self.ust.klucz_api)

        self.log_zapytan, self.log_bledow = loggery(self.ust.katalog_logow)

        self.historia = HistoriaRozmowy(
            prompt_systemowy=self.ust.prompt_systemowy,
            budzet_tokenow=self.ust.budzet_tokenow_kontekstu,
            licznik_tokenow=self._policz_tokeny,
        )
        self.log_zapytan.info(
            "Inicjalizacja chatbota | model=%s | budzet_kontekstu=%d tokenow",
            self.ust.nazwa_modelu,
            self.ust.budzet_tokenow_kontekstu,
        )

    # ------------------------------------------------------------------ #
    # Liczenie tokenów                                                    #
    # ------------------------------------------------------------------ #
    def _policz_tokeny(self, tekst: str) -> int:
        """Liczy tokeny tekstu przez API modelu.

        Jeśli wywołanie sieciowe się nie powiedzie, używamy przybliżenia
        (~4 znaki na token), żeby kontrola kontekstu działała także offline.
        """
        if not tekst:
            return 0
        try:
            wynik = self.klient.models.count_tokens(model=self.ust.nazwa_modelu, contents=tekst)
            return int(wynik.total_tokens)
        except Exception:
            return max(1, len(tekst) // 4)

    # ------------------------------------------------------------------ #
    # Główne API klasy                                                    #
    # ------------------------------------------------------------------ #
    def odpowiedz(self, wiadomosc_uzytkownika: str) -> str:
        """Przyjmuje wiadomość użytkownika i zwraca odpowiedź modelu.

        Pełna ścieżka: dopisanie do historii -> kontrola kontekstu ->
        wywołanie modelu (z ponawianiem) -> zapis odpowiedzi -> logowanie.
        """
        if not wiadomosc_uzytkownika or not wiadomosc_uzytkownika.strip():
            raise BladOdpowiedziModelu("Pusta wiadomość użytkownika.")

        self.historia.dodaj("user", wiadomosc_uzytkownika.strip())

        # Kontrola kontekstu PRZED wysłaniem — przycinamy najstarsze wpisy.
        try:
            usuniete = self.historia.przytnij_do_budzetu()
            if usuniete:
                self.log_zapytan.info("Kontrola kontekstu: usunieto %d najstarszych wiadomosci", usuniete)
        except LimitKontekstuPrzekroczony as e:
            self.historia.cofnij_ostatnia()  # nie zostawiamy "wiszącej" wiadomości
            self.log_bledow.warning("Przekroczenie limitu kontekstu: %s", e)
            raise

        czas_start = time.perf_counter()
        try:
            tekst = self._wywolaj_model()
        except (BladPolaczenia, BladOdpowiedziModelu) as e:
            # Wycofujemy wiadomość użytkownika, by historia pozostała spójna.
            self.historia.cofnij_ostatnia()
            self.log_bledow.error("%s: %s", type(e).__name__, e)
            raise

        czas_ms = (time.perf_counter() - czas_start) * 1000
        self.historia.dodaj("model", tekst)
        self.log_zapytan.info(
            "OK | dlugosc_pytania=%d zn. | tokeny_kontekstu=%d | czas=%.0f ms",
            len(wiadomosc_uzytkownika),
            self.historia.liczba_tokenow(),
            czas_ms,
        )
        return tekst

    def _wywolaj_model(self) -> str:
        """Wywołuje API Gemini z ponawianiem i mapuje błędy na własne wyjątki."""
        konfiguracja = types.GenerateContentConfig(
            system_instruction=self.ust.prompt_systemowy,
            temperature=self.ust.temperatura,
            top_p=self.ust.top_p,
            top_k=self.ust.top_k,
            max_output_tokens=self.ust.maks_tokenow_odpowiedzi,
            thinking_config=types.ThinkingConfig(thinking_budget=self.ust.budzet_myslenia),
        )

        ostatni_blad: Optional[Exception] = None
        for proba in range(1, self.ust.liczba_prob + 1):
            try:
                odpowiedz = self.klient.models.generate_content(
                    model=self.ust.nazwa_modelu,
                    contents=self.historia.do_formatu_gemini(),
                    config=konfiguracja,
                )
                tekst = (odpowiedz.text or "").strip()
                if not tekst:
                    # Pusta odpowiedź = najczęściej blokada filtrów bezpieczeństwa.
                    raise BladOdpowiedziModelu(
                        "Model zwrócił pustą odpowiedź (możliwa blokada treści)."
                    )
                return tekst

            except genai_errors.APIError as e:
                # Błędy zwrócone przez API Google (limity, 5xx, autoryzacja).
                ostatni_blad = BladPolaczenia(f"Błąd API Gemini: {e}")
                self.log_bledow.warning("Proba %d/%d nieudana: %s", proba, self.ust.liczba_prob, e)
                time.sleep(self.ust.opoznienie_ponowienia_s * proba)

            except BladOdpowiedziModelu:
                raise  # nie ponawiamy — problem nie jest przejściowy

            except Exception as e:  # np. brak sieci, timeout gniazda
                ostatni_blad = BladPolaczenia(f"Błąd połączenia z modelem: {e}")
                self.log_bledow.warning("Proba %d/%d nieudana: %s", proba, self.ust.liczba_prob, e)
                time.sleep(self.ust.opoznienie_ponowienia_s * proba)

        # Wyczerpaliśmy próby.
        raise ostatni_blad or BladPolaczenia("Nieznany błąd połączenia.")

    # ------------------------------------------------------------------ #
    # Pomocnicze                                                          #
    # ------------------------------------------------------------------ #
    def wyczysc_historie(self) -> None:
        """Resetuje rozmowę (prompt systemowy pozostaje)."""
        self.historia.wiadomosci.clear()
        self.log_zapytan.info("Historia rozmowy wyczyszczona.")
