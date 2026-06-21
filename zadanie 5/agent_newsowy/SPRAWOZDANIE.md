# Sprawozdanie — Zadanie 5: Agent AI

**Przedmiot:** Aktualne Problemy Informatyki Stosowanej — Geoinformatyka II st.
**Model:** Gemini 2.5 Flash (natywne function calling)

---

## Wprowadzenie

Zbudowałem autonomicznego agenta, który na zadany temat sam przygotowuje raport
prasowy w PDF. Agent dostaje tylko cel („zrób raport o ostatnich wydarzeniach na
temat X") i zestaw narzędzi, a **sam decyduje**, których narzędzi i w jakiej
kolejności użyć: najpierw wyszukuje wiadomości, potem je streszcza, ocenia
istotność tematu i na końcu zapisuje raport do pliku PDF, którego nazwa zawiera
ocenioną istotność.

Celowo nie korzystałem z gotowego frameworka agentowego (LangChain/LangGraph).
Zamiast tego napisałem **własną pętlę** opartą na natywnym mechanizmie *function
calling* modelu Gemini. Dzięki temu dokładnie kontroluję i loguję każdą decyzję
agenta, a sam mechanizm „rozumowania i działania" jest dla mnie w pełni
przejrzysty.

Cały system uruchamiałem na żywo — przykłady (z prawdziwymi danymi z internetu i
wygenerowanym plikiem PDF) są na końcu.

---

## Co po kolei zrobiłem

Zacząłem od czterech narzędzi, bo to one wyznaczają, co agent w ogóle potrafi:
wyszukiwanie wiadomości, streszczanie, ocena istotności i zapis do PDF. Każde
napisałem jako zwykłą funkcję z typowanymi argumentami i dokładnym docstringiem,
bo to z nich model odczytuje, do czego narzędzie służy.

Potem napisałem pętlę agenta: wysyłam do modelu polecenie i deklaracje narzędzi,
odbieram jego decyzję (albo żądanie wywołania narzędzia, albo gotową odpowiedź),
wykonuję narzędzie i zwracam wynik z powrotem jako „obserwację" — i tak w kółko, aż
agent uzna cel za osiągnięty. Na końcu dopracowałem prompt systemowy (cel + zasady,
bez sztywnej listy kroków) oraz obsługę limitów darmowego API (ponawianie po
błędzie 429).

---

## 1. Architektura Agenta AI

**Przepływ danych między narzędziami a modelem.** Sercem systemu jest pętla
„rozumowanie → działanie → obserwacja" (klasyczny schemat ReAct), zaimplementowana
ręcznie w `rdzen/agent.py`:

```
   ┌─────────────────────────────────────────────┐
   │  model Gemini (orkiestrator)                 │
   │  decyduje: gotowa odpowiedź czy wywołanie?   │
   └───────────────┬──────────────────────────────┘
                   │ żądanie wywołania (function_call)
                   ▼
        AgentNewsowy uruchamia funkcję
                   │ wynik = "obserwacja"
                   ▼
        obserwacja wraca do modelu (function_response)
                   │
                   └──► kolejna decyzja ... aż do końca
```

Model nie wykonuje kodu samodzielnie — zwraca jedynie **nazwę narzędzia i
argumenty** (w formacie wygenerowanym na podstawie docstringów). Moja pętla
przechwytuje to żądanie, uruchamia odpowiednią funkcję Pythona i **zwraca surowy
wynik** modelowi jako obserwację. Model czyta obserwację i decyduje o kolejnym
kroku.

**Co się stanie, gdy narzędzie zwróci błąd.** Program się nie wywraca. Błąd jest
przechwytywany (`_wywolaj_narzedzie` opakowuje wywołanie w `try/except`) i zwracany
modelowi jako zwykła obserwacja w stylu `Błąd narzędzia ...: ...`. Model czyta ten
komunikat i może zareagować — np. ponowić wyszukiwanie z innym sformułowaniem.
Dodatkowo same narzędzia są odporne: gdy brakuje klucza NewsAPI, narzędzie
wyszukiwania **samo przełącza się** na darmowy kanał RSS, zamiast zwracać błąd.

**Czy agent może używać narzędzi wielokrotnie.** Tak. Pętla nie ma narzuconej
liczby ani kolejności wywołań (jest tylko bezpiecznik `maks_iteracji_agenta`).
Agent może wywołać to samo narzędzie kilka razy — np. ponowić `szukaj_wiadomosci`,
jeśli pierwsze wyniki były puste lub nietrafione.

**Mechanizm samokorekty.** Wynika on właśnie z trzech powyższych własności:
iteracyjnej pętli, zwracania błędów jako obserwacji oraz promptu, który wprost
instruuje, by nie przerywać po błędzie, tylko spróbować inaczej. W praktyce: jeśli
`szukaj_wiadomosci` zwróci `{"blad": "Brak wyników..."}`, agent dostaje tę
informację i — zgodnie z promptem — ponawia wyszukiwanie ze zmienionym tematem,
zamiast od razu próbować streszczać pustkę.

---

## 2. Model językowy

**Wybrany model: `gemini-2.5-flash`.** Wybrałem go z dwóch głównych powodów. Po
pierwsze, dobrze obsługuje **function calling** — niezawodnie zwraca poprawne
wywołania narzędzi z sensownymi argumentami, co dla agenta jest kluczowe. Po
drugie, jest szybki i mam do niego klucz, a do wieloetapowego planowania (4 kroki)
jego możliwości w zupełności wystarczają.

Przy okazji testów zrobiłem praktyczne obserwacje warte odnotowania. Sprawdziłem
też wariant **`gemini-2.5-flash-lite`** — jest lżejszy i ma osobny limit, ale
okazał się **za słaby do wieloetapowej pętli**: po wyszukiwaniu wiadomości
przerywał, zamiast kontynuować do streszczenia i PDF. To dobrze pokazuje, że dla
zadań agentowych liczy się zdolność modelu do planowania, nie tylko do pojedynczej
odpowiedzi. Z kolei `gemini-2.0-flash` nie jest już dostępny w darmowym planie
(limit 0). Wniosek: do tego zadania właściwym wyborem jest pełny `gemini-2.5-flash`.

---

## 3. Prompt systemowy

**Jak jest skonstruowany.** Prompt (`rdzen/prompty.py`) jest celowo zbudowany wokół
**celu i zasad**, a nie wokół sztywnego algorytmu. Zawiera: rolę agenta, jasno
postawiony cel, listę dostępnych narzędzi z krótkim opisem oraz cztery zasady
działania (opieraj się tylko na danych z narzędzi; zachowaj logiczny przepływ
danych; samodzielnie koryguj błędy; zakończ po zapisaniu PDF).

**Zastosowane techniki promptowania:**
- **Wzorzec roli (persona):** „Jesteś samodzielnym agentem-researcherem prasowym" —
  ustawia kontekst i ton.
- **Promptowanie przez ograniczenia:** twarde zasady („opieraj się WYŁĄCZNIE na
  danych z narzędzi", „nie zmyślaj") ograniczają halucynacje i pilnują, by dane
  faktycznie przepływały między narzędziami.
- **Schemat ReAct:** prompt nawiązuje do cyklu działanie → obserwacja, zgodnie z
  tym, jak działa pętla.
- **Brak hard-coded workflow:** świadomie nie podaję sztywnej numerowanej procedury
  — wymieniam narzędzia i cel, a kolejność zostawiam modelowi (to istota agenta).

**Jak umożliwiłem samokorektę.** Osobna zasada mówi wprost: „jeśli narzędzie zwróci
błąd lub puste/niepełne dane, nie przerywaj — przeanalizuj komunikat i spróbuj
inaczej". W połączeniu z tym, że błędy faktycznie trafiają do modelu jako
obserwacje, daje to działającą pętlę samokorekty.

**Jak zdefiniowałem cel.** Cel jest podany na początku promptu i powtórzony w
pierwszej wiadomości użytkownika: przygotować raport PDF o ostatnich wydarzeniach na
zadany temat. Zasada nr 4 („zakończ dopiero po pomyślnym zapisaniu PDF") działa jak
**warunek zakończenia** — agent wie, kiedy zadanie jest skończone.

---

## 4. Funkcje (narzędzia)

Zaimplementowałem cztery narzędzia (`rdzen/narzedzia.py`):

1. **`szukaj_wiadomosci(temat: str) -> str`** — wyszukuje najnowsze artykuły.
   Korzysta z NewsAPI (endpoint `everything`), a gdy brak klucza — z darmowego
   kanału Google News RSS. Zwraca JSON z listą artykułów (tytuł, opis, URL, data).
   Agent używa go jako pierwszego kroku, by zdobyć materiał źródłowy.
2. **`podsumuj_wiadomosci(dane_wiadomosci: str, temat: str) -> str`** — to
   „pod-narzędzie" oparte na LLM: bierze surowe artykuły i syntetyzuje je w spójne
   podsumowanie w Markdown (nie listuje, lecz łączy wątki). Agent używa go po
   zebraniu wiadomości.
3. **`ocen_istotnosc(podsumowanie: str, temat: str) -> str`** — ocenia istotność
   informacji dla dziedziny, zwracając jedno z trzech słów: `wysoka`/`srednia`/
   `niska`, według jasno zdefiniowanych kryteriów (przełomowość, pilność, związek z
   tematem).
4. **`zapisz_raport_pdf(tresc_markdown, temat, istotnosc) -> str`** — narzędzie
   końcowe: konwertuje Markdown → HTML → PDF i zapisuje plik o nazwie
   `<data>_<temat>_<istotność>.pdf`.

**Typowanie i docstringi.** Każde narzędzie ma **typowane parametry i wartość
zwracaną** (np. `temat: str -> str`) oraz **dokładny docstring**. Nie jest to
kosmetyka — model językowy buduje schemat narzędzia właśnie na podstawie sygnatury i
docstringa, i to z nich „rozumie", do czego narzędzie służy, jakich argumentów
oczekuje i kiedy go użyć. Dlatego docstringi piszę konkretnie i wskazuję w nich
moment użycia (np. „użyj jako PIERWSZEGO kroku", „użyj jako OSTATNIEGO kroku").
W mojej implementacji przekazuję funkcje Pythona wprost do konfiguracji modelu, a
SDK Gemini automatycznie tworzy z nich deklaracje narzędzi.

---

## 5. Podejmowanie decyzji przez agenta

Agent nie wykonuje zaszytego skryptu. Po otrzymaniu polecenia czyta opisy
(docstringi) dostępnych narzędzi i na podstawie bieżącego stanu pętli generuje
decyzję: albo wywołanie konkretnego narzędzia z wygenerowanymi argumentami, albo
gotową odpowiedź końcową. Wybór narzędzia to dopasowanie **intencji** („potrzebuję
najpierw materiału źródłowego") do **opisu narzędzia** („wyszukuje najnowsze
artykuły, użyj jako pierwszego kroku"). Po każdym wyniku model ocenia, czego jeszcze
brakuje do celu, i podejmuje kolejną decyzję. To dlatego docstringi są tak ważne —
to one są dla modelu „instrukcją obsługi" narzędzi.

---

## 6. Przykłady wyników i ocena jakości

### Autonomiczny wybór narzędzi (prawdziwy ślad)

Przy uruchomieniu na temacie „energetyka jądrowa w Polsce" agent samodzielnie
wykonał kolejne kroki we właściwej kolejności (log z pętli):

```
Agent wybrał narzędzie: szukaj_wiadomosci   | argumenty: ['temat']
Agent wybrał narzędzie: podsumuj_wiadomosci | argumenty: ['dane_wiadomosci', 'temat']
Agent wybrał narzędzie: ocen_istotnosc      | argumenty: ['podsumowanie', 'temat']
```

Nie podałem mu tej kolejności — wynika ona z jego własnej decyzji opartej na opisach
narzędzi. Widać też, że poprawnie przekazuje dane między krokami (wynik wyszukiwania
trafia jako `dane_wiadomosci` do streszczenia, a streszczenie jako `podsumowanie` do
oceny istotności).

### Pełny przebieg i wygenerowany raport (prawdziwe dane)

Uruchomienie całego potoku na realnych wiadomościach dało:

```
>>> 1. szukaj_wiadomosci   -> pobrano dane (2960 znaków)
>>> 2. podsumuj_wiadomosci ->
   ## Energetyka Jądrowa w Polsce: Kluczowe Wydarzenia i Plany
   ### Cele i Strategia
   * Polska planuje budowę dwóch dużych elektrowni jądrowych...
   * Docelowa moc ma znacząco zwiększyć krajowe moce wytwórcze...
>>> 3. ocen_istotnosc      -> srednia
>>> 4. zapisz_raport_pdf   ->
   Sukces: raport zapisany w pliku
   raporty/2026-06-21_energetyka_jadrowa_w_polsce_srednia.pdf
```

Powstał prawdziwy plik PDF z nazwą w wymaganym formacie
`<data>_<temat>_<istotność>.pdf`, zawierający sformatowane podsumowanie i nagłówek z
oceną istotności.

### Ocena jakości

Wyniki oceniam dobrze. Podsumowania są spójne, rzeczowe i trzymają się danych z
wyszukiwania — model łączy powtarzające się wątki zamiast je listować. Ocena
istotności jest sensowna, a nazewnictwo plików (z transliteracją polskich znaków,
np. „jądrowa" → „jadrowa") i formatowanie PDF działają poprawnie.

Słabsze strony i wnioski praktyczne:
- **Limity darmowego API.** Agent wykonuje kilka wywołań modelu na sesję, więc
  szybko wyczerpuje darmowe limity (dzienny 20 zapytań dla pełnego flash, minutowy
  5/min). Dlatego dodałem ponawianie po 429, a do testów używałem też modeli z
  osobnym budżetem. W zastosowaniu produkcyjnym potrzebny byłby płatny plan.
- **Dobór modelu.** Lżejszy `flash-lite` bywa zawodny w wieloetapowej pętli (potrafi
  przerwać po pierwszym kroku), więc do agenta warto użyć pełnego modelu.
- **NewsAPI jest anglocentryczne.** Dla tematów po polsku potrafi zwrócić zero
  wyników — dlatego narzędzie wyszukiwania w takiej sytuacji automatycznie
  przełącza się na kanał Google News RSS, który dobrze pokrywa polskie zapytania.

---

## Podsumowanie i możliwy rozwój

Agent realizuje pełny, samodzielny przepływ: od wyszukania wiadomości, przez
streszczenie i ocenę istotności, po zapis raportu PDF — bez sztywno zakodowanej
kolejności kroków, z obsługą błędów i samokorektą. Naturalnym rozszerzeniem (ocena
5.0) byłoby dołożenie narzędzia do **głębokiego pobierania treści artykułów**
(np. biblioteką Crawl4AI): agent nie tylko czytałby nagłówki z NewsAPI, ale wchodził
pod linki, pobierał pełną treść i na jej podstawie tworzył bogatsze podsumowanie.
