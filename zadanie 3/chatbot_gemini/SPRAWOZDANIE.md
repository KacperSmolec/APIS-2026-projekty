# Sprawozdanie — Zadanie 3: Production Ready Chatbot

**Przedmiot:** Aktualne Problemy Informatyki Stosowanej — Geoinformatyka II st.
**Model:** Google Gemini 2.5 Flash (przez API)

---

## Wprowadzenie

W ramach tego zadania zbudowałem chatbota `GeoBot`, który ma pełnić rolę asystenta
z dziedziny geoinformatyki — odpowiadać na pytania z zakresu GIS, teledetekcji,
analizy przestrzennej i programowania w Pythonie. Całość oparłem o model językowy
**Gemini 2.5 Flash**, do którego łączę się przez API.

Zależało mi na tym, żeby nie był to tylko skrypt wysyłający zapytania, ale program
zorganizowany tak, jak organizuje się rzeczy przeznaczone do dłuższego życia:
z podziałem na moduły, sensowną obsługą błędów, logowaniem i testami. Stąd całą
logikę zamknąłem w klasie i rozbiłem projekt na osobne pliki odpowiadające za
poszczególne zadania (rozmowa, historia, konfiguracja, logi, wyjątki).

Poniżej opisuję, jak system działa i dlaczego podjąłem takie, a nie inne decyzje.
Wszystkie elementy uruchamiałem na żywo — przykłady z prawdziwych przebiegów są na
końcu.

---

## Co po kolei zrobiłem

Zaczynałem od wyboru modelu. Padło na Gemini 2.5 Flash, bo mam do niego darmowy
klucz, dobrze radzi sobie z polskim i jest szybki. Potem zaprojektowałem strukturę
projektu — zamiast trzymać wszystko w jednym pliku rozdzieliłem odpowiedzialności,
żeby kod dało się czytać i testować po kawałku.

Kolejnym krokiem była sama klasa chatbota: złożenie promptu systemowego, historii
rozmowy i parametrów generacji w jedno wywołanie modelu. Gdy to działało, dodałem
kontrolę długości kontekstu (liczę ją w tokenach), a na końcu obudowałem wszystko
obsługą błędów i logowaniem. Na bieżąco pisałem testy, które sprawdzają logikę bez
łączenia się z siecią.

---

## 1. Architektura systemu

Projekt ma budowę modularną — każdy plik odpowiada za jedną rzecz:

```
Użytkownik (CLI / notebook / kod)
        │  wiadomość tekstowa
        ▼
┌──────────────────────────────────────────────┐
│              ChatbotGemini  (rdzen/chatbot.py)│
│  • prompt systemowy                            │
│  • spina cały przepływ                         │
└───────┬───────────────┬───────────────┬───────┘
        │               │               │
        ▼               ▼               ▼
 HistoriaRozmowy    logowanie       wywołanie modelu
 (historia.py)      (logowanie.py)  google-genai → Gemini
 • przechowywanie   • zapytania.log  • parametry generacji
 • kontrola         • bledy.log      • ponawianie
   kontekstu (tokeny)                • mapowanie błędów → wyjatki.py
```

Najważniejszy jest przepływ przy pojedynczym zapytaniu (metoda
`ChatbotGemini.odpowiedz`). Wygląda on tak:

1. Wiadomość użytkownika jest sprawdzana i dopisywana do historii.
2. Liczę, ile tokenów zajmuje cały kontekst, i jeśli trzeba — przycinam najstarsze
   wiadomości (o tym w punkcie 6).
3. Składam zapytanie: prompt systemowy + historia w formacie Gemini + parametry
   generacji, i wysyłam do modelu (z ponawianiem, jeśli coś pójdzie nie tak).
4. Odpowiedź trafia z powrotem do historii, a do logu zapytań idzie wpis z
   metadanymi (czas, liczba tokenów).
5. Jeśli po drodze wystąpi błąd, zapisuję go do logu błędów, **wycofuję** wiadomość
   użytkownika z historii (żeby nie została „wisząca") i zgłaszam odpowiedni wyjątek
   warstwie wyżej.

Komponenty, o których warto wspomnieć osobno: **model LLM** (Gemini, generuje
odpowiedzi), **historia rozmowy** (lista wiadomości doklejana do każdego zapytania,
bo model sam z siebie nie pamięta nic z poprzednich tur) oraz **kontroler
kontekstu**, który pilnuje, żeby rozmowa nie urosła ponad ustalony budżet tokenów.

---

## 2. Model językowy i sposób integracji

Wybrałem **`gemini-2.5-flash`**. Wariant „flash" jest szybki i tani — w moich
testach odpowiedź przychodziła zwykle w 0,6–1,1 sekundy — a do prostego asystenta
nie potrzebuję najcięższego modelu. Wersja „pro" byłaby dokładniejsza, ale
wolniejsza i z ostrzejszymi limitami w darmowym planie, więc uznałem, że nie ma
sensu jej tu używać.

Model działa **przez API** (w chmurze Google), nie lokalnie. Główna zaleta to brak
wymagań sprzętowych — nie muszę mieć GPU ani uruchamiać lokalnego serwera modelu, a
start jest natychmiastowy. Wadą jest zależność od sieci i limity darmowego planu,
ale te ostatnie obsługuję (patrz punkt 7).

Z bibliotek użyłem:

- **`google-genai`** — nowe, oficjalne SDK Google do modeli Gemini. Wybrałem je
  zamiast starszego `google-generativeai`, bo jest aktualnie rozwijane i ma
  spójny interfejs: `generate_content`, `count_tokens` oraz `ThinkingConfig`
  (przydało się — patrz punkt 5).
- **`python-dotenv`** — żeby trzymać klucz i ustawienia w pliku `.env`, z dala od
  kodu.
- **`pytest`** — do testów (choć testy napisałem tak, że działają też bez niego).

---

## 3. Prompt systemowy

Prompt systemowy to instrukcja, która ustawia całą rozmowę: mówi modelowi, kim
jest, w czym się specjalizuje, w jakim języku i stylu ma odpowiadać. Działa z
wyższym priorytetem niż pojedyncze wiadomości użytkownika, dzięki czemu bot przez
całą rozmowę zachowuje spójny charakter i nie zaczyna nagle odpowiadać po angielsku
ani odpływać od tematu.

Mój prompt (z `rdzen/konfiguracja.py`) brzmi:

> „Jesteś asystentem geoinformatycznym o imieniu GeoBot. Pomagasz studentom
> kierunku Geoinformatyka w zagadnieniach z zakresu GIS, teledetekcji, analizy
> przestrzennej oraz programowania w Pythonie. Odpowiadasz domyślnie po polsku
> (chyba że użytkownik poprosi inaczej), rzeczowo i konkretnie. Jeśli czegoś nie
> wiesz lub pytanie wykracza poza Twoją
> wiedzę, otwarcie się do tego przyznajesz zamiast zmyślać. Gdy to pomocne,
> podajesz krótki przykład kodu lub wzór."

Świadomie upchnąłem w nim kilka rzeczy naraz: rolę i dziedzinę, domyślny język
polski (bez tego model lubi przeskakiwać na angielski, ale zostawiam furtkę, by
odpowiedział inaczej na wyraźną prośbę), oczekiwany styl (zwięzłość) oraz prośbę
o przyznawanie się do niewiedzy — to ostatnie realnie
ogranicza zmyślanie. W SDK Gemini przekazuję go jako `system_instruction`, czyli
dedykowanym kanałem, a nie jako zwykłą wiadomość.

---

## 4. Historia rozmowy

Historię trzymam jako listę obiektów `Wiadomosc(rola, tresc)`
(`rdzen/historia.py`), gdzie `rola` to `"user"` albo `"model"` — taką konwencję
przyjmuje Gemini. Prompt systemowy trzymam **osobno** od dialogu i nigdy go nie
usuwam.

Jest to potrzebne, bo model językowy jest bezstanowy — przy każdym zapytaniu
zaczyna „od zera" i nie pamięta poprzednich tur. Dlatego za każdym razem
przepisuję całą dotychczasową historię do formatu API:

```python
[{"role": "user",  "parts": [{"text": "..."}]},
 {"role": "model", "parts": [{"text": "..."}]},
 ...]
```

i wysyłam ją razem z nowym pytaniem. Dzięki temu model „widzi" całą rozmowę i może
odwoływać się do tego, co padło wcześniej. Sprawdziłem, że to działa — w
przykładach na końcu bot poprawnie rozumie pytania w stylu „a jaki ma zakres?", nie
pytając, o co mi chodzi.

---

## 5. Parametry modelu

Ustawiłem następujące parametry generacji (`rdzen/konfiguracja.py`):

| Parametr | Wartość | Za co odpowiada |
|---|---|---|
| `temperatura` | 0.4 | losowość doboru słów — niska oznacza odpowiedzi spójne i powtarzalne |
| `top_p` | 0.9 | *nucleus sampling* — model wybiera z najmniejszego zbioru słów o łącznym prawdopodobieństwie 0.9 |
| `top_k` | 40 | ogranicza wybór do 40 najbardziej prawdopodobnych tokenów |
| `max_output_tokens` | 1024 | maksymalna długość odpowiedzi |
| `thinking_budget` | 0 | wyłączone rozszerzone „myślenie" Gemini 2.5 |

Temperatura to dla mnie najważniejszy parametr. Przy wysokiej (≈1.0) odpowiedzi są
zróżnicowane, czasem zaskakujące, ale i mniej przewidywalne; przy niskiej (≈0.1)
robią się niemal deterministyczne. Dla asystenta, od którego oczekuję poprawnych i
powtarzalnych odpowiedzi, ważniejsza jest stabilność niż kreatywność, więc
ustawiłem ją dość nisko, na 0.4. `top_p` i `top_k` działają podobnie — odcinają
mało prawdopodobny „ogon" słów, przez co model rzadziej wtrąca bzdury.
`max_output_tokens` trzymam na 1024, żeby odpowiedzi nie rozlewały się w
nieskończoność. Wszystkie te wartości da się zmienić przez zmienne środowiskowe,
bez ruszania kodu.

Przy okazji natknąłem się na ciekawą rzecz. Gemini 2.5 Flash domyślnie zużywa część
`max_output_tokens` na wewnętrzne „myślenie", przez co przy małym limicie zaczęło
mi **ucinać właściwą odpowiedź** w połowie zdania. Ustawienie `thinking_budget=0`
to rozwiązało — dla zwykłej rozmowy nie potrzebuję rozszerzonego rozumowania, a
zyskuję przewidywalną długość odpowiedzi i krótszy czas oczekiwania.

---

## 6. Kontrola długości kontekstu

Długość kontekstu kontroluję **w tokenach**, a nie w liczbie wiadomości. Spotkałem
się częściej z tym drugim podejściem („zostaw ostatnie N wiadomości"), ale wydało
mi się nieprecyzyjne — jedna wiadomość może mieć pięć tokenów, a inna pięćset, a to
właśnie tokeny, nie liczba wpisów, decydują o tym, czy zmieścimy się w oknie modelu
i ile za to zapłacimy.

Działa to tak (`HistoriaRozmowy.przytnij_do_budzetu`): przed każdym wywołaniem
modelu sumuję tokeny promptu systemowego i całego dialogu (licząc je przez
`count_tokens` z API, z lokalnym przybliżeniem ~4 znaki na token jako
zabezpieczeniem na wypadek braku sieci). Jeśli suma przekracza ustawiony budżet,
usuwam najstarsze wiadomości dialogu, aż kontekst się zmieści. Prompt systemowy
przy tym **zostaje nietknięty** — nie chcę, żeby bot zapomniał, kim jest.

Po przekroczeniu limitu są możliwe dwa scenariusze. Zwykle po prostu „zapominane"
są najstarsze tury rozmowy (zapisuję to w logu: `Kontrola kontekstu: usunieto N
najstarszych wiadomosci`) i rozmowa toczy się dalej. Gorszy przypadek to sytuacja,
gdy sama pojedyncza wiadomość użytkownika jest dłuższa niż cały budżet — wtedy
przycinanie nic nie da, więc zgłaszam wyjątek `LimitKontekstuPrzekroczony`, wycofuję
tę wiadomość i proszę użytkownika o jej skrócenie.

---

## 7. Obsługa błędów

Zdefiniowałem własne wyjątki (`rdzen/wyjatki.py`), wszystkie dziedziczące po
`BladChatbota`. Dzięki temu kod, który korzysta z chatbota, nie musi grzebać w
szczegółach SDK Google — reaguje na czytelne, „domenowe" błędy:

| Wyjątek | Kiedy leci | Co robię |
|---|---|---|
| `BladPolaczenia` | timeout, brak sieci, błąd API (4xx/5xx), przekroczony limit zapytań (429) | ponawiam (domyślnie 3 próby z rosnącym odstępem), a po wyczerpaniu prób pokazuję komunikat |
| `BladOdpowiedziModelu` | pusta odpowiedź / blokada filtrów / dziwny format | nie ponawiam, bo to nie jest błąd przejściowy |
| `LimitKontekstuPrzekroczony` | pojedyncza wiadomość > budżet tokenów | wycofuję wiadomość i proszę o skrócenie |

Trzy rzeczy uważam tu za najważniejsze. Po pierwsze, **ponawianie z narastającym
odstępem** — błędy sieciowe i 429 często są chwilowe, więc warto spróbować jeszcze
raz, zanim się poddam. Po drugie, **wycofywanie wiadomości po błędzie** — gdyby
zostawić w historii pytanie bez odpowiedzi, struktura user/model przestałaby się
przeplatać i kolejne zapytania mogłyby się sypać. Po trzecie, **tłumaczenie błędów
technicznych na własne typy**, o czym już wspomniałem.

Logi rozdzieliłem na dwa pliki (`rdzen/logowanie.py`), bo mają różne zastosowania.
Do `logi/zapytania.log` idą udane zapytania wraz z metadanymi (model, długość
pytania, liczba tokenów kontekstu, czas odpowiedzi) — to materiał, z którego można
później policzyć zużycie i koszty. Do `logi/bledy.log` trafiają wyłącznie
ostrzeżenia i błędy (dodatkowo wyświetlam je na konsoli), więc plik jest krótki i
łatwo go przejrzeć, gdy coś przestaje działać.

---

## Przykłady działania

Wszystko poniżej pochodzi z rzeczywistych uruchomień.

### Rozmowa wieloturowa z pamięcią i przycinaniem kontekstu

Uruchomiłem chatbota z celowo małym budżetem kontekstu (230 tokenów), żeby pokazać,
jak przycinanie wchodzi do gry:

```
>>> Tura 1: W jednym zdaniu: czym jest NDVI?
    NDVI (Normalized Difference Vegetation Index) to wskaźnik teledetekcyjny
    służący do oceny kondycji i zagęszczenia roślinności, obliczany na podstawie
    różnicy odbicia w bliskiej podczerwieni i czerwieni.
    [kontekst: 190 tok | wiadomosci w pamieci: 2]

>>> Tura 2: Jaki ma zakres wartosci? Krotko.
    Zakres wartości NDVI wynosi od -1 do +1.
    [kontekst: 215 tok | wiadomosci w pamieci: 4]

>>> Tura 3: Jeden przyklad zastosowania w rolnictwie, krotko.
    W rolnictwie NDVI jest używane do monitorowania zdrowia upraw i wykrywania
    obszarów stresu roślin.
    [kontekst: 249 tok | wiadomosci w pamieci: 5]   ← przycięta 1 najstarsza

>>> Tura 4: Czym rozni sie od EVI? Jedno zdanie.
    EVI różni się od NDVI tym, że jest mniej wrażliwy na wpływ atmosfery i
    nasycenie sygnału w obszarach o gęstej roślinności...
    [kontekst: 268 tok | wiadomosci w pamieci: 6]   ← przycięta kolejna
```

Odpowiedzi są poprawne, zwięzłe i po polsku — o to mi chodziło. Najlepiej widać tu
działanie historii: w turze 2 i 4 pytam „jaki ma zakres?" i „czym różni się od
EVI?", nie wspominając już o NDVI, a bot i tak wie, o co chodzi. Widać też, że
liczba pamiętanych wiadomości przestaje rosnąć liniowo — to efekt przycinania.

Fragment `logi/zapytania.log` z tego przebiegu:

```
... | OK | dlugosc_pytania=32 zn. | tokeny_kontekstu=215 | czas=579 ms
... | Kontrola kontekstu: usunieto 1 najstarszych wiadomosci
... | OK | dlugosc_pytania=49 zn. | tokeny_kontekstu=249 | czas=916 ms
```

### Błąd połączenia (celowo zły klucz API)

```
Zlapano BladPolaczenia (zgodnie z oczekiwaniem).
Wiadomosci w historii po bledzie: 0 (spojna - wycofano)
```

`logi/bledy.log`:

```
... | WARNING | Proba 1/2 nieudana: 400 INVALID_ARGUMENT ... API key not valid ...
... | WARNING | Proba 2/2 nieudana: 400 INVALID_ARGUMENT ... API key not valid ...
... | ERROR   | BladPolaczenia: Błąd API Gemini: 400 INVALID_ARGUMENT ...
```

Widać tu zarówno ponawianie, jak i to, że po nieudanej próbie historia zostaje
pusta — wiadomość została wycofana, więc nic się nie rozjechało.

### Limit darmowego planu (HTTP 429) — z prawdziwego życia

Przy intensywniejszym testowaniu API zwróciło `429 RESOURCE_EXHAUSTED` — darmowy
plan pozwala na 5 zapytań na minutę. Co ciekawe, nie musiałem nic dorabiać: system
potraktował to jako błąd przejściowy i sam ponowił zapytanie, dokładnie tak, jak
powinien zachować się program przeznaczony do realnego użytku.

---

## Podsumowanie

Chatbot jest w pełni działający — rozmawia, pamięta kontekst, pilnuje jego długości,
loguje swoją pracę i nie wywraca się na błędach sieci czy modelu. Gdybym chciał
pójść dalej, naturalnym krokiem byłoby wystawienie go jako usługi REST (np. przez
FastAPI) i dodanie obsługi wielu użytkowników, każdy z własną historią rozmowy.
Architektura jest na to gotowa: klasa `ChatbotGemini` jest samowystarczalna,
więc wystarczyłoby owinąć ją menedżerem sesji i cienką warstwą HTTP.
