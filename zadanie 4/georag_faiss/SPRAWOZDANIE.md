# Sprawozdanie — Zadanie 4: RAG w zastosowaniach geoinformatycznych

**Przedmiot:** Aktualne Problemy Informatyki Stosowanej — Geoinformatyka II st.
**Temat ekspercki:** klasyfikacja pokrycia terenu z obrazów satelitarnych
**Modele:** Gemini 2.5 Flash (generacja) + gemini-embedding-001 (embeddingi)

---

## Wprowadzenie

Zbudowałem system RAG, który pełni rolę eksperta od jednego, dość wąskiego
zagadnienia z teledetekcji — **klasyfikacji pokrycia terenu (land cover) z obrazów
satelitarnych metodami uczenia maszynowego**. Jako bazę wiedzy zebrałem cztery
publikacje naukowe z arXiv i tak zaprojektowałem system, żeby model językowy
odpowiadał na pytania wyłącznie na podstawie ich treści, a nie z własnej, ogólnej
wiedzy.

Świadomie zrezygnowałem z gotowych frameworków pokroju LangChain i złożyłem cały
potok z osobnych, czytelnych klocków: wczytywanie PDF, dzielenie na fragmenty,
embeddingi, baza wektorowa i generacja. Dzięki temu dokładnie wiem, co dzieje się
na każdym etapie, i mogłem każdy z nich opisać oraz przetestować osobno. Do
przechowywania wektorów wybrałem **FAISS**, a zarówno embeddingi, jak i odpowiedzi
generuję modelami **Gemini**.

Cały potok uruchomiłem na żywo — przykłady odpowiedzi są na końcu sprawozdania.

---

## Co po kolei zrobiłem

Najpierw zebrałem korpus: pobrałem z arXiv cztery prace o klasyfikacji pokrycia
terenu (segmentacja semantyczna obrazów satelitarnych, sieci rekurencyjne dla
danych wieloczasowych, zbiór benchmarkowy PatternNet oraz metoda nadzoru
długozasięgowego) i zapisałem je jako PDF-y w katalogu `dane/`.

Potem zaimplementowałem kolejne etapy potoku. Zacząłem od wczytywania i czyszczenia
tekstu z PDF, następnie napisałem własny dzielnik na fragmenty, podpiąłem model
embeddingowy Gemini i zbudowałem na tym indeks FAISS, który zapisuję na dysk. Na
końcu dołożyłem warstwę generacji: pytanie zamieniam na wektor, wyszukuję najbardziej
podobne fragmenty i wstrzykuję je do promptu jako kontekst dla modelu. Po drodze
natknąłem się na limity darmowego API (o tym w sekcji o embeddingach) i dorobiłem
mechanizm ograniczania tempa zapytań.

---

## 1. Zastosowanie systemów RAG

**Czym jest RAG.** RAG (Retrieval-Augmented Generation) to połączenie wyszukiwania
informacji z generowaniem tekstu. Zamiast pytać model językowy „z pamięci",
najpierw **wyszukujemy** w zewnętrznej bazie wiedzy fragmenty najlepiej pasujące do
pytania, a dopiero potem prosimy model, żeby ułożył odpowiedź **na ich podstawie**.
Model dostaje więc nie tylko pytanie, ale i konkretny materiał źródłowy.

**Czym różni się od zwykłego chatbota.** Zwykły chatbot odpowiada wyłącznie z
wiedzy „zaszytej" w wagach modelu podczas treningu. To rodzi trzy problemy, które
RAG rozwiązuje:

- **Halucynacje** — model potrafi pewnym tonem podać nieprawdę. W RAG odpowiedź ma
  oparcie w konkretnych dokumentach, a model jest instruowany, żeby nie wychodzić
  poza nie. Można wręcz wskazać źródło każdej informacji.
- **Brak wiedzy specjalistycznej i aktualnej** — model nie zna treści, których nie
  było w jego danych treningowych (np. świeżych albo niszowych publikacji). RAG po
  prostu dokłada te dokumenty do bazy, bez przetrenowywania modelu.
- **Brak kontroli nad źródłami** — w RAG dokładnie wiadomo, skąd pochodzi odpowiedź
  (u mnie: nazwa pliku i numer strony), co jest kluczowe w zastosowaniach naukowych.

W moim przypadku oznacza to system, który odpowiada na pytania o klasyfikację
pokrycia terenu wyłącznie w oparciu o zebrane publikacje, a gdy czegoś w nich nie
ma — uczciwie to przyznaje.

---

## 2. Architektura systemu RAG

System ma dwie fazy. **Faza budowy indeksu** (offline, raz na korpus) i **faza
zapytania** (przy każdym pytaniu).

```
BUDOWA INDEKSU:
  PDF ─► WczytywaczPDF ─► DzielnikTekstu ─► ModelEmbeddingow ─► BazaWektorowaFAISS ─► dysk
        (pypdf,           (zdaniowy,         (Gemini,            (IndexFlatIP)
         czyszczenie)      z zakładką)        RETRIEVAL_DOCUMENT)

ZAPYTANIE:
  pytanie ─► ModelEmbeddingow ─► BazaWektorowaFAISS ─► budowa ─► Gemini ─► odpowiedź
            (RETRIEVAL_QUERY)    (top-k, kosinus)      promptu    (LLM)     + źródła
```

Poszczególne komponenty:

**a) Loader dokumentów** (`rdzen/wczytywanie.py`, klasa `WczytywaczPDF`). Wczytuje
wszystkie PDF-y z katalogu, wyciąga tekst strona po stronie i czyści go — skleja
wyrazy przeniesione myślnikiem na końcu wiersza i normalizuje białe znaki, bo
ekstrakcja z PDF lubi wstawiać przypadkowe znaki nowej linii. Każda strona staje
się rekordem z metadanymi (nazwa pliku, numer strony), które niosę dalej aż do
odpowiedzi.

**b) Chunking** (`rdzen/dzielenie.py`, klasa `DzielnikTekstu`). Dzieli strony na
mniejsze fragmenty. Szczegóły w punkcie 6.

**c) Embeddingi** (`rdzen/embeddingi.py`, klasa `ModelEmbeddingow`). Zamienia
fragmenty (a potem pytania) na wektory liczb. Szczegóły w punkcie 4.

**d) Vector store** (`rdzen/baza_wektorowa.py`, klasa `BazaWektorowaFAISS`).
Przechowuje wektory i pozwala szybko znaleźć najbardziej podobne. Szczegóły w
punkcie 7.

**e) Retriever** — to nie osobna klasa, lecz metoda `szukaj` bazy wektorowej w
połączeniu z embeddingiem pytania. Szczegóły w punkcie 5.

Całość spina klasa **`SilnikRAG`** (`rdzen/silnik_rag.py`), która udostępnia dwie
operacje: `zbuduj_indeks()` oraz `zapytaj(pytanie)`.

---

## 3. Generowanie odpowiedzi

Po znalezieniu najlepszych fragmentów składam z nich **kontekst** — numeruję je i
do każdego dopisuję źródło:

```
[1] (źródło: arxiv_2406.14220v2.pdf, str. 3)
<treść fragmentu>

[2] (źródło: ...)
<treść fragmentu>
...
```

Następnie buduję zapytanie do modelu z dwóch części: **promptu systemowego** (stałe
instrukcje) i **treści użytkownika** w formacie `KONTEKST: ... \n PYTANIE: ...`.
Prompt systemowy (`rdzen/konfiguracja.py`) jest celowo rygorystyczny:

> „Odpowiadasz na pytania WYŁĄCZNIE na podstawie fragmentów (...). Jeśli w
> kontekście nie ma odpowiedzi, napisz dokładnie: 'Na podstawie dostarczonych
> publikacji nie potrafię odpowiedzieć na to pytanie.' Nie korzystaj z wiedzy
> spoza kontekstu i niczego nie zmyślaj. (...) Na końcu podaj numery
> wykorzystanych fragmentów, np. [1, 3]."

Trzy rzeczy są tu istotne. Po pierwsze, jasna **instrukcja odmowy** — model ma się
przyznać do niewiedzy zamiast zmyślać. Po drugie, prośba o **wskazanie numerów
fragmentów**, dzięki czemu widać, na czym model oparł odpowiedź. Po trzecie, niska
**temperatura (0.2)** — w systemie eksperckim zależy mi na powtarzalności i
wierności źródłom, a nie na kreatywności. Wyłączam też wewnętrzne „myślenie" modelu
(`thinking_budget=0`), żeby cały budżet tokenów szedł na właściwą odpowiedź.

---

## 4. Embeddingi

**Wybrany model: `gemini-embedding-001`** (768 wymiarów). Wybrałem go z kilku
powodów. Jest dostępny przez to samo API co model generujący, więc nie muszę
instalować ciężkich bibliotek (torch, sentence-transformers) ani trzymać modelu
lokalnie — liczy się to zwłaszcza na słabszym sprzęcie. Jest też wielojęzyczny, co
ma znaczenie, bo publikacje są po angielsku, a ja pytam po polsku — i mimo to
wyszukiwanie działa (pytanie „dane wieloczasowe" trafia w anglojęzyczny fragment o
*multi-temporal data*).

Co ważne, model ma **tryby zadania** (`task_type`), które pozwalają inaczej kodować
dokumenty (`RETRIEVAL_DOCUMENT`) i pytania (`RETRIEVAL_QUERY`). To tzw. embedding
asymetryczny — model „wie", że krótkie pytanie i długi fragment dokumentu pełnią
różne role, i koduje je tak, żeby do siebie pasowały. Korzystam z obu trybów
odpowiednio przy budowie indeksu i przy zapytaniu.

**Jak embeddingi reprezentują tekst.** Embedding to wektor liczb (tutaj 768),
będący „współrzędnymi" fragmentu w wielowymiarowej przestrzeni znaczeń. Model jest
wytrenowany tak, żeby teksty o **podobnym znaczeniu** miały wektory **blisko
siebie** — i to niezależnie od użytych słów. Dzięki temu pytanie i pasujący
fragment są blisko, nawet jeśli nie mają wspólnych słów (np. polskie pytanie i
angielski tekst). Po wygenerowaniu **normalizuję** wektory do długości 1, co
upraszcza liczenie podobieństwa (patrz punkt 5).

**Wpływ na jakość retrieval.** Embeddingi to serce całego systemu — jeśli wektory
źle oddają znaczenie, to nawet najlepszy model językowy dostanie nieistotny
kontekst i odpowie słabo (zasada „śmieci na wejściu, śmieci na wyjściu"). W praktyce
widać to po **podobieństwach**: dla trafnych pytań znalezione fragmenty mają
podobieństwo rzędu 0.72–0.78, a dla pytania spoza tematu spada ono do ~0.45 —
model embeddingowy sam „sygnalizuje", że nic dobrze nie pasuje.

**Praktyczna uwaga o limitach.** Darmowy plan API liczy każdy osadzany fragment
osobno (limit ok. 100/min). Mój korpus to 234 fragmenty, więc dorobiłem w module
embeddingów **ograniczanie tempa** (pilnowanie okna minutowego) oraz **ponawianie
po błędzie 429** z odczekaniem zalecanego czasu. Dzięki temu budowa indeksu
przechodzi do końca mimo limitów.

---

## 5. Retrieval

**Jak działa wyszukiwanie podobieństwa.** Pytanie zamieniam na wektor (tym samym
modelem embeddingowym, w trybie zapytania), a potem szukam w bazie tych fragmentów,
których wektory są najbliżej wektora pytania. „Najbliżej" znaczy: o największym
podobieństwie. Zwracam **k = 4** najlepszych fragmentów (parametr konfigurowalny) —
to kompromis: zbyt mało grozi pominięciem istotnej informacji, zbyt dużo zalewa
model szumem i rozprasza go.

**Metryka podobieństwa: kosinusowa.** Mierzę **podobieństwo kosinusowe**, czyli
kosinus kąta między wektorami (1 = identyczny kierunek/znaczenie, 0 = brak
związku). Wybrałem je, bo dla embeddingów liczy się **kierunek** wektora
(znaczenie), a nie jego długość. Zastosowałem przy tym wygodną sztuczkę: skoro
wcześniej znormalizowałem wszystkie wektory do długości 1, to ich **iloczyn
skalarny równa się podobieństwu kosinusowemu**. Dzięki temu mogłem użyć w FAISS
najprostszego i najszybszego indeksu opartego na iloczynie skalarnym
(`IndexFlatIP`), nie tracąc nic na poprawności.

---

## 6. Chunkowanie dokumentów

**Dlaczego w ogóle dzielę dokumenty.** Z dwóch powodów. Po pierwsze, do bazy
wektorowej chcę wrzucać **spójne, niewielkie kawałki** — embedding całej
50-stronicowej pracy byłby „rozmytą średnią" wielu tematów i do niczego by nie
pasował precyzyjnie. Po drugie, do promptu modelu wstrzykuję tylko znalezione
fragmenty, więc muszą być na tyle krótkie, żeby zmieścić ich kilka, a na tyle
długie, żeby niosły sensowną, samodzielną informację.

**Jak dobrałem długość.** Ustawiłem docelową długość fragmentu na **ok. 180 słów**.
To mniej więcej jeden–dwa akapity naukowe — wystarczająco dużo, by fragment był
zrozumiały sam z siebie, a wystarczająco mało, by był „o jednej rzeczy". Dodatkowo
dałem **zakładkę (overlap) 2 zdań**: ostatnie zdania jednego fragmentu powtarzam na
początku kolejnego. To zabezpiecza przed sytuacją, w której ważna myśl zostaje
przecięta na granicy dwóch fragmentów i żaden z nich nie zawiera jej w całości.

**Zastosowana metoda — dzielenie zdaniowe.** Napisałem własny dzielnik
(`DzielnikTekstu`): najpierw rozbijam tekst na **zdania**, a potem pakuję zdania w
fragmenty, aż uzbieram docelową liczbę słów. Kluczowe jest to, że **nigdy nie tnę w
środku zdania** — granica fragmentu zawsze wypada między zdaniami, więc jest
semantycznie sensowna.

**Alternatywy.** Najprostsza to **dzielenie po stałej liczbie znaków** — szybkie,
ale brutalne (potrafi uciąć słowo czy zdanie w pół). Popularny jest
`RecursiveCharacterTextSplitter`, który próbuje dzielić najpierw po akapitach,
potem zdaniach, potem słowach — to dobry kompromis i to on bywa domyślnym wyborem w
gotowych frameworkach. Na drugim biegunie jest **chunking semantyczny**, gdzie
granice wyznacza się tam, gdzie zmienia się temat (na podstawie embeddingów kolejnych
zdań) — najdokładniejszy, ale i najbardziej kosztowny. Wybrałem dzielenie zdaniowe,
bo jest proste, w pełni pod moją kontrolą i dobrze pasuje do tekstów naukowych,
które mają wyraźną strukturę zdaniową.

---

## 7. Vector Store (FAISS)

**Czym jest baza wektorowa.** To wyspecjalizowana baza do przechowywania wektorów i
**szybkiego wyszukiwania najbardziej podobnych** spośród nich. Zwykła baza danych
odpowiada na pytanie „znajdź rekord równy X", a baza wektorowa na „znajdź rekordy
najbliższe X" w przestrzeni znaczeń — i robi to wydajnie nawet dla milionów
wektorów.

**Wybrane rozwiązanie: FAISS.** Użyłem biblioteki **FAISS** (Facebook AI Similarity
Search) z indeksem `IndexFlatIP`. „Flat" oznacza wyszukiwanie dokładne
(przeglądające wszystkie wektory) — dla mojego korpusu (234 fragmenty) jest
błyskawiczne i nie potrzebuję indeksów przybliżonych. „IP" to iloczyn skalarny,
który — dzięki normalizacji wektorów — daje podobieństwo kosinusowe. FAISS jest
lekki, szybki i działa lokalnie na CPU bez żadnego serwera.

**Jak przechowuję embeddingi i metadane.** FAISS sam trzyma **tylko wektory** — nie
zna treści ani źródeł. Dlatego metadane (tekst fragmentu, plik, numer strony, numer
fragmentu) trzymam **równolegle**, w liście Pythona, w tej samej kolejności co
wektory w indeksie. Przy wyszukiwaniu FAISS zwraca pozycje (indeksy) najlepszych
wektorów, a ja po tych pozycjach odczytuję odpowiadające im fragmenty z listy. Przy
zapisie na dysk serializuję indeks funkcją FAISS (`write_index`) do pliku
`indeks.faiss`, a metadane do `fragmenty.json`. Dzięki temu kolejne uruchomienia
tylko wczytują gotową bazę, bez ponownej (kosztownej) wektoryzacji.

---

## Przykłady działania

Poniższe odpowiedzi pochodzą z prawdziwego uruchomienia na zbudowanym indeksie
(234 fragmenty z 4 publikacji).

### Pytanie w temacie — architektury sieci

```
PYTANIE: Jakie architektury sieci neuronowych stosuje się do klasyfikacji
         pokrycia terenu?

ODPOWIEDŹ: Do klasyfikacji pokrycia terenu stosuje się różne architektury sieci
neuronowych, w tym Unet, Linknet, FPN i PSPNet. Są to modele typu koder-dekoder z
połączeniami pomijającymi, oparte na konwolucyjnych sieciach neuronowych (CNN),
działające na wielu skalach. (...) [1, 2, 4]

ŹRÓDŁA:
   [1] arxiv_2406.14220v2.pdf str.3  sim=0.748
   [2] arxiv_2406.14220v2.pdf str.2  sim=0.739
   [3] arxiv_2309.04225v1.pdf str.13 sim=0.730
   [4] arxiv_2406.14220v2.pdf str.1  sim=0.730
```

### Pytanie w temacie — dane wieloczasowe (pytanie PL, dokumenty EN)

```
PYTANIE: Jakie jest znaczenie danych wieloczasowych (multi-temporal) w
         klasyfikacji pokrycia terenu?

ODPOWIEDŹ: Wykorzystanie szeregów czasowych obrazów satelitarnych, zamiast
pojedynczego obrazu, jest korzystne w klasyfikacji pokrycia terenu. Pozwala to na
rozróżnianie klas na podstawie ich różnych profili czasowych. (...) [1, 4]

ŹRÓDŁA:
   [1] arxiv_1704.04055v1.pdf str.1  sim=0.744
   (...)
```

### Pytanie spoza korpusu — system odmawia

```
PYTANIE: Jaka jest stolica Francji?

ODPOWIEDŹ: Na podstawie dostarczonych publikacji nie potrafię odpowiedzieć na to
pytanie.

ŹRÓDŁA:
   [1] arxiv_1704.04055v1.pdf str.1  sim=0.453   ← wyraźnie niższe podobieństwo
   (...)
```

**Ocena jakości.** Dla pytań w temacie odpowiedzi są poprawne, zwięzłe i — co
najważniejsze — **mają pokrycie w cytowanych źródłach** (model wskazuje numery
fragmentów, a ja widzę plik i stronę). Działa też wyszukiwanie **międzyjęzyczne**:
polskie pytanie trafnie dobiera angielskie fragmenty. Najlepiej system wypada w
teście negatywnym — na pytanie spoza korpusu **nie zmyśla**, tylko odmawia, a
spadek podobieństwa z ~0.75 do ~0.45 dobrze pokazuje, że to embeddingi „wyczuwają"
brak dopasowania. Słabszą stroną bywa to, że przy ogólnie sformułowanym pytaniu
wszystkie najlepsze fragmenty mogą pochodzić z jednej pracy — wtedy odpowiedź jest
poprawna, ale jednostronna; pomogłoby tu wymuszenie różnorodności źródeł.

---

## Podsumowanie i możliwy rozwój

System realizuje pełny potok RAG: od surowych PDF-ów, przez własne dzielenie na
fragmenty, embeddingi i indeks FAISS, aż po generowanie odpowiedzi opartej o
znaleziony kontekst, z podaniem źródeł. Naturalnym kolejnym krokiem byłaby
**ewaluacja jakości** odpowiedzi — np. metryką *faithfulness* (czy odpowiedź wynika
z kontekstu) i *answer relevance* (czy odpowiada na pytanie), liczonymi automatycznie
metodą LLM-as-a-Judge. Po stronie retrievalu wartościowe byłoby dołożenie
**re-rankera** oraz wymuszenie różnorodności źródeł, żeby odpowiedzi nie opierały
się na jednej publikacji.
