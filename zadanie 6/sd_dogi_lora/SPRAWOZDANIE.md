# Sprawozdanie — Zadanie 6: Generator obrazów (Stable Diffusion)

**Przedmiot:** Aktualne Problemy Informatyki Stosowanej — Geoinformatyka II st.
**Temat:** dotrenowanie Stable Diffusion do generowania wybranych ras psów

---

## Wprowadzenie

Tematem zadania jest dotrenowanie modelu **Stable Diffusion** tak, by lepiej
generował obrazy konkretnych ras psów. Model bazowy potrafi narysować „psa w
ogóle", ale ma kłopot z wiernym oddaniem cech charakterystycznych poszczególnych
ras. Dotrenowanie na dedykowanym zbiorze ma tę zdolność poprawić.

Wybrałem pięć wyraźnie różniących się ras ze zbioru **Stanford Dogs**: beagle,
golden retriever, owczarka niemieckiego (german shepherd), husky syberyjskiego i
boksera (boxer). Jako metodę dotrenowania wybrałem **LoRA**, bo jest oszczędna
pamięciowo i da się ją uruchomić na zwykłym GPU (np. darmowy Colab T4).

W tym sprawozdaniu opisuję cały proces: jak zbudowany jest potok treningowy, jak
działają modele dyfuzyjne i sam Stable Diffusion, na czym polega transfer learning i
LoRA, jak LoRA wypada przy DreamBooth oraz jakie hiperparametry dobrałbym i dlaczego.

**Uwaga o wykonaniu.** Trening modelu dyfuzyjnego wymaga karty GPU — na komputerze,
na którym przygotowywałem projekt (Mac, bez CUDA), nie da się go sensownie policzyć.
Dlatego część praktyczną przygotowałem jako **gotowy do uruchomienia notebook na
Google Colab** (`notebooki/trening_lora_colab.ipynb`), który wykonuje cały potok i
zapisuje obrazy przed/po treningu. Niniejsze sprawozdanie skupia się na opisie
metody, podjętych decyzjach i oczekiwanych wynikach.

---

## 1. Architektura i pipeline treningowy

Potok treningowy przekształca surowe pary obraz–tekst w zaktualizowane wagi modelu.
Przepływ danych wygląda tak:

```
Stanford Dogs (PDF/obrazy ras)
        │  wybór 5 ras
        ▼
przygotowanie danych ──► pary (obraz, prompt "photo of <rasa> dog") + metadata.jsonl
        │
        ▼
trening LoRA (diffusers + accelerate)
   obraz ─► VAE ─► latent ─► +szum (dyfuzja w przód)
   prompt ─► CLIP ─► embedding tekstu
   U-Net (z warstwami LoRA) przewiduje szum  ──► strata MSE ──► aktualizacja wag LoRA
        │
        ▼
wagi LoRA  ──► doładowane do modelu bazowego przy generacji
```

**Przygotowanie promptów.** Każdy obraz dostaje prompt o **spójnej, sztywnej
strukturze** `photo of <rasa> dog` (np. „photo of beagle dog"). Spójność jest tu
celowa: model uczy się wiązać konkretną nazwę rasy z jej cechami wizualnymi, więc
zależy nam, by ten sam koncept był zawsze opisywany tak samo. Pary zapisuję do pliku
`metadata.jsonl` w formacie wymaganym przez bibliotekę Hugging Face Diffusers (kod w
`rdzen/przygotowanie_danych.py`).

**Sposób trenowania.** Korzystam z gotowego skryptu treningowego z biblioteki
`diffusers` (`train_text_to_image_lora.py`), uruchamianego przez **Hugging Face
Accelerate** (`accelerate launch`). To standardowe, „produkcyjne" podejście —
oddziela konfigurację sprzętu od samej logiki treningu, więc ten sam kod działa na
różnych GPU. W trakcie treningu obraz jest kompresowany przez VAE do przestrzeni
latentnej, dokładany jest do niego losowy szum (proces dyfuzji w przód), a U-Net —
na podstawie embeddingu tekstu — uczy się ten szum przewidzieć. Różnica między
przewidzianym a prawdziwym szumem (strata **MSE**) napędza aktualizację wag.

---

## 2. Modele generatywne

**Czym są modele dyfuzyjne i jak działają.** Modele dyfuzyjne (jak Stable Diffusion)
uczą się generować obrazy przez **stopniowe usuwanie szumu**. Najpierw, w procesie
„w przód", do prawdziwego obrazu dokładany jest krok po kroku losowy szum, aż zostaje
z niego czysty szum gaussowski. Sieć neuronowa uczy się odwracać ten proces:
przewidywać, jaki szum został dodany, żeby móc go odjąć. Generacja to uruchomienie
tego odwrotnego procesu od czystego szumu — sieć iteracyjnie „odszumia" losowy obraz,
aż powstaje sensowna grafika zgodna z promptem.

**Inne podejścia do generowania obrazów:**
- **GAN (Generative Adversarial Networks):** dwie sieci rywalizują — generator tworzy
  obrazy, a dyskryminator ocenia, czy są prawdziwe. Generują szybko (jeden przebieg),
  ale trening bywa niestabilny i podatny na *mode collapse* (model produkuje mało
  zróżnicowane wyniki). Modele dyfuzyjne są wolniejsze (wiele kroków odszumiania), ale
  stabilniejsze i dają zwykle wyższą jakość oraz różnorodność.
- **VAE (Variational Autoencoders):** kompresują obraz do ciągłej przestrzeni latentnej
  i odtwarzają go z niej. Są matematycznie stabilne, ale obrazy bywają rozmyte
  („uśrednione"). Co ciekawe, VAE nie jest tu konkurencją „albo–albo": Stable Diffusion
  używa VAE jako jednego ze swoich komponentów (do kompresji do przestrzeni latentnej).

---

## 3. Architektura Stable Diffusion

Stable Diffusion to **latentny model dyfuzyjny (LDM)** — kluczowe jest to, że nie
pracuje na pikselach, lecz w skompresowanej **przestrzeni latentnej**, co ogromnie
zmniejsza koszt obliczeń. Składa się z trzech głównych części:

- **VAE (enkoder + dekoder):** enkoder kompresuje obraz z przestrzeni pikseli do
  mniejszej przestrzeni latentnej (redukcja wymiarów przestrzennych ~8-krotnie),
  a dekoder na końcu zamienia latent z powrotem w obraz.
- **Enkoder tekstu CLIP:** zamienia prompt na wektory kontekstowe (embeddingi), które
  „mówią" modelowi, co ma narysować.
- **U-Net odszumiający:** sieć w kształcie litery U z blokami ResNet i warstwami
  **cross-attention**. To właśnie cross-attention wplata kontekst tekstowy (z CLIP) w
  stan latentny obrazu, dzięki czemu generacja jest sterowana promptem. U-Net
  przewiduje szum do usunięcia w danym kroku.

Generacja: tekst → CLIP → embedding; z czystego szumu w przestrzeni latentnej U-Net
iteracyjnie odszumia (kierując się embeddingiem), a na końcu VAE-dekoder zamienia
gotowy latent w obraz.

---

## 4. Transfer learning i zamrażanie warstw

**Transfer learning** polega na wykorzystaniu wiedzy, którą model zdobył na ogromnym,
ogólnym zbiorze (Stable Diffusion trenowano na miliardach obrazów z LAION), i
dostosowaniu go do węższego zadania (tu: konkretne rasy psów). Nie uczymy modelu od
zera — korzystamy z tego, że „umie" już rysować kształty, światło, tekstury, i tylko
dostrajamy go do nowej specjalizacji.

**Zamrażanie warstw** to ustawienie części wag jako nietrenowalnych
(`requires_grad=False`) — podczas treningu te wagi się nie zmieniają. Daje to dwie
korzyści: chroni przed **katastrofalnym zapominaniem** (model nie traci ogólnej
wiedzy o komponowaniu obrazu) oraz mocno zmniejsza zużycie pamięci GPU (mniej
gradientów do policzenia i przechowania).

**Od czego zależy, które i ile warstw zamrozić.** Generalna zasada: wczesne warstwy
sieci wychwytują uniwersalne cechy (krawędzie, kolory, oświetlenie), a późniejsze —
bardziej abstrakcyjne, „dziedzinowe" koncepty. Im bardziej nowe zadanie przypomina
to oryginalne, tym więcej można zamrozić (uczymy tylko wierzch). Im mniej danych
mamy, tym więcej warto zamrozić, by uniknąć przeuczenia. W tym projekcie VAE i enkoder
CLIP są w całości zamrożone, a aktualizacje trafiają tylko do warstw atencji U-Net —
bo to one odpowiadają za powiązanie tekstu (nazwy rasy) z obrazem.

---

## 5. LoRA — wyjaśnienie własnymi słowami

**LoRA (Low-Rank Adaptation)** to sprytny sposób na dotrenowanie dużego modelu bez
ruszania jego oryginalnych wag. Zamiast modyfikować wielkie macierze wag (co jest
kosztowne pamięciowo), **zamrażamy oryginalne wagi** modelu, a obok wstrzykujemy
**małe, dodatkowe macierze niskiego rzędu** (oznaczane A i B), które jako jedyne się
uczą. Efekt działania to suma: `W_nowe = W_0 + ΔW`, gdzie `W_0` to zamrożona waga
bazowa, a `ΔW` to poprawka wyliczona przez LoRA.

Sztuczka polega na **niskim rzędzie** tych dodatkowych macierzy — dzięki temu liczba
trenowanych parametrów spada o ponad 90–98%. To właśnie sprawia, że LoRA mieści się w
pamięci zwykłego GPU.

**Różnica względem klasycznego fine-tuningu.** W klasycznym fine-tuningu aktualizujemy
(potencjalnie) wszystkie wagi modelu — to wymaga dużo pamięci i mocy, a wynik to ciężki
plik (cały model). W LoRA oryginalny model zostaje nietknięty, uczą się tylko małe
„nakładki". Plusy: mało pamięci, szybko, lekkie pliki wynikowe (~kilka–kilkanaście MB),
można mieć wiele LoRA do jednego modelu bazowego. Minus: przy bardzo złożonych zmianach
LoRA może mieć nieco mniejszą „pojemność" niż pełny fine-tuning.

---

## 6. LoRA vs DreamBooth

**W projekcie zastosowałem LoRA.** Dla porównania opisuję obie metody:

- **LoRA** wstrzykuje małe macierze niskiego rzędu do warstw atencji. Jest oszczędna
  (mieści się na 6–8 GB VRAM), szybka i daje lekkie pliki. Świetna do szybkiego
  dostrajania stylu/konceptów i do pracy na słabszym sprzęcie.
- **DreamBooth** dotrenowuje cały (lub dużą część) U-Net i wiąże nowy koncept z rzadkim
  tokenem w promptcie (np. „sks dog"). Daje maksymalną wierność konkretnego obiektu, ale
  wymaga dużo VRAM (>12 GB) i tworzy ciężkie pliki (~GB).

**Kiedy co stosować.** LoRA — gdy zależy nam na efektywności, pracy na konsumenckim GPU,
wielu lekkich adaptacjach albo szybkiej iteracji (jak tutaj: 5 ras, ograniczony sprzęt).
DreamBooth — gdy potrzebujemy maksymalnej wierności jednego, konkretnego obiektu (np.
„ten jeden pies") i mamy mocny sprzęt oraz miejsce na duże checkpointy.

---

## 7. Trening LoRA: wiele klas vs osobne modele per klasa

Istnieją dwa podejścia do dotrenowania pięciu ras za pomocą LoRA:

- **Jedna LoRA na wszystkich 5 rasach naraz (wspólny model).** Zaleta: oszczędność —
  jeden lekki plik, krótszy łączny czas treningu, wygodne zarządzanie. Wada: ryzyko
  **„przeciekania" cech** między rasami (cross-contamination) — model dzieli te same
  macierze dla wszystkich ras, więc cechy jednej mogą wpływać na generację innej.
- **Osobne modele LoRA dla każdej rasy (5 modeli).** Zaleta: pełna **izolacja klas** —
  cała pojemność modelu skupia się na jednej rasie, bez przecieków, zwykle najwyższa
  wierność. Wady: więcej plików do utrzymania, dłuższy łączny czas treningu, brak
  możliwości łatwego mieszania ras w jednym przebiegu.

W dołączonym notebooku realizuję wariant **jednej wspólnej LoRA na 5 rasach** (prostszy
i szybszy). Wariant izolowany sprowadzałby się do uruchomienia tego samego treningu
osobno na podzbiorach danych per rasa.

---

## 8. Hiperparametry

Dobór hiperparametrów (w notebooku) jest podporządkowany temu, by trening zmieścił się
na darmowym GPU i był stabilny:

| Hiperparametr | Wartość | Dlaczego |
|---|---|---|
| rozdzielczość | 512×512 | natywna rozdzielczość SD v1.5 — zgodność z modelem bazowym |
| `train_batch_size` | 1 | minimalny, by nie przepełnić pamięci GPU (uniknięcie CUDA OOM) |
| `gradient_accumulation_steps` | 4 | symuluje efektywny batch = 4 → stabilniejsze gradienty bez wzrostu VRAM |
| `mixed_precision` | fp16 | ~2× mniejsze zużycie pamięci, szybszy trening |
| `gradient_checkpointing` | wł. | zwalnia pamięć kosztem nieco wolniejszego backpropu |
| `max_train_steps` | 1000 | kompromis: dość, by model nauczył się ras, krótko, by nie przeuczyć |
| `learning_rate` (LoRA) | 1e-4 | typowa wartość dla LoRA (wyższa niż w pełnym fine-tuningu, bo uczymy mało parametrów) |
| `lr_scheduler` | constant | prosty, przewidywalny przy krótkim treningu |
| `seed` | 42 | powtarzalność wyników |

**Jak wpływają na trening i jakość.** *Learning rate* jest najczulszy: za wysoki →
trening „rozjeżdża się" i pojawiają się artefakty; za niski → model uczy się zbyt wolno.
*Liczba kroków* decyduje o tym, ile model się nauczy — za mało to słabe odwzorowanie
rasy, za dużo to przeuczenie (model zaczyna „uczyć się obrazów na pamięć" i traci
różnorodność). *Batch + akumulacja gradientu* stabilizują uczenie. *fp16* i *gradient
checkpointing* nie zmieniają jakości docelowej, lecz umożliwiają w ogóle zmieszczenie
treningu w pamięci. Dla DreamBooth użyłoby się znacznie niższego LR (np. 2e-6), bo
aktualizuje się tam dużo więcej parametrów.

---

## 9. Wyniki i analiza (przed vs po)

Część praktyczna (notebook) generuje obrazy każdej z 5 ras **przed** treningiem
(model bazowy SD v1.5) i **po** doładowaniu wytrenowanej LoRA, a następnie zestawia je
obok siebie. Ponieważ treningu nie wykonywałem na maszynie bez GPU, poniżej opisuję,
czego należy się spodziewać i jak to oceniać (rzeczywiste obrazy powstają po
uruchomieniu notebooka na Colab i trafiają do katalogów `wyniki/przed` i `wyniki/po`).

**Oczekiwany efekt.** Model bazowy generuje psy „poprawne, ale generyczne" — często
myli cechy ras albo uśrednia je do typowego psa. Po dotrenowaniu LoRA generacje
powinny wyraźniej oddawać cechy charakterystyczne: proporcje sylwetki, kształt uszu i
pyska, typ i kolor sierści. Przykładowo dla huskyego — gęste, dwuwarstwowe futro i
charakterystyczne umaszczenie; dla owczarka niemieckiego — sylwetka i umaszczenie
siodłowe; dla beagle'a — proporcje i ubarwienie tricolor.

**Na co zwracać uwagę przy ocenie** (zgodnie z poleceniem):
- **jakość wizualna** — ostrość, realizm, brak zniekształceń;
- **zgodność z promptem** — czy wygenerowany pies to faktycznie zadana rasa;
- **różnorodność w obrębie rasy** — czy model potrafi pokazać różne ujęcia/tła, czy
  „zaciął się" na jednym wyglądzie;
- **artefakty** — zniekształcone łapy, oczy, dodatkowe kończyny;
- **uczenie na pamięć** — czy wyniki nie są zbyt podobne do konkretnych zdjęć
  treningowych (oznaka przeuczenia).

**Spodziewany kompromis.** Przy zbyt długim treningu lub zbyt małej liczbie zdjęć na
rasę model może zacząć odtwarzać konkretne obrazy treningowe (spadek różnorodności).
Przy wariancie jednej wspólnej LoRA możliwe są drobne „przecieki" cech między rasami;
podejście z osobnymi modelami per rasa dałoby czystszą izolację kosztem dłuższego
treningu i większej liczby plików.

---

## Podsumowanie

Sprawozdanie opisuje kompletny proces dotrenowania Stable Diffusion metodą LoRA do
generowania pięciu ras psów: od przygotowania par obraz–prompt, przez działanie
modeli dyfuzyjnych i architekturę SD, transfer learning i LoRA, po porównanie z
DreamBooth, warianty treningu i dobór hiperparametrów. Część praktyczną dostarczam
jako gotowy notebook na Google Colab (trening wymaga GPU), który realizuje cały potok
i pozwala porównać generację przed i po dotrenowaniu.
