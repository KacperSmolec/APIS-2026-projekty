# Dotrenowanie Stable Diffusion (LoRA) — rasy psów

Pipeline dotrenowania modelu **Stable Diffusion v1.5** metodą **LoRA** do
generowania obrazów wybranych ras psów, na podstawie zbioru **Stanford Dogs**.
Projekt z przedmiotu APIS (Geoinformatyka II st.).

## Zakres

Repozytorium zawiera **opis metodyki** (sprawozdanie) oraz **gotowy do
uruchomienia na Google Colab notebook** realizujący cały potok: pobranie danych,
przygotowanie 5 ras, trening LoRA i porównanie generacji przed/po.

> Trening modelu dyfuzyjnego wymaga karty **GPU** — nie da się go sensownie
> policzyć na CPU. Dlatego notebook przeznaczony jest na Colab (T4 GPU), a samo
> sprawozdanie opisuje proces, decyzje i oczekiwane wyniki.

## Wybrane rasy

beagle, golden retriever, german shepherd (owczarek niemiecki), siberian husky,
boxer (bokser). Każdy obraz jest parowany z promptem o spójnej strukturze:
`photo of <rasa> dog`.

## Struktura

```
sd_dogi_lora/
├── rdzen/
│   └── przygotowanie_danych.py   # wybór 5 ras + metadata.jsonl
├── notebooki/
│   └── trening_lora_colab.ipynb  # pełny potok (Colab/GPU)
├── requirements.txt
├── README.md
└── SPRAWOZDANIE.md
```

## Jak uruchomić (Colab)

1. Otwórz `notebooki/trening_lora_colab.ipynb` w Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Uruchom komórki po kolei: instalacja → pobranie Stanford Dogs (wymaga
   `kaggle.json`) → przygotowanie danych → generacja **przed** → trening LoRA →
   generacja **po** → zestawienie porównawcze.

## Metoda

Wybrano **LoRA** (Low-Rank Adaptation): zamrażamy oryginalne wagi modelu i uczymy
tylko małe macierze niskiego rzędu wstrzykiwane w warstwy atencji U-Net. Pozwala to
dotrenować model na słabszym GPU (mały narzut pamięci, lekkie pliki wynikowe),
zachowując ogólną wiedzę modelu bazowego.

Szczegóły, porównanie z DreamBooth oraz analiza — w `SPRAWOZDANIE.md`.
