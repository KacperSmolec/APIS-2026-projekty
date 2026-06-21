# GeoRAG-FAISS — system RAG dla teledetekcji

System Retrieval-Augmented Generation, który odpowiada na pytania na podstawie
korpusu publikacji naukowych (PDF) z zakresu **klasyfikacji pokrycia terenu z
obrazów satelitarnych**. Zbudowany od podstaw, bez frameworka LangChain —
z bezpośrednim użyciem `pypdf`, **FAISS** oraz modeli **Gemini** (embeddingi +
generacja).

## Co obejmuje

| Element | Realizacja |
|---|---|
| Wczytywanie PDF | `pypdf` bezpośrednio, tekst per strona z metadanymi |
| Chunking | **własny dzielnik zdaniowy** ze zakładką (overlap) |
| Embeddingi | `gemini-embedding-001` (768D, tryby document/query) |
| Vector store | **FAISS** (`IndexFlatIP` + wektory znormalizowane = kosinus) |
| Retriever | wyszukiwanie k najbliższych fragmentów |
| Generacja z kontekstem | Gemini 2.5 Flash, prompt oparty wyłącznie o znalezione fragmenty |
| System jako klasa | `SilnikRAG` (pakiet `rdzen`) |
| Trwałość | indeks + metadane zapisywane na dysk i wczytywane |

## Architektura

```
Budowa indeksu:
  PDF ─► WczytywaczPDF ─► DzielnikTekstu ─► ModelEmbeddingow ─► BazaWektorowaFAISS ─► dysk
        (pypdf)          (zdaniowy)        (Gemini, DOCUMENT)   (IndexFlatIP)

Zapytanie:
  pytanie ─► ModelEmbeddingow ─► BazaWektorowaFAISS ─► kontekst ─► Gemini ─► odpowiedź + źródła
            (Gemini, QUERY)      (top-k kosinus)                  (LLM)
```

## Struktura

```
georag_faiss/
├── rdzen/
│   ├── wczytywanie.py     # WczytywaczPDF (pypdf + czyszczenie tekstu)
│   ├── dzielenie.py       # DzielnikTekstu (własny chunking zdaniowy)
│   ├── embeddingi.py      # ModelEmbeddingow (Gemini, rate limiting)
│   ├── baza_wektorowa.py  # BazaWektorowaFAISS (build/save/load/search)
│   ├── silnik_rag.py      # SilnikRAG (orkiestracja + generacja)
│   ├── konfiguracja.py    # Ustawienia
│   └── wyjatki.py
├── dane/                  # publikacje PDF (korpus)
├── indeks/                # zapisany indeks FAISS (generowany)
├── notebooki/demo_rag.ipynb
├── testy/test_rag.py      # testy offline (bez sieci)
├── zbuduj_indeks.py       # CLI: budowa bazy
├── zapytaj.py             # CLI: odpytywanie
└── requirements.txt
```

## Uruchomienie

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="twoj_klucz"     # lub plik api_key.txt / .env

# 1. Zbuduj bazę wektorową z PDF-ów w katalogu dane/
python zbuduj_indeks.py

# 2. Zadawaj pytania
python zapytaj.py

# Testy (offline)
python -m pytest testy/ -v
```

## Korpus

Cztery publikacje z arXiv dotyczące klasyfikacji pokrycia terenu i teledetekcji
(segmentacja semantyczna, sieci rekurencyjne dla danych wieloczasowych, zbiór
PatternNet, nadzór długozasięgowy). Pliki w `dane/`.

Szczegóły decyzji projektowych — w `SPRAWOZDANIE.md`.
