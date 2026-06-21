# Agent newsowy (Gemini function calling)

Autonomiczny agent AI, który na zadany temat samodzielnie wyszukuje wiadomości,
tworzy podsumowanie, ocenia istotność informacji i zapisuje raport PDF. Zbudowany
**bez frameworka LangChain** — na natywnym mechanizmie *function calling* modelu
**Gemini**, z własną pętlą decyzyjną.

## Co obejmuje

| Element | Realizacja |
|---|---|
| Narzędzie: wyszukiwanie | `szukaj_wiadomosci` — NewsAPI (+ fallback Google News RSS bez klucza) |
| Narzędzie: podsumowanie | `podsumuj_wiadomosci` — synteza artykułów w Markdown (LLM) |
| Narzędzie: ocena istotności | `ocen_istotnosc` — poziom wysoka/srednia/niska wg kryteriów |
| Narzędzie: raport PDF | `zapisz_raport_pdf` — Markdown → HTML → PDF, nazwa `<data>_<temat>_<istotność>.pdf` |
| Pętla agenta | własna pętla function-calling (Thought → Action → Observation) |
| Prompt systemowy | cel + zasady + samokorekta, bez sztywnego algorytmu kroków |
| Samodzielne decyzje | model sam wybiera narzędzia i ich kolejność |

## Architektura

```
Użytkownik: "raport o temacie X"
        │
        ▼
   AgentNewsowy  ──(pętla)──►  Gemini decyduje: które narzędzie?
        │                              │
        │   ◄── obserwacja (wynik) ────┘
        ▼
  Narzędzia (sam wybór kolejności):
   szukaj_wiadomosci ─► podsumuj_wiadomosci ─► ocen_istotnosc ─► zapisz_raport_pdf
   (NewsAPI/RSS)        (LLM, Markdown)        (LLM, kryteria)   (Markdown→HTML→PDF)
                                                                  │
                                                                  ▼
                                              raporty/<data>_<temat>_<istotność>.pdf
```

## Struktura

```
agent_newsowy/
├── rdzen/
│   ├── agent.py         # AgentNewsowy — pętla function-calling
│   ├── narzedzia.py     # 4 narzędzia (typowane, z docstringami)
│   ├── prompty.py       # prompt systemowy agenta
│   ├── llm.py           # wywołania modelu z ponawianiem (429)
│   └── konfiguracja.py
├── raporty/             # wygenerowane PDF-y (tworzone w runtime)
├── notebooki/demo_agenta.ipynb
├── testy/test_narzedzia.py   # testy offline
├── uruchom.py           # CLI
└── requirements.txt
```

## Uruchomienie

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="twoj_klucz"      # wymagany (lub api_key.txt / .env)
export NEWSAPI_KEY="twoj_klucz"         # opcjonalny; bez niego fallback RSS

python uruchom.py "energetyka jądrowa w Polsce"

# Testy (offline)
python -m pytest testy/ -v
```

## Uwaga o limitach API

Agent wykonuje kilka wywołań modelu na sesję (pętla decyzyjna + narzędzia LLM).
Darmowy plan Gemini ma dzienny i minutowy limit zapytań — kod ma wbudowane
ponawianie po błędzie 429. Najlepiej działa z modelem `gemini-2.5-flash`
(pełny model dobrze radzi sobie z wieloetapowym planowaniem).

Szczegóły decyzji projektowych — w `SPRAWOZDANIE.md`.
