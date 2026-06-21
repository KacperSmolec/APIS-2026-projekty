# APIS 2026 — Projekty (Geoinformatyka II st.)

Zbiór czterech projektów z przedmiotu *Aktualne Problemy Informatyki Stosowanej*.
Każdy projekt to osobne, samodzielne repo-podkatalog z własnym kodem, testami,
notebookiem i sprawozdaniem (`SPRAWOZDANIE.md`).

| # | Projekt | Katalog | Stos technologiczny |
|---|---------|---------|---------------------|
| 3 | Production-ready chatbot | [`zadanie 3/chatbot_gemini`](zadanie%203/chatbot_gemini) | Gemini 2.5 Flash, kontrola kontekstu w tokenach, logowanie |
| 4 | System RAG | [`zadanie 4/georag_faiss`](zadanie%204/georag_faiss) | pypdf, własny chunking, embeddingi Gemini, FAISS |
| 5 | Autonomiczny agent AI | [`zadanie 5/agent_newsowy`](zadanie%205/agent_newsowy) | function calling Gemini, NewsAPI, raport PDF |
| 6 | Stable Diffusion (LoRA) | [`zadanie 6/sd_dogi_lora`](zadanie%206/sd_dogi_lora) | diffusers, LoRA, notebook Colab |

## Uruchamianie

Każdy projekt ma własny `README.md` i `requirements.txt`. Wymagany jest klucz
Google Gemini (zmienna `GEMINI_API_KEY` lub plik `api_key.txt`); zadanie 5
opcjonalnie korzysta z klucza NewsAPI. Pliki z kluczami są wykluczone z repozytorium.

## Uwagi

- Projekty 3–5 były uruchamiane i weryfikowane na żywo (testy jednostkowe + realne
  wywołania modelu). Projekt 6 wymaga GPU — dostarczony jest jako notebook gotowy do
  uruchomienia na Google Colab wraz z opisowym sprawozdaniem.
- Duże/generowane artefakty (indeksy, raporty, dane, logi) są wykluczone z repo i
  odtwarzane przy uruchomieniu.
