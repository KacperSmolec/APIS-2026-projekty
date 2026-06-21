# GeoBot — Production Ready Chatbot (Gemini)

Chatbot oparty o model **Google Gemini 2.5 Flash**, pełniący rolę asystenta
geoinformatycznego (projekt z przedmiotu APIS — Geoinformatyka II st.).

## Co obejmuje

| Element | Realizacja |
|---|---|
| Integracja z modelem (API) | Gemini 2.5 Flash przez SDK `google-genai` |
| Wysyłanie zapytań i generowanie odpowiedzi | `ChatbotGemini.odpowiedz()` |
| Prompt systemowy | Zdefiniowany w `konfiguracja.py` (asystent geoinformatyczny) |
| Historia rozmowy | Klasa `HistoriaRozmowy` |
| Chatbot jako klasa | `ChatbotGemini` (pakiet `rdzen`) |
| Parametry generacji | temperatura, top_p, top_k, max_output_tokens |
| Kontrola długości kontekstu | **liczona w tokenach**, nie w liczbie wiadomości |
| Obsługa błędów | Własne wyjątki + ponawianie + wycofywanie historii |
| Logowanie zapytań i błędów | Osobne pliki `logi/zapytania.log` i `logi/bledy.log` |

## Architektura

```
Użytkownik (CLI / notebook)
        │  tekst
        ▼
  ChatbotGemini  ── prompt systemowy
        │
        ├─ HistoriaRozmowy ── kontrola kontekstu (tokeny)
        ├─ logowanie ──────── zapytania.log / bledy.log
        └─ wywołanie modelu ─ google-genai → Gemini 2.5 Flash
                              (ponawianie + mapowanie błędów)
```

## Struktura

```
chatbot_gemini/
├── rdzen/
│   ├── chatbot.py        # klasa ChatbotGemini (logika główna)
│   ├── historia.py       # historia + kontrola kontekstu w tokenach
│   ├── konfiguracja.py   # ustawienia i prompt systemowy
│   ├── logowanie.py      # konfiguracja loggerów
│   └── wyjatki.py        # własna hierarchia wyjątków
├── notebooki/
│   └── demo_chatbota.ipynb
├── testy/
│   └── test_chatbot.py   # testy offline (atrapy, bez sieci)
├── demo_cli.py           # interaktywne demo w terminalu
├── requirements.txt
└── .env.example
```

## Uruchomienie

```bash
pip install -r requirements.txt

# Klucz API: zmienna środowiskowa albo plik api_key.txt w katalogu repo
export GEMINI_API_KEY="twoj_klucz"      # lub: cp .env.example .env

# Interaktywne demo
python demo_cli.py

# Testy (offline)
python -m pytest testy/ -v
```

## Wybrane decyzje projektowe

* **Gemini zamiast modelu lokalnego** — brak konieczności posiadania mocnego GPU,
  szybki start, dobra jakość języka polskiego.
* **Kontekst liczony w tokenach** — dokładniejsze niż okno liczone w wiadomościach;
  jedna wiadomość może mieć 5 lub 500 tokenów.
* **Wyłączone „myślenie" modelu** (`thinking_budget=0`) — dla chatbota
  konwersacyjnego daje przewidywalny koszt i krótszą latencję.
* **Wstrzykiwany klient** — pozwala testować logikę bez sieci (atrapy).

Szczegóły w `SPRAWOZDANIE.md`.
