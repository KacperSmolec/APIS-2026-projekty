"""Narzędzia (tools) agenta newsowego.

Każde narzędzie to zwykła funkcja Pythona z **typowanymi parametrami** i
**docstringiem**. To na ich podstawie model językowy (Gemini) buduje schemat
narzędzia i decyduje, kiedy oraz z jakimi argumentami je wywołać — dlatego opisy
są konkretne i mówią wprost, kiedy danego narzędzia użyć i co ono zwraca.

Narzędzia korzystają ze wspólnego kontekstu (klient Gemini + ustawienia), który
ustawiany jest raz przez agenta funkcją ``ustaw_kontekst``. Dzięki temu same
sygnatury funkcji pozostają czyste (tylko argumenty istotne dla modelu).
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import markdown
from google.genai import types
from xhtml2pdf import pisa

from .llm import generuj, myslenie

logger = logging.getLogger("agent.narzedzia")

# --- współdzielony kontekst, wstrzykiwany przez agenta ---
_KLIENT = None
_UST = None


def ustaw_kontekst(klient, ustawienia) -> None:
    """Ustawia wspólny kontekst narzędzi (klient Gemini + ustawienia)."""
    global _KLIENT, _UST
    _KLIENT = klient
    _UST = ustawienia


# ====================================================================== #
#  NARZĘDZIE 1: wyszukiwanie wiadomości                                   #
# ====================================================================== #
def szukaj_wiadomosci(temat: str) -> str:
    """Wyszukuje najnowsze artykuły prasowe na zadany temat.

    Użyj tego narzędzia jako PIERWSZEGO kroku, aby zebrać świeże informacje o
    wybranym zagadnieniu. Korzysta z serwisu NewsAPI, a gdy brak klucza API —
    z darmowego kanału Google News RSS.

    Args:
        temat: Słowa kluczowe lub fraza do wyszukania (np. "sztuczna inteligencja").

    Returns:
        Tekst JSON z listą artykułów; każdy artykuł ma pola: tytul, opis, url,
        data. W razie problemu zwraca JSON z polem "blad".
    """
    logger.info("[narzędzie] szukaj_wiadomosci(temat=%r)", temat)
    liczba = _UST.liczba_artykulow if _UST else 5

    if _UST and _UST.klucz_newsapi:
        artykuly = _newsapi(temat, liczba, _UST.klucz_newsapi)
        if artykuly is not None:
            return json.dumps(artykuly, ensure_ascii=False)
        logger.warning("NewsAPI nie powiodło się — przełączam na fallback RSS.")

    artykuly = _google_news_rss(temat, liczba)
    return json.dumps(artykuly, ensure_ascii=False)


def _newsapi(temat: str, liczba: int, klucz: str) -> Optional[list]:
    url = "https://newsapi.org/v2/everything?" + "&".join([
        f"q={quote_plus(temat)}", f"pageSize={liczba}",
        "sortBy=publishedAt", f"apiKey={klucz}",
    ])
    try:
        with urlopen(Request(url, headers={"User-Agent": "agent-newsowy"}), timeout=20) as r:
            dane = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        logger.warning("Błąd NewsAPI: %s", e)
        return None
    if dane.get("status") != "ok":
        logger.warning("NewsAPI status: %s", dane.get("message"))
        return None
    return [
        {
            "tytul": (a.get("title") or "")[:160],
            "opis": (a.get("description") or "")[:300],
            "url": a.get("url") or "",
            "data": a.get("publishedAt") or "",
        }
        for a in dane.get("articles", [])[:liczba]
    ]


def _google_news_rss(temat: str, liczba: int) -> list:
    """Fallback bez klucza: kanał RSS wyszukiwarki Google News."""
    url = f"https://news.google.com/rss/search?q={quote_plus(temat)}&hl=pl&gl=PL&ceid=PL:pl"
    try:
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as r:
            xml = r.read().decode("utf-8")
    except Exception as e:
        return [{"blad": f"Nie udało się pobrać wiadomości: {e}"}]

    korzen = ET.fromstring(xml)
    wynik = []
    for item in korzen.iter("item"):
        tytul = item.findtext("title") or ""
        link = item.findtext("link") or ""
        data = item.findtext("pubDate") or ""
        opis = re.sub(r"<[^>]+>", "", item.findtext("description") or "")[:300]
        wynik.append({"tytul": tytul[:160], "opis": opis, "url": link, "data": data})
        if len(wynik) >= liczba:
            break
    return wynik or [{"blad": "Brak wyników dla podanego tematu."}]


# ====================================================================== #
#  NARZĘDZIE 2: inteligentne podsumowanie                                 #
# ====================================================================== #
def podsumuj_wiadomosci(dane_wiadomosci: str, temat: str) -> str:
    """Tworzy spójne, eksperckie podsumowanie na podstawie zebranych artykułów.

    Użyj tego narzędzia PO zebraniu wiadomości. Nie zwraca listy artykułów, lecz
    syntetyzuje je w jedną, uporządkowaną notatkę w formacie Markdown (nagłówki,
    punkty). Łączy powtarzające się informacje w spójną narrację.

    Args:
        dane_wiadomosci: Tekst (np. JSON) z tytułami i opisami zebranych artykułów.
        temat: Temat, którego dotyczy podsumowanie.

    Returns:
        Podsumowanie w formacie Markdown albo komunikat o błędzie.
    """
    logger.info("[narzędzie] podsumuj_wiadomosci(temat=%r, dł.=%d)", temat, len(dane_wiadomosci))
    if not dane_wiadomosci or len(dane_wiadomosci.strip()) < 10:
        return "Błąd: zbyt mało danych do podsumowania. Najpierw wyszukaj wiadomości."

    instrukcja = (
        f"Jesteś analitykiem prasowym. Na podstawie poniższych artykułów o temacie "
        f"'{temat}' napisz zwięzłe, uporządkowane podsumowanie najważniejszych "
        f"wydarzeń w formacie Markdown. Wymagania:\n"
        f"- zacznij od nagłówka '## Najważniejsze wydarzenia',\n"
        f"- pogrupuj informacje tematycznie, użyj wypunktowań,\n"
        f"- łącz powtarzające się wątki, nie wymyślaj faktów spoza danych,\n"
        f"- pisz po polsku, rzeczowo.\n\n"
        f"DANE ŹRÓDŁOWE:\n{dane_wiadomosci}"
    )
    try:
        odp = generuj(
            _KLIENT, model=_UST.model_llm, contents=instrukcja,
            config=types.GenerateContentConfig(
                temperature=_UST.temperatura,
                max_output_tokens=1200,
                thinking_config=myslenie(_UST.model_llm),
            ),
        )
        return (odp.text or "").strip() or "Błąd: model zwrócił puste podsumowanie."
    except Exception as e:
        return f"Błąd podczas tworzenia podsumowania: {e}"


# ====================================================================== #
#  NARZĘDZIE 3: ocena istotności                                         #
# ====================================================================== #
def ocen_istotnosc(podsumowanie: str, temat: str) -> str:
    """Ocenia istotność zebranych informacji dla danej dziedziny.

    Użyj tego narzędzia PO przygotowaniu podsumowania, aby przypisać poziom
    istotności. Kryteria: czy wydarzenia są przełomowe/pilne, jak bardzo
    dotyczą bezpośrednio tematu, czy są świeże i konkretne.

    Args:
        podsumowanie: Wcześniej przygotowane podsumowanie (Markdown).
        temat: Dziedzina/temat, względem którego oceniamy istotność.

    Returns:
        Dokładnie jedno słowo: 'wysoka', 'srednia' albo 'niska'.
    """
    logger.info("[narzędzie] ocen_istotnosc(temat=%r)", temat)
    instrukcja = (
        "Oceń istotność poniższego podsumowania dla dziedziny "
        f"'{temat}'. Zastosuj kryteria:\n"
        "- 'wysoka': przełomowe, pilne wydarzenia o dużym wpływie na dziedzinę;\n"
        "- 'srednia': istotne aktualizacje, ale bez przełomu;\n"
        "- 'niska': informacje poboczne, marketingowe lub luźno związane.\n"
        "Odpowiedz WYŁĄCZNIE jednym słowem: wysoka, srednia albo niska.\n\n"
        f"PODSUMOWANIE:\n{podsumowanie}"
    )
    try:
        odp = generuj(
            _KLIENT, model=_UST.model_llm, contents=instrukcja,
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=10,
                thinking_config=myslenie(_UST.model_llm),
            ),
        )
        slowo = (odp.text or "").strip().lower()
    except Exception as e:
        logger.warning("Błąd oceny istotności: %s — przyjmuję 'srednia'.", e)
        return "srednia"
    for poziom in ("wysoka", "srednia", "niska"):
        if poziom in slowo:
            return poziom
    return "srednia"


# ====================================================================== #
#  NARZĘDZIE 4: generowanie raportu PDF                                   #
# ====================================================================== #
def zapisz_raport_pdf(tresc_markdown: str, temat: str, istotnosc: str) -> str:
    """Zapisuje gotowe podsumowanie jako sformatowany raport PDF.

    Użyj tego narzędzia jako OSTATNIEGO kroku, gdy podsumowanie jest gotowe, a
    istotność oceniona. Konwertuje Markdown -> HTML -> PDF. Nazwa pliku ma format
    <data>_<temat>_<istotnosc>.pdf.

    Args:
        tresc_markdown: Pełna treść raportu w formacie Markdown.
        temat: Temat raportu (użyty w nazwie pliku i nagłówku).
        istotnosc: Poziom istotności: 'wysoka', 'srednia' albo 'niska'.

    Returns:
        Komunikat o sukcesie ze ścieżką do pliku albo komunikat o błędzie.
    """
    logger.info("[narzędzie] zapisz_raport_pdf(temat=%r, istotnosc=%r)", temat, istotnosc)
    try:
        bezpieczny_temat = re.sub(r"[^a-z0-9]+", "_", temat.lower()).strip("_")[:50] or "raport"
        data = datetime.now().strftime("%Y-%m-%d")
        nazwa = f"{data}_{bezpieczny_temat}_{istotnosc.lower()}.pdf"

        katalog = _UST.katalog_raportow if _UST else Path("raporty")
        katalog.mkdir(parents=True, exist_ok=True)
        sciezka = katalog / nazwa

        cialo_html = markdown.markdown(tresc_markdown, extensions=["extra", "sane_lists"])
        html = f"""
        <html><head><meta charset="utf-8"><style>
        body {{ font-family: Helvetica, Arial, sans-serif; font-size: 11pt; }}
        h1 {{ color: #1a3d6d; }} h2 {{ color: #2a5a8c; }}
        .naglowek {{ border-bottom: 2px solid #1a3d6d; margin-bottom: 12px; }}
        .meta {{ color: #555; font-size: 9pt; }}
        </style></head><body>
        <div class="naglowek">
          <h1>Raport prasowy: {temat.title()}</h1>
          <p class="meta">Data: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
             Oceniona istotność: <b>{istotnosc.upper()}</b></p>
        </div>
        {cialo_html}
        </body></html>
        """
        with open(sciezka, "wb") as f:
            wynik = pisa.CreatePDF(html, dest=f)
        if wynik.err:
            return f"Błąd: generowanie PDF nie powiodło się (kod {wynik.err})."
        return f"Sukces: raport zapisany w pliku {sciezka}"
    except Exception as e:
        return f"Błąd podczas generowania PDF: {e}"


# Lista narzędzi udostępnianych agentowi (kolejność dowolna — agent sam decyduje).
NARZEDZIA = [szukaj_wiadomosci, podsumuj_wiadomosci, ocen_istotnosc, zapisz_raport_pdf]
