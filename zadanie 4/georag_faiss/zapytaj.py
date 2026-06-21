"""Interaktywne odpytywanie systemu RAG.

Wymaga wcześniej zbudowanej bazy (`python zbuduj_indeks.py`).
Uruchomienie:
    python zapytaj.py
"""
from rdzen import SilnikRAG, BrakIndeksu


def main() -> None:
    silnik = SilnikRAG()
    try:
        silnik.wczytaj_indeks()
    except BrakIndeksu as e:
        print(f"[BŁĄD] {e}")
        return

    print("System RAG gotowy. Zadaj pytanie (puste = koniec).")
    while True:
        try:
            pytanie = input("\nPytanie: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not pytanie:
            break

        wynik = silnik.zapytaj(pytanie)
        print(f"\nOdpowiedź:\n{wynik.odpowiedz}\n")
        print("Źródła:")
        for i, (fr, sim) in enumerate(wynik.zrodla, start=1):
            print(f"  [{i}] {fr.zrodlo}, str. {fr.strona}  (podobieństwo {sim:.3f})")


if __name__ == "__main__":
    main()
