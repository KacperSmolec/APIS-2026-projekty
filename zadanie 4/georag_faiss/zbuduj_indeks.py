"""Budowanie bazy wektorowej z PDF-ów w katalogu `dane/`.

Uruchomienie:
    python zbuduj_indeks.py
"""
from rdzen import SilnikRAG

if __name__ == "__main__":
    silnik = SilnikRAG()
    liczba = silnik.zbuduj_indeks()
    print(f"\nGotowe. Zaindeksowano {liczba} fragmentów. Baza zapisana w katalogu 'indeks/'.")
