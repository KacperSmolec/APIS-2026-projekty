"""Uruchomienie agenta newsowego z linii poleceń.

Użycie:
    python uruchom.py "energetyka jądrowa"
    python uruchom.py            # zapyta o temat interaktywnie
"""
import sys

from rdzen import AgentNewsowy


def main() -> None:
    temat = " ".join(sys.argv[1:]).strip()
    if not temat:
        temat = input("Podaj temat raportu: ").strip()
    if not temat:
        print("Nie podano tematu.")
        return

    agent = AgentNewsowy()
    print(f"\n=== Agent pracuje nad tematem: {temat} ===\n")
    wynik = agent.uruchom(temat)

    print("\n--- ŚLAD DECYZJI AGENTA ---")
    for i, krok in enumerate(wynik.slad, 1):
        print(f"{i}. {krok.narzedzie}({', '.join(krok.argumenty)}) -> {krok.obserwacja[:90]}...")

    print("\n--- ODPOWIEDŹ KOŃCOWA ---")
    print(wynik.odpowiedz)


if __name__ == "__main__":
    main()
