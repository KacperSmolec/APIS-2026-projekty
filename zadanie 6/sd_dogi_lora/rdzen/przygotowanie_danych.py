"""Przygotowanie danych treningowych ze zbioru Stanford Dogs.

Skrypt wybiera obrazy 5 wskazanych ras psów, kopiuje je do katalogu treningowego
i tworzy plik ``metadata.jsonl`` w formacie wymaganym przez bibliotekę
Hugging Face Diffusers (każdy obraz dostaje prompt o spójnej strukturze
"photo of <rasa> dog").

Uruchomienie (np. na Colab, po pobraniu i rozpakowaniu zbioru):
    python -m rdzen.przygotowanie_danych
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# Katalog z rozpakowanym zbiorem Stanford Dogs (folder "Images" z podfolderami ras).
KATALOG_ZRODLOWY = Path("dane/Images")
KATALOG_TRENINGOWY = Path("dane/zbior_treningowy")

# Wybrane 5 ras. Klucz = fragment nazwy folderu w Stanford Dogs,
# wartość = nazwa rasy używana w promptcie.
WYBRANE_RASY = {
    "beagle": "beagle",
    "golden_retriever": "golden retriever",
    "German_shepherd": "german shepherd",
    "Siberian_husky": "siberian husky",
    "boxer": "boxer",
}

# Ile maksymalnie obrazów na rasę (mniej = szybszy trening na słabszym GPU).
MAKS_OBRAZOW_NA_RASE = 80


def znajdz_folder_rasy(katalog: Path, fragment: str) -> Path | None:
    """Znajduje podfolder zawierający w nazwie zadany fragment (np. 'beagle')."""
    if not katalog.exists():
        return None
    for folder in katalog.iterdir():
        if folder.is_dir() and fragment.lower() in folder.name.lower():
            return folder
    return None


def przygotuj() -> int:
    """Kopiuje obrazy wybranych ras i tworzy metadata.jsonl. Zwraca liczbę par."""
    KATALOG_TRENINGOWY.mkdir(parents=True, exist_ok=True)
    metadane = []

    for fragment, nazwa_rasy in WYBRANE_RASY.items():
        folder = znajdz_folder_rasy(KATALOG_ZRODLOWY, fragment)
        if folder is None:
            print(f"[UWAGA] Nie znaleziono folderu dla rasy '{nazwa_rasy}' (fragment: {fragment}).")
            continue

        print(f"Przetwarzam: {nazwa_rasy}  (folder: {folder.name})")
        obrazy = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for obraz in obrazy[:MAKS_OBRAZOW_NA_RASE]:
            nowa_nazwa = f"{nazwa_rasy.replace(' ', '_')}_{obraz.name}"
            shutil.copy2(obraz, KATALOG_TRENINGOWY / nowa_nazwa)
            metadane.append({"file_name": nowa_nazwa, "text": f"photo of {nazwa_rasy} dog"})

    plik_meta = KATALOG_TRENINGOWY / "metadata.jsonl"
    with open(plik_meta, "w", encoding="utf-8") as f:
        for wpis in metadane:
            f.write(json.dumps(wpis, ensure_ascii=False) + "\n")

    print(f"\nGotowe. Zapisano {len(metadane)} par obraz-prompt w: {KATALOG_TRENINGOWY}")
    return len(metadane)


if __name__ == "__main__":
    przygotuj()
