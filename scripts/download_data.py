"""Pobierz oryginalny CSV przez publiczne API Kaggle (bez kopiowania z mirrorów)."""

from pathlib import Path
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASET = "bobbyscience/league-of-legends-diamond-ranked-games-10-min"
FILENAME = "high_diamond_ranked_10min.csv"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"


def main():
    destination = ROOT / "data" / "raw" / FILENAME
    if destination.exists():
        print(f"Plik już istnieje: {destination}. Usuń go, jeśli chcesz pobrać ponownie.")
        return
    request = urllib.request.Request(URL, headers={"User-Agent": "lol-statistical-analysis/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            csv = zipped.read(FILENAME)
    except (urllib.error.URLError, zipfile.BadZipFile, KeyError) as exc:
        raise SystemExit(
            "Nie udało się pobrać publicznego archiwum Kaggle. "
            f"Pobierz {FILENAME} ręcznie ze strony datasetu i umieść w data/raw/. "
            f"Szczegóły: {exc}"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv)
    metadata = {
        "source": f"https://www.kaggle.com/datasets/{DATASET}",
        "download_url": URL,
        "file": FILENAME,
        "sha256": hashlib.sha256(csv).hexdigest(),
        "bytes": len(csv),
    }
    (destination.parent / "source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Pobrano {destination} ({len(csv):,} bajtów).")


if __name__ == "__main__":
    main()
