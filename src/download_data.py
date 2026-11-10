"""Download the open Transfermarkt dataset (CC0) used for training.

Source: dcaribou/transfermarkt-datasets (also on Kaggle).
If a download fails, get the CSVs manually from Kaggle
("Football Data from Transfermarkt") and drop them into data/raw/.
"""
import requests
from config import RAW

BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
FILES = ["players", "player_valuations", "appearances"]


def main():
    for name in FILES:
        target = RAW / f"{name}.csv.gz"
        if target.exists() or (RAW / f"{name}.csv").exists():
            print(f"skip {name} (already there)")
            continue
        url = f"{BASE}/{name}.csv.gz"
        print("downloading", url)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    print("done ->", RAW)


if __name__ == "__main__":
    main()
