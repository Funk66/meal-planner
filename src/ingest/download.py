from __future__ import annotations

from pathlib import Path
import shutil

import kagglehub
import pandas as pd

DATASET_ID = "realalexanderwei/food-com-recipes-with-ingredients-and-tags"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _select_csv(csv_files: list[Path]) -> Path:
    if not csv_files:
        raise FileNotFoundError("No CSV files found in downloaded dataset")

    def rank(path: Path) -> tuple[bool, int, str]:
        name = path.name.lower()
        return ("recipe" not in name, len(name), name)

    return sorted(csv_files, key=rank)[0]


def _find_local_csv() -> Path | None:
    if not DATA_DIR.exists():
        return None
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    return csv_files[0] if csv_files else None


def _download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    csv_files = sorted(dataset_path.rglob("*.csv"))
    selected = _select_csv(csv_files)
    destination = DATA_DIR / selected.name
    if not destination.exists():
        shutil.copy2(selected, destination)
    return destination


def main() -> None:
    csv_path = _find_local_csv()
    if csv_path is None:
        print("No local CSV found in data/. Downloading dataset...")
        csv_path = _download_dataset()
    else:
        print(f"Using cached CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print("Columns:", list(df.columns))
    print(df.head())


if __name__ == "__main__":
    main()
