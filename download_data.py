"""Download UCI Breast Cancer Coimbra (dataset 451) into ./data/dataR2.csv.

Dataset attribution: Patricio et al. (2018), DOI: 10.24432/C52P59, CC BY 4.0.
"""
from pathlib import Path

from stats import DATA_PATH, load_data


def main() -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt") from exc

    print("Fetching UCI Breast Cancer Coimbra dataset (ID 451)...")
    dataset = fetch_ucirepo(id=451)
    features = dataset.data.features.copy().reset_index(drop=True)
    targets = dataset.data.targets.copy().reset_index(drop=True)
    if targets.shape[1] != 1:
        raise RuntimeError("Expected one classification column in the UCI dataset.")
    targets.columns = ["Classification"]
    frame = features.join(targets)

    out: Path = DATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    checked = load_data(out)
    print(f"Saved {len(checked)} rows to {out}")
    print("Group counts:", checked["Classification"].value_counts().sort_index().to_dict())
    print("Labels: 1=healthy controls, 2=breast cancer patients")


if __name__ == "__main__":
    main()
