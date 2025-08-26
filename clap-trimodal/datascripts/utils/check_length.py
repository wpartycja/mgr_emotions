import numpy as np
import pandas as pd
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path} with common encodings.")


def _read_lines_any_encoding(path: Path):
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path} with common encodings.")


def _print_stats(arr: np.ndarray, title: str):
    print(title)
    print(f"Min: {arr.min():.0f}")
    print(f"Median (50th): {np.percentile(arr, 50):.0f}")
    print(f"75th percentile: {np.percentile(arr, 75):.0f}")
    print(f"90th percentile: {np.percentile(arr, 90):.0f}")
    print(f"Max: {arr.max():.0f}")
    print(f"Average: {arr.mean():.2f}")


def print_meld_token_stats(csv_files):
    all_tokens = []
    for fname in csv_files:
        path = Path(fname)
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        df = _read_csv_any_encoding(path)
        if "Utterance" not in df.columns:
            raise KeyError(f"'Utterance' column not found in {path}")
        for t in df["Utterance"].fillna("").astype(str):
            all_tokens.append(len(word_tokenize(t)))
    arr = np.array(all_tokens, dtype=int)
    _print_stats(arr, "Token count stats (MELD):")


def print_iemocap_token_stats(root_dir):
    root = Path(root_dir)
    txt_files = sorted(root.glob("Session*/transcriptions/*.txt"))
    all_tokens = []
    for fp in txt_files:
        for line in _read_lines_any_encoding(fp):
            line = line.strip()
            if ":" not in line:
                continue
            text = line.split(":", 1)[1].strip()
            if not text:
                continue
            all_tokens.append(len(word_tokenize(text)))
    arr = np.array(all_tokens, dtype=int)
    _print_stats(arr, "Token count stats (IEMOCAP):")


if __name__ == "__main__":
    MELD_CSVS = [
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_pt/meld/train.csv",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_pt/meld/val.csv",
        "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_pt/meld/test.csv",
    ]
    print_meld_token_stats(MELD_CSVS)

    print("\n")
    
    IEMOCAP_ROOT = "/home/wpartycja/mgr-data-science/3sem/mgr_emotions/data/processed_pt/iemocap"
    print_iemocap_token_stats(IEMOCAP_ROOT)
