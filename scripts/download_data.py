import os, zipfile, json, re, time
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm

DATA = Path("data")
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
RAW.mkdir(parents=True, exist_ok=True)
EXTERNAL.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "kashnitsky/hierarchical-dataset"
KAGGLE_FILE_HINT = "train.csv"  # just to sanity-check

PUBLIC_TS_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
# Simple daily series (Melbourne min temp). We'll treat it as "hiring pipeline" proxy for demo.

CORPUS_URLS = [
    # small demo corpus; you can expand this list freely
    "https://en.wikipedia.org/wiki/Work_permit",
    "https://en.wikipedia.org/wiki/Blue_Card_(European_Union)",
    "https://en.wikipedia.org/wiki/Labour_law",
    "https://en.wikipedia.org/wiki/Minimum_wage",
    "https://ec.europa.eu/commission/presscorner/api/documents?query=work%20permit",
]

def try_kaggle():
    try:
        import kaggle  # noqa
    except Exception:
        print("[KAGGLE] kaggle package not installed or no credentials. Skipping Kaggle download.")
        return False

    print("[KAGGLE] Attempting to download:", KAGGLE_DATASET)
    code = os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p "{RAW}"')
    if code != 0:
        print("[KAGGLE] Kaggle CLI failed. Skipping.")
        return False

    # Unzip any new zips
    for zf in RAW.glob("*.zip"):
        print(f"[KAGGLE] Unzipping {zf} ...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(RAW)
        zf.unlink()

    # sanity check
    files = list(RAW.glob("*.csv"))
    ok = any(KAGGLE_FILE_HINT in f.name for f in files) or len(files) > 0
    print(f"[KAGGLE] CSV files found: {[f.name for f in files]}")
    return ok

def download_file(url: str, out_path: Path):
    print(f"[HTTP] {url} -> {out_path}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.write_bytes(r.content)

def download_public_ts():
    out = RAW / "daily_min_temperatures.csv"
    download_file(PUBLIC_TS_URL, out)
    print("[OK] Saved", out)

def sanitize_filename(url: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9\-_.]', '_', url)
    return s[:120]

def download_corpus(urls: List[str]):
    corpus_dir = EXTERNAL / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for url in tqdm(urls, desc="Downloading corpus"):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            name = sanitize_filename(url) + ".txt"
            (corpus_dir / name).write_text(r.text, encoding="utf-8")
        except Exception as e:
            print("  [warn] failed:", url, e)
            continue
    # index list
    (corpus_dir / "_sources.json").write_text(json.dumps(urls, indent=2), encoding="utf-8")
    print("[OK] Corpus saved to", corpus_dir)

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    used_kaggle = try_kaggle()
    if not used_kaggle:
        print("[INFO] Using public fallback dataset.")
        download_public_ts()

    download_corpus(CORPUS_URLS)
    print("[DONE] Data download complete.")
