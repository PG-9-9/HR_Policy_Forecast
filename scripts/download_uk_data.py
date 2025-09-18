# scripts/download_uk_data.py
import os, re, json, time
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

DATA = Path("data")
RAW = DATA / "raw"
CORPUS = DATA / "external" / "uk_corpus"
RAW.mkdir(parents=True, exist_ok=True)
CORPUS.mkdir(parents=True, exist_ok=True)

# --- 1) ONS vacancies time series (CSV) ---
# We’ll grab the “Vacancies ratio per 100 employee jobs – Total” CSV using the provided link.
ONS_SERIES_PAGE = "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/timeseries/ap2z/unem"
CSV_NAME = "ons_vacancies_ratio_total.csv"
CSV_DOWNLOAD_LINK = "https://www.ons.gov.uk/generator?format=csv&uri=/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/timeseries/ap2z/unem"

def download_ons_csv():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    if (RAW / CSV_NAME).exists():
        print("[OK] CSV already exists:", RAW / CSV_NAME)
        return
    link = CSV_DOWNLOAD_LINK
    print("[ONS] CSV:", link)
    r = requests.get(link, headers=headers, timeout=60)
    r.raise_for_status()
    (RAW / CSV_NAME).write_bytes(r.content)
    print("[OK] Saved", RAW / CSV_NAME)

# --- 2) GOV.UK policy corpus (HTML + PDFs) ---
GOV_SOURCES = [
    # Immigration Rules updates index (HTML):
    "https://www.gov.uk/guidance/immigration-rules/updates",
    # Home Office news items (examples of policy-impacting updates):
    "https://www.gov.uk/government/news/major-immigration-reforms-delivered-to-restore-order-and-control",
    "https://www.gov.uk/government/news/record-numbers-of-visa-sponsor-licences-revoked-for-rule-breaking",
    # White paper (PDF) + Statement of Changes (PDF):
    "https://assets.publishing.service.gov.uk/media/6821aec3f16c0654b19060ac/restoring-control-over-the-immigration-system-white-paper.pdf",
    "https://assets.publishing.service.gov.uk/media/6863a3ea08bf2f5376121a67/E03394848_-_HC_997_-_Immigration_Rules_Changes__Print_Ready_.pdf",
]

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)[:160]

def fetch_url(url: str):
    print("[GET]", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    if url.lower().endswith(".pdf"):
        (CORPUS / (sanitize(url) + ".pdf")).write_bytes(r.content)
    else:
        (CORPUS / (sanitize(url) + ".html")).write_text(r.text, encoding="utf-8")

def run():
    download_ons_csv()
    for u in GOV_SOURCES:
        try:
            fetch_url(u)
            time.sleep(0.5)
        except Exception as e:
            print("  [warn] failed:", u, e)

if __name__ == "__main__":
    run()