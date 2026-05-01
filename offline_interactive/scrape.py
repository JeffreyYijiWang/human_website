#!/usr/bin/env python3
import re
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

URL = "https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/PNG_format/thorax/index.html"
OUT_DIR = Path("images/male/male_thorax")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; dataset-downloader/1.0)"})

    r = s.get(URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    png_links = []
    for a in soup.select("a[href$='.png']"):
        href = a.get("href", "").strip()
        if href:
            png_links.append(urljoin(URL, href))

    print(f"Found {len(png_links)} files.")

    for i, file_url in enumerate(png_links, 1):
        filename = file_url.split("/")[-1]
        out_path = OUT_DIR / filename

        if out_path.exists():
            print(f"[{i}/{len(png_links)}] skip {filename} (exists)")
            continue

        print(f"[{i}/{len(png_links)}] download {filename}")
        with s.get(file_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        time.sleep(0.1)  # be polite to the server

if __name__ == "__main__":
    main()
