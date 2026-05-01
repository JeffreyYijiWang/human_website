import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# The base URL for the PNG dataset
BASE_URL = "https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Female-Images/PNG_format/"

# Mapping your folder names to the URL variables
# Add or remove from this list as needed
BODY_PARTS = ["abdomen", "head", "legs", "pelvis", "thighs", "thorax"]

def download_body_part(part_name, session):
    # 1. Setup local directory (e.g., male_abdomen)
    target_dir = Path(f"images/female/female_{part_name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Construct the specific URL
    # Example: .../PNG_format/abdomen/
    part_url = urljoin(BASE_URL, f"{part_name}/index.html")
    
    print(f"\n--- Processing: {part_name} ---")
    print(f"URL: {part_url}")
    
    try:
        r = session.get(part_url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        # 3. Find all PNG links
        links = [urljoin(part_url, a['href']) for a in soup.select("a[href$='.png']")]
        print(f"Found {len(links)} files for {part_name}.")

        # 4. Download files
        for i, file_url in enumerate(links, 1):
            filename = file_url.split("/")[-1]
            out_path = target_dir / filename

            if out_path.exists():
                continue

            print(f"[{i}/{len(links)}] Downloading {filename}...", end="\r")
            
            with session.get(file_url, stream=True) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Politeness delay
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Error processing {part_name}: {e}")

def main():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Compatible; Research-Bot/1.0)"})
    
    for part in BODY_PARTS:
        download_body_part(part, s)
    
    print("\n\n✅ All downloads complete.")

if __name__ == "__main__":
    main()