import requests
from bs4 import BeautifulSoup
import time

headers = {"User-Agent": "Mozilla/5.0"}

def scrape_assessment_details(url):
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            print(f"⚠️ Failed {url}")
            return {}

        soup = BeautifulSoup(res.text, "html.parser")

        details = {}
        for block in soup.find_all("div", class_="product-catalogue-training-calendar__row"):
            h_tag = block.find("h4")
            p_tag = block.find("p")
            if h_tag and p_tag:
                key = h_tag.get_text(strip=True)
                val = p_tag.get_text(strip=True)
                details[key] = val

        return details

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return {}


import pandas as pd

# Load your existing CSV if needed
df = pd.read_csv("shl_individual_test_links_full.csv")

detail_records = []

for i, row in df.iterrows():
    url = row["URL"]
    print(f"[{i+1}/{len(df)}] Scraping details for: {row['Assessment Name']}")
    details = scrape_assessment_details(url)

    # Merge base info + details
    full_record = {**row.to_dict(), **details}
    detail_records.append(full_record)
    time.sleep(1)  # be polite to the site

df_full = pd.DataFrame(detail_records)
df_full.to_csv("shl_assessments_full_details.csv", index=False)
print("✅ Saved full dataset to shl_assessments_full_details.csv")
