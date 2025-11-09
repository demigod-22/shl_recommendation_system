import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {"User-Agent": "Mozilla/5.0"}
root = "https://www.shl.com"
base_url = "https://www.shl.com/products/product-catalog/?start={}&type=1"

data = []

for start in range(0, 384, 12):  # 0 ... 384
    url = base_url.format(start)
    print(f"Scraping {url}")
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print(f"⚠️ Failed to load page start={start}")
        continue

    soup = BeautifulSoup(res.text, "html.parser")
    rows = soup.find_all("tr", {"data-entity-id": True})

    for r in rows:
        cells = r.find_all("td")
        if len(cells) < 4:
            continue

        # Name + URL
        name_tag = cells[0].find("a")
        name = name_tag.get_text(strip=True)
        href = name_tag["href"]
        if not href.startswith("http"):
            href = root + href

        # Remote Testing: check for circle--yes
        remote_tag = cells[1].find("span", class_="catalogue__circle--yes")
        remote_testing = "Yes" if remote_tag else "No"

        # Adaptive/IRT: same pattern
        adaptive_tag = cells[2].find("span", class_="catalogue__circle--yes")
        adaptive_irt = "Yes" if adaptive_tag else "No"

        # Test Type: may contain multiple keys
        test_spans = cells[3].find_all("span", class_="product-catalogue__key")
        test_type = ",".join([s.get_text(strip=True) for s in test_spans]) if test_spans else None

        data.append({
            "Assessment Name": name,
            "URL": href,
            "Remote Testing": remote_testing,
            "Adaptive/IRT": adaptive_irt,
            "Test Type": test_type
        })

    time.sleep(1)

df = pd.DataFrame(data)
print(f"✅ Collected {len(df)} assessments.")
df.to_csv("shl_individual_test_links_full.csv", index=False)
print("✅ Saved to shl_individual_test_links_full.csv")
