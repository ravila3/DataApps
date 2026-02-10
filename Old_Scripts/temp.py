import requests
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CIK = "0000789019"  # Microsoft
USER_AGENT = "Rafael Avila rafaelavila3@gmail.com"

TAGS = [
    "us-gaap:DepreciationDepletionAndAmortization",
    "us-gaap:Depreciation",
    "us-gaap:AmortizationOfIntangibleAssets",
    "us-gaap:DepreciationAndAmortization"
]

ALLOWED_FORMS = {"10-Q", "10-K"}

# ---------------------------------------------------------
# DOWNLOAD SEC JSON
# ---------------------------------------------------------
url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
headers = {"User-Agent": USER_AGENT}

resp = requests.get(url, headers=headers)
resp.raise_for_status()
data = resp.json()

# ---------------------------------------------------------
# EXTRACT FACTS FOR EACH TAG
# ---------------------------------------------------------
records = []

for tag in TAGS:
    if tag not in data["facts"]["us-gaap"]:
        continue

    fact = data["facts"]["us-gaap"][tag]

    for unit, items in fact.get("units", {}).items():
        if unit != "USD":
            continue

        for entry in items:
            form = entry.get("form")
            if form not in ALLOWED_FORMS:
                continue

            start = entry.get("start")
            end = entry.get("end")
            val = entry.get("val")
            acc = entry.get("accn")

            # Only duration contexts (start + end)
            if not start or not end:
                continue

            # Convert to datetime
            try:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)
            except:
                continue

            # Compute quarter label
            q = f"{end_dt.year}-Q{((end_dt.month - 1)//3) + 1}"

            records.append({
                "tag": tag,
                "value": val,
                "start": start,
                "end": end,
                "quarter": q,
                "form": form,
                "accession": acc,
                "unit": unit
            })

# ---------------------------------------------------------
# BUILD DATAFRAME
# ---------------------------------------------------------
df = pd.DataFrame(records)

# Deduplicate: keep the latest accession per tag+quarter
df = df.sort_values(["tag", "quarter", "accession"])
df = df.drop_duplicates(subset=["tag", "quarter"], keep="last")

# Pivot to quarterly table
table = df.pivot(index="quarter", columns="tag", values="value")
table = table.sort_index()

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------
print("\n================ QUARTERLY TABLE ================\n")
print(table)

print("\n================ PANDAS-READY JSON ================\n")
print(df.to_json(orient="records", indent=2))