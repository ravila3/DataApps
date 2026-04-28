import streamlit as st
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
from datetime import datetime
from bs4 import BeautifulSoup

BASE = "https://data.sec.gov"
HEADERS = {
    "User-Agent": "Your Name your@email.com"
}

# ---------- Fetch helpers ----------

def get_latest_10q_accession(cik: str):
    url = f"{BASE}/submissions/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()

    recent = data["filings"]["recent"]
    for form, accn in zip(recent["form"], recent["accessionNumber"]):
        if form == "10-Q":
            return accn  # e.g. "0001030894-26-000032"
    return None


def accession_to_folder(accn: str) -> str:
    return accn.replace("-", "")


def get_filing_index_json(cik: str, folder: str):
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/index.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def download_file(cik: str, folder: str, filename: str) -> str:
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{filename}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.text


# ---------- Instance detection ----------

def is_instance_document(text: str) -> bool:
    t = text.lower()
    return (
        "<xbrli:xbrl" in t
        or "http://www.xbrl.org/2003/instance" in t
        or "<ix:nonfraction" in t
        or "<ix:nonnumeric" in t
    )


def extract_instance_from_zip(cik: str, folder: str, zip_name: str):
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{zip_name}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))

    for name in z.namelist():
        text = z.read(name).decode("utf-8", errors="ignore")
        if is_instance_document(text):
            return name, text

    return None, None


def find_instance_document(cik: str, folder: str, index_json: dict):
    # 1. Inline XBRL (.htm)
    for item in index_json["directory"]["item"]:
        name = item["name"]
        lower = name.lower()
        if lower.endswith(".htm"):
            text = download_file(cik, folder, name)
            if is_instance_document(text):
                return name, text

    # 2. ZIP package
    for item in index_json["directory"]["item"]:
        name = item["name"]
        lower = name.lower()
        if lower.endswith(".zip"):
            fname, text = extract_instance_from_zip(cik, folder, name)
            if text:
                return fname, text

    # 3. XML fallback
    for item in index_json["directory"]["item"]:
        name = item["name"]
        lower = name.lower()
        if lower.endswith(".xml"):
            text = download_file(cik, folder, name)
            if is_instance_document(text):
                return name, text

    raise ValueError("No XBRL instance found")


# ---------- XML helpers ----------

def parse_contexts(root):
    ns = "{http://www.xbrl.org/2003/instance}"
    contexts = {}
    for ctx in root.findall(f".//{ns}context"):
        cid = ctx.attrib["id"]
        period = ctx.find(f"{ns}period")

        start = period.find(f"{ns}startDate")
        end = period.find(f"{ns}endDate")
        instant = period.find(f"{ns}instant")

        contexts[cid] = {
            "start": start.text if start is not None else None,
            "end": end.text if end is not None else None,
            "instant": instant.text if instant is not None else None,
        }
    return contexts


def parse_units(root):
    ns = "{http://www.xbrl.org/2003/instance}"
    units = {}
    for unit in root.findall(f".//{ns}unit"):
        uid = unit.attrib["id"]
        measure = unit.find(f"{ns}measure")
        if measure is not None:
            units[uid] = measure.text.split(":")[-1]
    return units


def extract_xml_facts(xml_text: str):
    root = ET.fromstring(xml_text)
    facts = []
    for elem in root:
        tag = elem.tag.split("}")[-1]
        if not tag or not tag[0].isupper():
            continue
        ctx = elem.attrib.get("contextRef")
        unit = elem.attrib.get("unitRef")
        val = (elem.text or "").strip()
        facts.append(
            {
                "name": tag,
                "contextRef": ctx,
                "unitRef": unit,
                "value": val,
            }
        )
    return facts, root


# ---------- Inline XBRL helpers ----------

def extract_ixbrl_facts(html_text: str):
    soup = BeautifulSoup(html_text, "lxml")
    facts = []

    # numeric
    for tag in soup.find_all(["ix:nonfraction", "ix:nonFraction"]):
        facts.append(
            {
                "name": tag.get("name"),
                "contextRef": tag.get("contextref"),
                "unitRef": tag.get("unitref"),
                "decimals": tag.get("decimals"),
                "value": tag.text.strip(),
            }
        )

    # non-numeric (we mostly care about numeric for companyfacts-like)
    for tag in soup.find_all(["ix:nonnumeric", "ix:nonNumeric"]):
        facts.append(
            {
                "name": tag.get("name"),
                "contextRef": tag.get("contextref"),
                "unitRef": None,
                "value": tag.text.strip(),
            }
        )

    return facts

def parse_ixbrl_contexts(html_text):
    soup = BeautifulSoup(html_text, "lxml")
    contexts = {}

    # Find ANY tag whose name ends with "context"
    for ctx in soup.find_all(lambda tag: tag.name.lower().endswith("context")):
        cid = ctx.get("id")
        if not cid:
            continue

        # Find period inside this context
        period = None
        for child in ctx.descendants:
            if hasattr(child, "name") and child.name and child.name.lower().endswith("period"):
                period = child
                break

        if not period:
            continue

        def find_date(tag, suffix):
            for child in tag.descendants:
                if hasattr(child, "name") and child.name and child.name.lower().endswith(suffix):
                    return child.text.strip()
            return None

        start = find_date(period, "startdate")
        end = find_date(period, "enddate")
        instant = find_date(period, "instant")

        contexts[cid] = {
            "start": start,
            "end": end,
            "instant": instant,
        }

    return contexts

# ---------- Normalization ----------

def normalize_facts(facts, contexts, units, accn, form):
    out = {"us-gaap": {}}

    for f in facts:
        st.write(f)  # debug
        name = f.get("name")
        if not name:
            continue

        tag = name.split(":")[-1]
        ctx = f.get("contextRef")
        st.write(f"Processing fact: {tag}, context: {ctx}")  # debug
        unit = f.get("unitRef")
        val_raw = f.get("value")

        if not ctx or not val_raw:
            continue

        try:
            val = float(val_raw.replace(",", ""))
        except Exception:
            continue

        c = contexts.get(ctx, {})
        start = c.get("start")
        end = c.get("end")
        instant = c.get("instant")

        dt_str = end or instant
        if not dt_str:
            continue

        dt = datetime.fromisoformat(dt_str)
        fy = dt.year
        fp = f"Q{((dt.month - 1) // 3) + 1}"
        frame = f"CY{fy}{fp}"

        unit_label = units.get(unit, "USD")

        out["us-gaap"].setdefault(tag, {"units": {}})
        out["us-gaap"][tag]["units"].setdefault(unit_label, [])

        out["us-gaap"][tag]["units"][unit_label].append(
            {
                "start": start,
                "end": end,
                "val": val,
                "accn": accn,
                "form": form,
                "fy": fy,
                "fp": fp,
                "frame": frame,
            }
        )

    return out


# ---------- Full pipeline ----------

def fetch_and_normalize_latest_10q(cik: str):
    accn = get_latest_10q_accession(cik)
    if not accn:
        raise ValueError("No 10-Q found for CIK " + cik)

    folder = accession_to_folder(accn)
    index_json = get_filing_index_json(cik, folder)

    filename, content = find_instance_document(cik, folder, index_json)

    st.write(content)  # debug
    # If inline XBRL, we still need contexts/units from XML (usually in ZIP)
    if filename.lower().endswith(".htm"):
        # Inline XBRL detected — ignore it
        # Always extract the XML instance from the ZIP
        zip_name = next(
            item["name"] for item in index_json["directory"]["item"]
            if item["name"].lower().endswith(".zip")
        )

        xml_fname, xml_text = extract_instance_from_zip(cik, folder, zip_name)
        if not xml_text:
            raise ValueError("Could not extract XML instance from ZIP")

        xml_facts, root = extract_xml_facts(xml_text)
        contexts = parse_contexts(root)
        units = parse_units(root)

        normalized = normalize_facts(xml_facts, contexts, units, accn, "10-Q")

    else:
        # pure XML instance
        xml_facts, root = extract_xml_facts(content)
        contexts = parse_contexts(root)
        units = parse_units(root)
        normalized = normalize_facts(xml_facts, contexts, units, accn, "10-Q")

    return accn, filename, normalized

if __name__ == "__main__":
    CIK = "0001030894"  # change as needed
    accn, fname, data = fetch_and_normalize_latest_10q(CIK)

    print("Accession:", accn)
    print("Instance file:", fname)
    print("Top-level us-gaap keys:", list(data["us-gaap"].keys())[:20])
    st.json(data)
