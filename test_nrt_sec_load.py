from annotated_types import doc
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

def extract_instance_from_zip(cik, folder, zip_name):
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{zip_name}"
    r = requests.get(url, headers=HEADERS)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    for name in z.namelist():
        text = z.read(name).decode("utf-8", errors="ignore")
        if "<context" in text or "<xbrli:context" in text:
            print("FOUND CONTEXTS IN:", name)


    for name in z.namelist():
        text = z.read(name).decode("utf-8", errors="ignore")

        # The REAL instance document always contains <xbrli:xbrl>
        if "<xbrli:xbrl" in text:
            st.write(f"Found instance in ZIP: {name}")  # debug
            return name, text
        else:
            st.write(f"Checked {name} in ZIP, not an instance document")  # debug

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

    for elem in root.iter():
        # Extract namespace and tag
        if "}" in elem.tag:
            ns, tag = elem.tag[1:].split("}")
            # st.write(f"Found element: {ns}:  {tag}")  # debug
            # st.write(f"Attributes: {elem.attrib}")  # debug
            # st.write(f"Text: {elem.text}")  # debug
        else:
            continue

        # Only capture facts from known taxonomies
        # if ns not in ("us-gaap", "dei", "ifrs-full"):
        #     continue
        if ns in ("xbrli", "xbrldi", "link", "xlink"):
            continue

        ctx = elem.attrib.get("contextRef")
        unit = elem.attrib.get("unitRef")
        val = (elem.text or "").strip()

        # Facts must have a contextRef
        if ctx is None:
            continue

        facts.append({
            "name": f"{ns}:{tag}",
            "contextRef": ctx,
            "unitRef": unit,
            "value": val,
        })

    return facts, root


# ---------- Inline XBRL helpers ----------

from bs4 import BeautifulSoup

def extract_ixbrl_facts(html_text: str):
    """
    Extract inline XBRL facts from <ix:nonFraction> and <ix:nonNumeric>.
    Returns a list of dicts with name, contextRef, unitRef, value, decimals.
    """
    soup = BeautifulSoup(html_text, "lxml")
    facts = []

    # nonFraction = numeric, nonNumeric = text / enums / extensible lists
    for tag in soup.find_all(["ix:nonfraction", "ix:nonFraction", "ix:nonnumeric", "ix:nonNumeric"]):
        name = tag.get("name")
        ctx = tag.get("contextref") or tag.get("contextRef")
        unit = tag.get("unitref") or tag.get("unitRef")
        decimals = tag.get("decimals")
        val = (tag.text or "").strip()

        facts.append(
            {
                "name": name,              # e.g. "us-gaap:Revenues"
                "contextRef": ctx,         # e.g. "c-91"
                "unitRef": unit,           # e.g. "u-1" or "USD"
                "decimals": decimals,
                "value": val,
            }
        )

    return facts


from lxml import etree, html

from lxml import etree, html
import re

def parse_ixbrl_contexts(html_text: str):
    """
    Extract <xbrli:context> from inline XBRL HTML.
    Handles Workiva-style embedded XML islands.
    """
    XBRLI = "http://www.xbrl.org/2003/instance"

    st.write(html_text) #debug

    # 1. Extract ALL embedded XML fragments from <script> or hidden <div>
    xml_fragments = re.findall(
        r"<xbrli:xbrl[\s\S]*?</xbrli:xbrl>",
        html_text,
        flags=re.IGNORECASE
    )
    
    st.write("xml_framents",xml_fragments) #debug

    doc = html.fromstring(html_text.encode("utf-8",errors="ignore"))
    ctx_nodes = doc.xpath('//*[local-name()="context"]')
    st.write("ctx_nodes",ctx_nodes) #debug

    contexts = {}

    for frag in xml_fragments:
        st.write(f"Parsing XML fragment:\n{frag[:200]}...")  # debug
        try:
            root = etree.fromstring(frag.encode("utf-8"))
        except Exception:
            continue

        # 2. Find all <xbrli:context> inside this fragment
        for ctx in root.xpath('//*[local-name()="context" and namespace-uri()=$ns]',
                              ns=XBRLI):

            cid = ctx.get("id")
            if not cid or cid in contexts:
                continue

            # 3. Extract period
            period = ctx.find(f"{{{XBRLI}}}period")
            if period is None:
                contexts[cid] = {"start": None, "end": None, "instant": None}
                continue

            start = period.find(f"{{{XBRLI}}}startDate")
            end = period.find(f"{{{XBRLI}}}endDate")
            instant = period.find(f"{{{XBRLI}}}instant")

            contexts[cid] = {
                "start": start.text.strip() if start is not None else None,
                "end": end.text.strip() if end is not None else None,
                "instant": instant.text.strip() if instant is not None else None,
            }

    st.write("parsed contexts:", contexts) #debug

    return contexts

# ---------- Normalization ----------

def normalize_facts(facts, contexts, units, accn, form):
    out = {"us-gaap": {}}

    for f in facts:
        # st.write(f)  # debug
        name = f.get("name")
        if not name:
            continue

        tag = name.split(":")[-1]
        ctx = f.get("contextRef")
        if tag=="RevenueFromContractWithCustomerExcludingAssessedTax":
            st.write(f"Processing fact: {tag}, context: {ctx}")  # debug
            st.write("f:", f)  # debug
        unit = f.get("unitRef")
        val_raw = f.get("value")

        if not ctx or not val_raw:
            continue

        try:
            val = float(val_raw.replace(",", ""))
        except Exception:
            continue

        c = contexts.get(ctx, {})
        if tag=="RevenueFromContractWithCustomerExcludingAssessedTax":
            st.write(f"Context for {ctx}: {c}")  # debug
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

        # Convert unitRef → string label
        if unit in units:
            measures = units[unit].get("measures", [])
            unit_label = measures[0] if measures else unit
        else:
            unit_label = unit

        # st.write("unit_label:", unit_label)  # debug

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

        if tag=="RevenueFromContractWithCustomerExcludingAssessedTax":
            st.write(f"Final normalized fact for {tag}:", out["us-gaap"].get(tag))  # debug

    return out

from lxml import etree

def parse_ixbrl_units(html_text: str):
    """
    Extract <xbrli:unit> blocks embedded in inline XBRL HTML.
    Handles:
      - simple units (<measure>)
      - compound units (<divide>)
      - numerator/denominator structures
    Returns:
      { unit_id: { "measures": [ "iso4217:USD" ] } }
      or for divide:
      { unit_id: { "measures": [ "iso4217:USD/xbrli:shares" ] } }
    """
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(html_text.encode("utf-8"), parser=parser)

    ns = {
        "xbrli": "http://www.xbrl.org/2003/instance",
    }

    units = {}

    for unit in tree.xpath("//xbrli:unit", namespaces=ns):
        uid = unit.get("id")
        if not uid:
            continue

        # ---------------------------------------------------------
        # CASE 1: Simple unit
        #   <xbrli:unit id="u-1">
        #       <xbrli:measure>iso4217:USD</xbrli:measure>
        #   </xbrli:unit>
        # ---------------------------------------------------------
        measures = [m.text for m in unit.findall("xbrli:measure", namespaces=ns) if m.text]

        if measures:
            units[uid] = {"measures": measures}
            continue

        # ---------------------------------------------------------
        # CASE 2: Divide unit (EPS, ratios)
        #   <xbrli:unit id="u-2">
        #       <xbrli:divide>
        #           <xbrli:unitNumerator>
        #               <xbrli:measure>iso4217:USD</xbrli:measure>
        #           </xbrli:unitNumerator>
        #           <xbrli:unitDenominator>
        #               <xbrli:measure>xbrli:shares</xbrli:measure>
        #           </xbrli:unitDenominator>
        #       </xbrli:divide>
        #   </xbrli:unit>
        # ---------------------------------------------------------
        divide = unit.find("xbrli:divide", namespaces=ns)
        if divide is not None:
            num = divide.find("xbrli:unitNumerator/xbrli:measure", namespaces=ns)
            den = divide.find("xbrli:unitDenominator/xbrli:measure", namespaces=ns)

            if num is not None and den is not None:
                label = f"{num.text}/{den.text}"
                units[uid] = {"measures": [label]}
                continue

        # ---------------------------------------------------------
        # CASE 3: Unknown / fallback
        # ---------------------------------------------------------
        units[uid] = {"measures": []}

    return units

def normalize_ixbrl_facts(ix_facts, contexts, units, accn: str, form_type: str):
    """
    Normalize inline facts into a companyfacts-like structure.
    Returns a list of normalized fact dicts.
    """
    normalized = []

    for f in ix_facts:
        ctx_id = f["contextRef"]
        unit_id = f["unitRef"]

        ctx = contexts.get(ctx_id, {})
        unit = units.get(unit_id, {"measures": [unit_id] if unit_id else []})

        norm = {
            "accn": accn,
            "form": form_type,
            "name": f["name"],                 # e.g. "us-gaap:Revenues"
            "contextRef": ctx_id,
            "unitRef": unit_id,
            "measures": unit.get("measures"),
            "value": f["value"],
            "decimals": f.get("decimals"),
            "start": ctx.get("start"),
            "end": ctx.get("end"),
            "instant": ctx.get("instant"),
        }
        st.write(f"Normalized fact: {norm}")  # debug

        normalized.append(norm)

    return normalized


# ---------- Full pipeline ----------

def fetch_and_normalize_latest_10q(cik: str):
    accn = get_latest_10q_accession(cik)
    if not accn:
        raise ValueError(f"No 10-Q found for CIK {cik}")

    folder = accession_to_folder(accn)
    index_json = get_filing_index_json(cik, folder)

    filename, content = find_instance_document(cik, folder, index_json)

    if filename.lower().endswith(".htm"):
        # Inline XBRL is the ONLY instance document
        ix_facts = extract_ixbrl_facts(content)          # <ix:nonFraction>, <ix:nonNumeric>
        contexts = parse_ixbrl_contexts(content)         # <xbrli:context> inside HTML
        units = parse_ixbrl_units(content)               # <xbrli:unit> inside HTML (you need this too)

        normalized = normalize_facts(ix_facts, contexts, units, accn, "10-Q")
        return accn, filename, normalized

    # ---------------------------------------------------------
    # PURE XML BRANCH
    # ---------------------------------------------------------
    else:
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
