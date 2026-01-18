"""
01_id_mapping.py

Builds SRR -> FUR mapping and validates overlap between Jighly et al. and your samples.
"""

import re
import pandas as pd
from pathlib import Path

# ------------------ PATHS ------------------
JIGHLY_XLSX = r"C:\Users\ms\Desktop\mango\data\main_data\jighly.xlsx"
RUNINFO_CSV = r"C:\Users\ms\Desktop\mango\data\main_data\SraRunInfo_PRJNA1175065.csv"
SAMPLES_CSV = r"C:\Users\ms\Desktop\mango\output\idea_2\core_ml\samples.csv"
OUT_MAP     = r"C:\Users\ms\Desktop\mango\data\main_data\srr_to_fur_map.csv"

# ------------------ STEP 1: LOAD TARGET IDS ------------------
print("=" * 60)
print("STEP 1: Loading target IDs")
print("=" * 60)

hr = pd.read_excel(JIGHLY_XLSX, sheet_name="TableS1")
hr_aus_srr = set(hr.loc[hr["Population"] == "Australia", "Geno"].astype(str))
print(f"Jighly Australian SRRs: {len(hr_aus_srr)}")

your = pd.read_csv(SAMPLES_CSV)
your_ids = set(your["sample_id"].astype(str))
print(f"Your FUR sample IDs: {len(your_ids)}")

# ------------------ STEP 2: BUILD MAPPING FROM RUNINFO ------------------
print("\n" + "=" * 60)
print("STEP 2: Building SRR -> FUR mapping from RunInfo")
print("=" * 60)

runinfo_path = Path(RUNINFO_CSV)
if not runinfo_path.exists():
    raise FileNotFoundError(
        f"RunInfo CSV not found:\n  {RUNINFO_CSV}\n"
        "Download from NCBI SRA (Send to -> File -> RunInfo)."
    )

ri = pd.read_csv(RUNINFO_CSV, dtype=str).fillna("")
ri.columns = [c.strip() for c in ri.columns]

# Extract FUR IDs
fur_pat = re.compile(r"\bFUR[0-9A-Za-z]+\b")

def extract_fur(x: str):
    m = fur_pat.search(x)
    return m.group(0) if m else None

# Find best column for FUR extraction
best_col, best_hits = None, 0
for c in ri.columns:
    extracted = ri[c].astype(str).map(extract_fur)
    hits = extracted.notna().sum()
    if hits > best_hits:
        best_hits = hits
        best_col = c

if best_col is None or best_hits == 0:
    raise RuntimeError("Could not extract any FUR IDs from RunInfo table.")

print(f"Best column for FUR extraction: '{best_col}' ({best_hits} hits)")

ri["sample_id"] = ri[best_col].astype(str).map(extract_fur)

# Find Run column
run_col = next((c for c in ["Run", "run", "RUN"] if c in ri.columns), None)
if run_col is None:
    raise RuntimeError(f"No 'Run' column found. Columns: {list(ri.columns)[:10]}...")

# Build mapping
m = ri.loc[ri[run_col].isin(hr_aus_srr), [run_col, "sample_id"]].dropna().drop_duplicates()
m = m.rename(columns={run_col: "Geno"})
m = m.loc[m["sample_id"].isin(your_ids)].copy()

m.to_csv(OUT_MAP, index=False)
print(f"Saved mapping: {OUT_MAP}")
print(f"Mapped: {len(m)} / {len(hr_aus_srr)} HR Australian SRRs")

# ------------------ STEP 3: CHECK OVERLAP ------------------
print("\n" + "=" * 60)
print("STEP 3: Validating overlap")
print("=" * 60)

map_df = pd.read_csv(OUT_MAP, dtype=str)
your_srr = pd.DataFrame({"sample_id": list(your_ids)}).merge(map_df, on="sample_id", how="left")["Geno"].dropna()
overlap = set(hr_aus_srr) & set(your_srr)
print(f"SRR overlap after mapping: {len(overlap)}")

# ------------------ STEP 4: IDENTIFY MISSING ------------------
print("\n" + "=" * 60)
print("STEP 4: Identifying missing mappings")
print("=" * 60)

missing_srr = sorted(set(hr_aus_srr) - set(m["Geno"]))
missing_fur = sorted(set(your_ids) - set(m["sample_id"]))

print(f"Missing HR SRRs (not mapped): {missing_srr if missing_srr else 'None'}")
print(f"Missing FUR IDs (no SRR): {len(missing_fur)} total")
if missing_fur:
    print(f"  First 10: {missing_fur[:10]}")

# ------------------ SUMMARY ------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total Jighly Australian samples: {len(hr_aus_srr)}")
print(f"Total your samples: {len(your_ids)}")
print(f"Successfully mapped: {len(m)}")
print(f"Mapping rate: {100 * len(m) / len(hr_aus_srr):.1f}%")
print(f"\nMapping file: {OUT_MAP}")