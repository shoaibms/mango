# -*- coding: utf-8 -*-
"""


KASP ASSAY SEQUENCE GENERATOR (16_kasp_assay_design.py)
- Fixed default paths (run with no args).
- FASTA: uses NCBI/RefSeq FASTA when available for NC_058xxx contigs; falls back to GWHABLA.
- VCF: fast indexed lookup only (pysam OR tabix/bcftools). No full-file streaming scan.
- Outputs Table S9 plus skipped targets file.
"""

from __future__ import annotations
import os
import re
import csv
import gzip
import math
import shutil
import argparse
import subprocess
import sys
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterable

# =============================================================================
# Chromosome name mapping (NCBI RefSeq ↔ GWHABLA)
# =============================================================================
def build_ncbi_to_gwhabla_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    # chrom 1..20 → NC_058137.1..NC_058156.1, and GWHABLA00000001..GWHABLA00000020
    for chrom_num in range(1, 21):
        ncbi_num = 58136 + chrom_num
        ncbi_name = f"NC_0{ncbi_num}.1"
        gwhabla_name = f"GWHABLA{chrom_num:08d}"
        mapping[ncbi_name] = gwhabla_name
        mapping[f"NC_0{ncbi_num}"] = gwhabla_name
    return mapping

NCBI_TO_GWHABLA = build_ncbi_to_gwhabla_map()

# =========================
# Fixed repo defaults
# =========================
DEFAULT_ROOT = r"C:\Users\ms\Desktop\mango"

DEFAULT_INPUT_CSV = os.path.join(
    DEFAULT_ROOT, "output", "idea_2", "breeder_tools", "tag_snps", "inversion_tag_snps_selected.csv"
)
DEFAULT_MERGE_METRICS_CSV = os.path.join(
    DEFAULT_ROOT, "output", "idea_2", "breeder_tools", "tag_snps", "inversion_tag_snps_manuscript_table.csv"
)
DEFAULT_OUT_CSV = os.path.join(
    DEFAULT_ROOT, "output", "idea_2", "breeder_tools", "Table_S9_Inversion_Tag_SNP_Assays.csv"
)

# Your two VCFs (NC_058xxx coordinates)
DEFAULT_VCF_PATHS = [
    os.path.join(DEFAULT_ROOT, "data", "main_data", "11_QF1.vcf.gz"),
    os.path.join(DEFAULT_ROOT, "data", "main_data", "12_QF2.vcf.gz"),
]

# FASTA candidates (we will also auto-discover under data\ncbi and data\main_data)
DEFAULT_FASTA_CANDIDATES = [
    # Prefer FASTA that contains NC_058xxx.* contigs (matches VCF coordinates)
    os.path.join(DEFAULT_ROOT, "data", "ncbi", "ncbi_refseq.fa.gz"),
    os.path.join(DEFAULT_ROOT, "data", "main_data", "ncbi_refseq.fa.gz"),
    os.path.join(DEFAULT_ROOT, "data", "ncbi_refseq.fa.gz"),
    # GWHABLA fallback (used for most contigs, but miinv7.0 will still fail without NC FASTA)
    os.path.join(DEFAULT_ROOT, "data", "GWHABLA00000000.genome.fasta.gz"),
    os.path.join(DEFAULT_ROOT, "data", "main_data", "GWHABLA00000000.genome.fasta.gz"),
    os.path.join(DEFAULT_ROOT, "GWHABLA00000000.genome.fasta.gz"),
]

FASTA_SEARCH_DIRS = [
    os.path.join(DEFAULT_ROOT, "data"),
    os.path.join(DEFAULT_ROOT, "data", "ncbi"),
    os.path.join(DEFAULT_ROOT, "data", "main_data"),
    DEFAULT_ROOT,
]

# Try to install pysam automatically if missing (so you don't need extra commands).
AUTO_INSTALL_PYSAM = True

# QC thresholds
DEFAULT_FLANK = 60
DEFAULT_WARN_COMPLEXITY = 0.7
DEFAULT_WARN_AT_RICH = 0.7

# Behaviour
DEFAULT_SKIP_VCF = False  # marker-grade default = DO allele lookup
REQUIRE_INDEXED_VCF = True  # never stream-scan huge VCFs


# =========================
# Helpers
# =========================
def parse_variant_id(vid: str) -> Tuple[str, int]:
    """
    Accepts 'NC_058137.1:14915061' or 'NC_058137.1:14915061:A:G' etc.
    Returns (chrom, pos).
    """
    parts = vid.split(":")
    if len(parts) < 2:
        raise ValueError(f"Unrecognised variant_id: {vid}")
    chrom = parts[0]
    pos = int(parts[1])
    return chrom, pos


def shannon_complexity(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    counts = Counter([b for b in seq if b in "ACGT"])
    n = sum(counts.values())
    if n == 0:
        return 0.0
    freqs = [c / n for c in counts.values()]
    h = -sum(f * math.log2(f) for f in freqs if f > 0.0)
    return min(1.0, h / 2.0)  # max entropy for DNA is 2 bits


def at_fraction(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    a = seq.count("A")
    t = seq.count("T")
    n = len(seq)
    return (a + t) / n if n else 0.0


def max_homopolymer_run(seq: str) -> int:
    seq = seq.upper()
    best = 0
    cur = 0
    prev = ""
    for ch in seq:
        if ch == prev:
            cur += 1
        else:
            prev = ch
            cur = 1
        best = max(best, cur)
    return best


def has_dinuc_repeat(seq: str, min_repeats: int = 6) -> bool:
    """
    Detect simple dinucleotide repeats like ATATATAT... (>= min_repeats units).
    """
    seq = seq.upper()
    if len(seq) < 2 * min_repeats:
        return False
    for i in range(len(seq) - 2 * min_repeats + 1):
        unit = seq[i:i+2]
        if unit[0] == "N" or unit[1] == "N":
            continue
        rep = unit * min_repeats
        if seq.startswith(rep, i):
            return True
    return False


def flank_warnings(left: str, right: str, warn_complexity: float, warn_at: float) -> Tuple[str, float, float, int]:
    seq = (left + right).upper()
    comp = shannon_complexity(seq)
    atf = at_fraction(seq)
    hom = max_homopolymer_run(seq)

    flags = []
    if comp < warn_complexity:
        flags.append("LOW_COMPLEXITY")
    if atf >= warn_at:
        flags.append("AT_RICH")
    if hom >= 8:
        flags.append("HOMOPOLYMER")
    if has_dinuc_repeat(seq, min_repeats=6):
        flags.append("DINUC_REPEAT")

    return ";".join(flags), comp, atf, hom


def discover_fasta_paths() -> List[str]:
    paths = list(DEFAULT_FASTA_CANDIDATES)
    exts = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")
    for d in FASTA_SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                low = fn.lower()
                if low.endswith(exts):
                    paths.append(os.path.join(root, fn))
    # de-dup, keep order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    # prefer NCBI-ish names first
    def score(p: str) -> int:
        lp = p.lower()
        s = 0
        if "refseq" in lp or "ncbi" in lp or "genomic" in lp or "gcf" in lp:
            s -= 10
        if "gwhabla" in lp:
            s += 5
        return s
    uniq.sort(key=score)
    return uniq


def open_text_maybe_gz(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def load_fasta_subset(fasta_path: str, needed_contigs: set) -> Dict[str, str]:
    """
    Reads FASTA and stores sequences only for contigs in needed_contigs.
    Works for .fa and .fa.gz.
    """
    seqs: Dict[str, List[str]] = {}
    cur_id: Optional[str] = None

    with open_text_maybe_gz(fasta_path) as f:
        for line in f:
            if not line:
                continue
            if line[0] == ">":
                cur_id = line[1:].strip().split()[0]
                if cur_id in needed_contigs:
                    seqs[cur_id] = []
                else:
                    cur_id = None
                continue
            if cur_id is not None:
                seqs[cur_id].append(line.strip())

    return {k: "".join(v) for k, v in seqs.items()}


def load_multi_fasta_for_targets(fasta_paths: List[str], target_contigs: set) -> Tuple[Dict[str, str], List[str]]:
    """
    Loads only target contigs from the first FASTA that contains them.
    We try multiple FASTAs; once a contig is loaded, we do not overwrite it.
    Returns (contig->sequence, used_fasta_paths).
    """
    genome: Dict[str, str] = {}
    used = []
    remaining = set(target_contigs)

    for fp in fasta_paths:
        if not os.path.exists(fp):
            continue
        # fast skip if we've already loaded all needed contigs
        if not remaining:
            break

        subset = load_fasta_subset(fp, remaining)
        if subset:
            used.append(fp)
            for k, v in subset.items():
                genome[k] = v
            remaining -= set(subset.keys())

    return genome, used


def ensure_vcf_index(vcf_path: str) -> None:
    if os.path.exists(vcf_path + ".tbi") or os.path.exists(vcf_path + ".csi"):
        return

    # Try pysam indexing (optionally auto-install)
    try:
        try:
            import pysam  # type: ignore
        except ImportError:
            if AUTO_INSTALL_PYSAM:
                print("[VCF] pysam not installed → attempting auto-install (one-time)...")
                subprocess.run([sys.executable, "-m", "pip", "install", "pysam"], check=True)
                import pysam  # type: ignore
            else:
                raise
        pysam.tabix_index(vcf_path, preset="vcf", force=True)
        return
    except Exception:
        pass

    # Try tabix / bcftools if installed
    if shutil.which("tabix"):
        subprocess.run(["tabix", "-p", "vcf", vcf_path], check=True)
        return
    if shutil.which("bcftools"):
        subprocess.run(["bcftools", "index", "-t", vcf_path], check=True)
        return

    raise RuntimeError(
        f"VCF index missing for {vcf_path} and cannot create one (need pysam OR tabix/bcftools)."
    )


def query_vcf_refalt(contig: str, pos_1based: int, vcf_paths: List[str]) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns (REF, ALT, SOURCE) for a single site by querying indexed VCF(s).
    Uses pysam if available, otherwise uses tabix/bcftools region query.
    """
    region = f"{contig}:{pos_1based}-{pos_1based}"

    # 1) pysam fast path
    try:
        import pysam  # type: ignore
        for vcf in vcf_paths:
            if not os.path.exists(vcf):
                continue
            ensure_vcf_index(vcf)
            vf = pysam.VariantFile(vcf)
            try:
                for rec in vf.fetch(contig, pos_1based - 1, pos_1based):
                    if rec.pos == pos_1based:
                        ref = rec.ref
                        alt = rec.alts[0] if rec.alts else None
                        vf.close()
                        return ref, alt, os.path.basename(vcf)
            finally:
                try:
                    vf.close()
                except Exception:
                    pass
        return None, None, "NOT_FOUND"
    except Exception:
        pass

    # 2) tabix fallback
    if shutil.which("tabix"):
        for vcf in vcf_paths:
            if not os.path.exists(vcf):
                continue
            ensure_vcf_index(vcf)
            cp = subprocess.run(["tabix", vcf, region], capture_output=True, text=True)
            if cp.returncode == 0 and cp.stdout.strip():
                line = cp.stdout.strip().splitlines()[0]
                fields = line.split("\t")
                if len(fields) >= 5:
                    ref = fields[3]
                    alt = fields[4].split(",")[0] if fields[4] else None
                    return ref, alt, os.path.basename(vcf)
        return None, None, "NOT_FOUND"

    # 3) bcftools fallback
    if shutil.which("bcftools"):
        for vcf in vcf_paths:
            if not os.path.exists(vcf):
                continue
            ensure_vcf_index(vcf)
            cp = subprocess.run(["bcftools", "view", "-H", "-r", region, vcf],
                                capture_output=True, text=True)
            if cp.returncode == 0 and cp.stdout.strip():
                line = cp.stdout.strip().splitlines()[0]
                fields = line.split("\t")
                if len(fields) >= 5:
                    ref = fields[3]
                    alt = fields[4].split(",")[0] if fields[4] else None
                    return ref, alt, os.path.basename(vcf)
        return None, None, "NOT_FOUND"

    raise RuntimeError("No pysam/tabix/bcftools available for indexed VCF lookup.")


# =========================
# Input parsing (tag-SNP format)
# =========================
@dataclass
class Target:
    inversion: str
    tag_label: str  # tag1_primary / tag2_redundant
    variant_id: str
    chrom: str
    pos: int
    fasta_chrom: str = ""
    status: str = ""
    r2_overall: Optional[float] = None
    r2_min_cluster: Optional[float] = None
    concordance: Optional[float] = None


def read_tag_targets(input_csv: str) -> List[Target]:
    rows = []
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        is_tag_format = ("tag1_variant_id" in cols) or ("tag2_variant_id" in cols)

        for r in reader:
            inv = r.get("Inversion", "") or r.get("inversion", "") or r.get("INV", "")
            inv = inv.strip()
            if not inv:
                continue

            def pull_float(keys: List[str]) -> Optional[float]:
                for k in keys:
                    if k in r and r[k] not in (None, "", "NA", "nan"):
                        try:
                            return float(r[k])
                        except Exception:
                            return None
                return None

            def pull_str(keys: List[str]) -> str:
                for k in keys:
                    if k in r and r[k]:
                        return str(r[k]).strip()
                return ""

            if is_tag_format:
                for tag_label, key in [("tag1_primary", "tag1_variant_id"), ("tag2_redundant", "tag2_variant_id")]:
                    vid = (r.get(key) or "").strip()
                    if not vid:
                        continue
                    chrom, pos = parse_variant_id(vid)
                    t = Target(
                        inversion=inv,
                        tag_label=tag_label,
                        variant_id=f"{chrom}:{pos}",
                        chrom=chrom,
                        pos=pos,
                        status=pull_str(["Status", "status"]),
                        r2_overall=pull_float(["r2_overall", "r2", "R2_overall"]),
                        r2_min_cluster=pull_float(["r2_min_cluster", "min_cluster", "R2_min_cluster"]),
                        concordance=pull_float(["concordance", "Concordance"]),
                    )
                    rows.append(t)
            else:
                # fallback: assume row already is one-variant-per-row with a 'Variant_ID' column
                vid = (r.get("Variant_ID") or r.get("variant_id") or "").strip()
                if not vid:
                    continue
                chrom, pos = parse_variant_id(vid)
                rows.append(Target(inv, "tag1_primary", f"{chrom}:{pos}", chrom, pos))

    return rows


def read_metrics(metrics_csv: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Returns dict keyed by (Inversion, Variant_ID) -> row dict
    """
    if not metrics_csv or (not os.path.exists(metrics_csv)):
        return {}
    out = {}
    with open(metrics_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            inv = (r.get("Inversion") or r.get("inversion") or "").strip()
            vid = (r.get("Variant_ID") or r.get("variant_id") or r.get("Tag_SNP") or "").strip()
            if not inv or not vid:
                continue
            # normalise to chrom:pos if needed
            try:
                chrom, pos = parse_variant_id(vid)
                vid_norm = f"{chrom}:{pos}"
            except Exception:
                vid_norm = vid
            out[(inv, vid_norm)] = r
    return out


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    ap.add_argument("--merge_metrics_csv", default=DEFAULT_MERGE_METRICS_CSV)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--vcf_paths", default=None, help="Comma-separated VCF .vcf.gz paths (indexed).")
    ap.add_argument("--fasta", default=None, help="Comma-separated FASTA paths (.fa/.fa.gz). Auto-discovered if omitted.")
    ap.add_argument("--flank", type=int, default=DEFAULT_FLANK)
    ap.add_argument("--warn_complexity", type=float, default=DEFAULT_WARN_COMPLEXITY)
    ap.add_argument("--warn_at_rich", type=float, default=DEFAULT_WARN_AT_RICH)
    ap.add_argument("--skip_vcf", action="store_true", default=DEFAULT_SKIP_VCF)
    args = ap.parse_args()

    print("=" * 70)
    print("KASP ASSAY SEQUENCE GENERATOR (v8 - reference-aware, no streaming scan)")
    print("=" * 70)
    print(f"[CONFIG] Flank length: {args.flank} bp")
    print(f"[CONFIG] Skip VCF lookup: {args.skip_vcf}")

    input_csv = args.input_csv
    out_csv = args.out_csv

    if not os.path.exists(input_csv):
        raise SystemExit(f"[ERROR] input_csv not found: {input_csv}")

    # VCF list
    vcf_paths = DEFAULT_VCF_PATHS
    if args.vcf_paths:
        vcf_paths = [p.strip() for p in args.vcf_paths.split(",") if p.strip()]

    # Read targets
    targets = read_tag_targets(input_csv)
    for t in targets:
        # Default: use NCBI contig name; if only GWHABLA FASTA is available, use mapped name
        t.fasta_chrom = NCBI_TO_GWHABLA.get(t.chrom, t.chrom)
    print(f"[MODE] Using input: {input_csv}")
    print(f"[INFO] Total targets: {len(targets)} "
          f"({sum(t.tag_label=='tag1_primary' for t in targets)} primary, "
          f"{sum(t.tag_label=='tag2_redundant' for t in targets)} redundant)")

    # Merge metrics if available
    metrics = read_metrics(args.merge_metrics_csv)
    if metrics:
        print(f"[CONFIG] Merging metrics from: {args.merge_metrics_csv} (records={len(metrics)})")

    # Determine which contigs we need from FASTA
    contigs_needed = set(t.fasta_chrom for t in targets)

    # Discover FASTAs and load only needed contigs
    fasta_paths = []
    if args.fasta:
        fasta_paths = [p.strip() for p in args.fasta.split(",") if p.strip()]
    else:
        fasta_paths = discover_fasta_paths()

    if not fasta_paths:
        raise SystemExit("[ERROR] No FASTA paths found.")

    missing_fasta = [p for p in fasta_paths if not os.path.exists(p)]
    for p in missing_fasta[:3]:
        print(f"[WARN] FASTA not found (skipping): {p}")

    existing_fasta = [p for p in fasta_paths if os.path.exists(p)]
    genome, used_fastas = load_multi_fasta_for_targets(existing_fasta, contigs_needed)
    print(f"[INFO] FASTA sources used: {len(used_fastas)}")
    for p in used_fastas[:5]:
        print(f"       - {p}")

    if not used_fastas:
        # Fail fast with actionable diagnostics
        discovered_existing = [p for p in discover_fasta_paths() if os.path.exists(p)]
        print("[ERROR] No FASTA sources could be opened.")
        print("        Searched directories:")
        for d in FASTA_SEARCH_DIRS:
            print(f"          - {d}")
        if discovered_existing:
            print("        FASTA files found (first 20):")
            for p in discovered_existing[:20]:
                print(f"          - {p}")
            print("        Set DEFAULT_FASTA_CANDIDATES to one of the above (prefer NCBI/RefSeq with NC_ contigs).")
        else:
            print("        No FASTA files were discovered under the search directories.")
        raise FileNotFoundError("No usable FASTA found; cannot extract flanks.")

    # VCF allele lookup (fast, indexed only)
    allele_map: Dict[Tuple[str, int], Tuple[Optional[str], Optional[str], str]] = {}
    if args.skip_vcf:
        print("[VCF] Skipping REF/ALT lookup (--skip_vcf).")
    else:
        print(f"[VCF] Looking up REF/ALT for {len(targets)} target positions (indexed fetch)...")
        # sanity check: require tools / indexes (no streaming)
        if REQUIRE_INDEXED_VCF:
            has_any = any(os.path.exists(v) for v in vcf_paths)
            if not has_any:
                print("[WARN] No VCFs found at DEFAULT_VCF_PATHS; REF/ALT lookup disabled.")
                args.skip_vcf = True
            else:
                try:
                    for v in vcf_paths:
                        if os.path.exists(v):
                            ensure_vcf_index(v)
                except RuntimeError as e:
                    print(f"[WARN] {e}")
                    print("[WARN] REF/ALT lookup disabled. Fix by installing pysam (auto-install may have failed) or install tabix/bcftools.")
                    args.skip_vcf = True

        for i, t in enumerate(targets, 1):
            key = (t.chrom, t.pos)
            if key in allele_map:
                continue
            ref, alt, src = query_vcf_refalt(t.chrom, t.pos, vcf_paths)
            allele_map[key] = (ref, alt, src)
            if i % 10 == 0 or i == len(targets):
                print(f"      ... {i}/{len(targets)}")

    # Build output
    out_rows = []
    skipped = []

    for t in targets:
        # FASTA fetch
        if t.fasta_chrom not in genome:
            skipped.append({
                "Inversion": t.inversion,
                "Tag_Label": t.tag_label,
                "Variant_ID": t.variant_id,
                "Reason": f"Chromosome {t.fasta_chrom} not present in loaded FASTA contigs"
            })
            continue

        seq = genome[t.fasta_chrom]
        pos0 = t.pos - 1
        L = args.flank
        if pos0 - L < 0 or pos0 + L >= len(seq):
            skipped.append({
                "Inversion": t.inversion,
                "Tag_Label": t.tag_label,
                "Variant_ID": t.variant_id,
                "Reason": f"Position {t.pos} out of bounds for FASTA contig (len={len(seq)})"
            })
            continue

        left = seq[pos0 - L:pos0].upper()
        right = seq[pos0 + 1:pos0 + 1 + L].upper()

        ref, alt, src = (None, None, "SKIPPED")
        if not args.skip_vcf:
            ref, alt, src = allele_map.get((t.chrom, t.pos), (None, None, "NOT_FOUND"))

        warn, comp, atf, hom = flank_warnings(left, right, args.warn_complexity, args.warn_at_rich)

        # Overlay metrics if available
        m = metrics.get((t.inversion, t.variant_id), {})

        status = (m.get("Status") or m.get("status") or t.status or "").strip()
        r2o = m.get("r2_overall") or m.get("r2") or ("" if t.r2_overall is None else f"{t.r2_overall:.4f}")
        r2m = m.get("r2_min_cluster") or m.get("min_cluster") or ("" if t.r2_min_cluster is None else f"{t.r2_min_cluster:.4f}")
        conc = m.get("concordance") or ("" if t.concordance is None else f"{t.concordance:.4f}")

        # Represent SNP allele in the flanking sequence
        mid = "N"
        if ref and alt:
            mid = f"[{ref}/{alt}]"

        out_rows.append({
            "Inversion": t.inversion,
            "Tag_Label": t.tag_label,
            "Variant_ID": t.variant_id,
            "Chrom": t.chrom,
            "FASTA_Chrom": t.fasta_chrom,
            "Pos": t.pos,
            "REF": ref or "",
            "ALT": alt or "",
            "VCF_Source": src,
            "Status": status,
            "r2_overall": r2o,
            "r2_min_cluster": r2m,
            "Concordance": conc,
            "Flank_Left": left,
            "Flank_Right": right,
            "Flanking_Sequence": f"{left}{mid}{right}",
            "Flank_Complexity": f"{comp:.3f}",
            "AT_Fraction": f"{atf:.3f}",
            "Homopolymer_MaxRun": hom,
            "Flank_Warning": warn,
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        writer.writeheader()
        writer.writerows(out_rows)

    skipped_csv = os.path.splitext(out_csv)[0] + "_skipped.csv"
    if skipped:
        with open(skipped_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Inversion", "Tag_Label", "Variant_ID", "Reason"])
            writer.writeheader()
            writer.writerows(skipped)

    # Summary
    print("=" * 70)
    print(f"[SUCCESS] {len(out_rows)} assay contexts saved to:\n          {out_csv}")
    if skipped:
        print(f"[INFO] Skipped targets saved to:\n          {skipped_csv}")

    n_with_alleles = sum(1 for r in out_rows if r["REF"] and r["ALT"])
    n_warn = sum(1 for r in out_rows if r["Flank_Warning"])
    print("\n[SUMMARY]")
    print(f"  Rows written: {len(out_rows)}")
    print(f"  With VCF REF/ALT: {n_with_alleles}/{len(out_rows)}")
    print(f"  With warnings: {n_warn}/{len(out_rows)}")
    print(f"  Skipped: {len(skipped)}")
    print("[DONE]")


if __name__ == "__main__":
    main()
