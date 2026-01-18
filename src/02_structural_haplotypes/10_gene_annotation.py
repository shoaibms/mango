#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_gene_annotation.py

Mango GS – Idea 2
=================
Build a clean, core gene annotation dictionary from the mango GFF3 file.

This script:
  * Parses the reference GFF3 (compressed or plain).
  * Extracts only 'gene' features.
  * Parses attributes (ID, Accession, Name, product/description/Note where present).
  * Produces a tidy CSV with one row per gene, ready for downstream mapping.

Default paths are set for your local mango project.

Outputs
-------
- gene_annotation_core.csv
"""

import os
import sys
import gzip
import argparse
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------
# Defaults (adapt to your local setup if needed)
# ---------------------------------------------------------------------

# NCBI CATAS_Mindica_2.1 genomic GFF (matches VCF: NC_058xxx)
DEFAULT_GFF = r"C:\Users\ms\Desktop\mango\data\ncbi\genomic.gff"

# Where to write the core gene annotation table used by other Idea 2 scripts
DEFAULT_OUTDIR = r"C:\Users\ms\Desktop\mango\output\idea_2\annotation"
DEFAULT_OUTFILE = "gene_annotation_core.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_makedirs(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def open_maybe_gzip(path: str):
    """Open a file that may be gzipped or plain text."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_gff_attributes(attr_str: str) -> Dict[str, str]:
    """
    Parse the 9th GFF column (attributes) into a dict.
    Example: "ID=gene1;Name=abc;Note=something"
    """
    attrs: Dict[str, str] = {}
    if not attr_str:
        return attrs
    for part in attr_str.split(";"):
        if not part:
            continue
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()
            attrs[key] = val
        else:
            # Attribute without '=' (rare); store as a flag
            attrs[part.strip()] = ""
    return attrs


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def build_gene_annotation_dict(
    gff_path: str,
    outdir: str,
    outfile: str = DEFAULT_OUTFILE
) -> str:
    """
    Parse GFF3 and build a tidy gene annotation CSV.

    Returns
    -------
    out_path : str
        Path to the written CSV file.
    """
    print("========================================================================")
    print("Mango GS – Idea 2: Build gene annotation dictionary")
    print("========================================================================")
    print(f"[INFO] GFF input:  {gff_path}")
    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Output file:{outfile}")
    print("")

    if not os.path.exists(gff_path):
        print(f"[ERROR] GFF file not found: {gff_path}", file=sys.stderr)
        sys.exit(1)

    safe_makedirs(outdir)

    rows: List[Dict[str, object]] = []
    n_lines = 0
    n_gene = 0

    with open_maybe_gzip(gff_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            n_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            seqid, source, ftype, start, end, score, strand, phase, attrs_str = parts

            # We only keep gene features here
            if ftype != "gene":
                continue

            n_gene += 1
            attrs = parse_gff_attributes(attrs_str)

            gene_id = attrs.get("ID", "")
            gene_accession = attrs.get("Accession", "")
            gene_name = attrs.get("Name", "")

            # Try to capture any product/description/Note-like fields if present
            product = (
                attrs.get("product")
                or attrs.get("Product")
                or attrs.get("description")
                or attrs.get("Description")
                or attrs.get("Note")
                or ""
            )

            try:
                start_i = int(start)
                end_i = int(end)
            except ValueError:
                start_i, end_i = None, None

            row = {
                "gene_id": gene_id,
                "chr": seqid,
                "start": start_i,
                "end": end_i,
                "length_bp": (end_i - start_i + 1) if (start_i is not None and end_i is not None) else None,
                "strand": strand if strand in {"+", "-"} else ".",
                "source": source,
                "type": ftype,
                "accession": gene_accession,
                "gene_name": gene_name,
                "product": product,
                "raw_attributes": attrs_str,
            }
            rows.append(row)

    if not rows:
        print("[WARN] No 'gene' features found in the GFF. Check the file and feature types.")
    else:
        print(f"[INFO] Processed {n_lines:,} non-comment lines from GFF.")
        print(f"[INFO] Extracted {n_gene:,} 'gene' features.")

    df = pd.DataFrame(rows)

    # Sort by chromosome and start position
    sort_cols = [c for c in ["chr", "start"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    out_path = os.path.join(outdir, outfile)
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Gene annotation dictionary -> {out_path}")
    print(f"[INFO] Total genes in annotation dict: {df.shape[0]:,}")
    print("")
    print("[OK] Gene annotation dictionary built successfully.")

    return out_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a core gene annotation dictionary from mango GFF3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gff",
        type=str,
        default=DEFAULT_GFF,
        help=f"Path to GFF3 annotation (default: {DEFAULT_GFF})",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for annotation table (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=DEFAULT_OUTFILE,
        help="Output CSV file name."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_gene_annotation_dict(
        gff_path=args.gff,
        outdir=args.outdir,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
