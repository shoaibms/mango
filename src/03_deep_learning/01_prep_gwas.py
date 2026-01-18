import os
import pandas as pd

# ================= CONFIG =================
# Folder where Idea 1 internal GWAS files live
# (adjust if your actual folder name differs)
INPUT_DIR = r"C:\Users\ms\Desktop\mango\output\idea_1\gwas_weights"

# Output combined GWAS summary (what 09_ai_vs_gwas_concordance.py expects)
OUTPUT_FILE = r"C:\Users\ms\Desktop\mango\output\idea_1\gwas\gwas_summary_by_trait.csv"

# Traits we care about (must match y_raw_traits.json and 09_* script)
TRAITS = ["FBC", "FF", "AFW", "TSS", "TC"]

# Column names in internal_gwas_<TRAIT>.csv
SNP_COL = "variant_id"
P_COL = "p_pc"      # PC-corrected p-value
BETA_COL_CANDIDATES = ["beta_pc", "beta"]  # we’ll use first one that exists
# ==========================================


def main():
    print(f"[INFO] Aggregating GWAS results from:\n  {INPUT_DIR}")

    merged_df = None

    for trait in TRAITS:
        fpath = os.path.join(INPUT_DIR, f"internal_gwas_{trait}.csv")
        if not os.path.exists(fpath):
            print(f"[WARN] File not found for trait {trait}:\n  {fpath}")
            continue

        print(f"  - Loading {trait} from:\n    {fpath}")
        df = pd.read_csv(fpath)

        if SNP_COL not in df.columns or P_COL not in df.columns:
            print(
                f"[WARN] {fpath} is missing required columns "
                f"('{SNP_COL}' and/or '{P_COL}'); skipping this trait."
            )
            continue

        keep_cols = [SNP_COL, P_COL]

        # Try to capture an effect size column if present
        beta_col_used = None
        for cand in BETA_COL_CANDIDATES:
            if cand in df.columns:
                beta_col_used = cand
                keep_cols.append(cand)
                break

        df_sub = df[keep_cols].copy()
        df_sub = df_sub.rename(
            columns={
                SNP_COL: "snp_id",
                P_COL: f"p_{trait}",
                **({beta_col_used: f"beta_{trait}"} if beta_col_used else {}),
            }
        )

        if merged_df is None:
            merged_df = df_sub
        else:
            # Outer join on snp_id so we keep all variants across traits
            merged_df = pd.merge(merged_df, df_sub, on="snp_id", how="outer")

    if merged_df is None:
        print("[ERROR] No GWAS files were successfully loaded. Nothing to write.")
        return

    out_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(out_dir, exist_ok=True)

    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Saved combined GWAS summary to:\n  {OUTPUT_FILE}")
    print(f"  Shape: {merged_df.shape[0]} SNPs x {merged_df.shape[1]} cols")
    print("  Columns:", list(merged_df.columns))


if __name__ == "__main__":
    main()
