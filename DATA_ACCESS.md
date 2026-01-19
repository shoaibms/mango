# Data Access

This study re-analyses publicly available data. No new data were generated.  

## Data Sources

| Data | Source | Reference |
|------|--------|-----------|
| Genotypes (VCF) | UQ Research Data Manager | Munyengwa et al. 2025 |
| Phenotypes | New Phytologist Supporting Information (Dataset S1) | Wilkinson et al. 2025 |
| Inversions | New Phytologist Supporting Information (Dataset S2) | Wilkinson et al. 2025 |
| Reference genome | CNCB Genome Warehouse | Wang et al. 2020 |
| Gene annotations | CNCB Genome Warehouse | Wang et al. 2020 |
| External validation summary | Horticulture Research (reported metrics) | Jighly et al. 2026 |

## Download Instructions

### 1) Genotypes (VCF)

**Source:** University of Queensland Research Data Manager (may require institutional access)  
**URL:** https://rdm.uq.edu.au/files/c28a32e6-b688-4afe-a6ef-9ce29e7da472

Download:
- `11_QF1.vcf.gz` (chr 1–10)
- `12_QF2.vcf.gz` (chr 11–20)

Place in: `data/vcf/`

### 2) Phenotypes & Inversions

**Source:** New Phytologist Supporting Information  
**URL:** https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.20252

Download: `nph20252-sup-0001-datasetss1-s3.xlsx`  
Sheets used:
- **Dataset S1:** Phenotypes (FBC, AFW, FF, TC, TSS)
- **Dataset S2:** Inversion genotypes (17 SVs)

Place in: `data/phenotypes/`

### 3) Reference Genome & Annotations

**Source:** CNCB Genome Warehouse  
**URL:** https://download.cncb.ac.cn/gwh/Plants/Mangifera_indica_mangoV1_GWHABLA00000000/

Download:
- `GWHABLA00000000.genome.fasta.gz`
- `GWHABLA00000000.gff.gz`

Place in: `data/reference/`

**Alternative (NCBI):** https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_011075055.1/

### 4) External Validation (Jighly et al.)

**Source:** Horticulture Research  
**URL/DOI:** https://doi.org/10.1093/hr/uhag004

No additional data files are required for our external validation step: `04_validation/02_external_validation.py` uses reported (published) cross-collection performance metrics.

## Verification (recommended)

```bash
# VCF integrity
gunzip -t data/vcf/11_QF1.vcf.gz
gunzip -t data/vcf/12_QF2.vcf.gz

# Reference files
gunzip -t data/reference/GWHABLA00000000.genome.fasta.gz
gunzip -t data/reference/GWHABLA00000000.gff.gz
```

## Expected Directory Structure

```text
data/
├── vcf/
│   ├── 11_QF1.vcf.gz
│   └── 12_QF2.vcf.gz
├── phenotypes/
│   └── nph20252-sup-0001-datasetss1-s3.xlsx
└── reference/
    ├── GWHABLA00000000.genome.fasta.gz
    └── GWHABLA00000000.gff.gz
```

## Phenotype Definitions

| Trait | Description | Unit |
|-------|-------------|------|
| FBC | Fruit blush colour | Score (0–10) |
| AFW | Average fruit weight | g |
| FF | Flesh firmness | kg/cm² |
| TC | Trunk circumference | cm |
| TSS | Total soluble solids | °Brix |

## References

1. Munyengwa, N. et al. (2025) Increased genomic predictive ability in mango using GWAS-preselected variants and fixed-effect SNPs. *Frontiers in Plant Science* 16, 1664012

2. Wilkinson, M.J. et al. (2025) Centromeres are hotspots for chromosomal inversions and breeding traits in mango. *New Phytologist* 245, 899–913

3. Wang, P. et al. (2020) The genome evolution and domestication of tropical fruit mango. *Genome Biology* 21, 60

4. Jighly, A. et al. (2026) Strategic global data integration to improve genomic prediction accuracy in trees breeding programs facing resource limitations, a case study in mango. *Horticulture Research* (online), https://doi.org/10.1093/hr/uhag004
