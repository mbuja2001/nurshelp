#!/usr/bin/env python3
"""
Download chaoyi-wu/PMC-CaseReport from Hugging Face and save as Parquet
Usage:
  python download_pmc_case_reports.py --output pmc_case_reports_full.parquet

Requires: datasets, pandas, pyarrow
"""

from pathlib import Path
import argparse


def main(dataset_id: str, output_path: Path):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please install the 'datasets' package: pip install datasets pandas pyarrow") from e

    print(f"🚀 Downloading {dataset_id} from Hugging Face...")
    ds = load_dataset(dataset_id, split="train", trust_remote_code=True)

    print("Converting to pandas DataFrame (this may use a lot of memory)...")
    df = ds.to_pandas()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(df)} rows to: {output_path}")
    df.to_parquet(output_path, index=False)

    print("✅ Success! Parquet saved.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='chaoyi-wu/PMC-CaseReport', help='Hugging Face dataset id')
    p.add_argument('--output', default='pmc_case_reports_full.parquet', help='Output Parquet path')
    args = p.parse_args()

    main(args.dataset, Path(args.output))
