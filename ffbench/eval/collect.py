"""Merge several run JSONLs into one aggregated table.

    python -m ffbench.eval.collect ffbench_results/verify_*.jsonl --out ffbench_results/verify_all
"""
from __future__ import annotations

import argparse
import glob

from ffbench.eval.report import aggregate, format_table, read_jsonl, write_csv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", default=None, help="stem for the merged CSV")
    args = ap.parse_args(argv)
    rows = []
    for pattern in args.jsonl:
        for path in sorted(glob.glob(pattern)):
            rows += read_jsonl(path)
    aggs = aggregate(rows)
    print(format_table(aggs))
    if args.out:
        print("csv:", write_csv(aggs, args.out + ".csv"))


if __name__ == "__main__":
    main()
