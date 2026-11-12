"""Generate a fictional UCI-shaped retail extract for offline development.

This is synthetic test data, not a copy of the UCI dataset or a live feed.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate(output: Path, seed: int = 42, include_quality_issues: bool = False) -> int:
    rng = random.Random(seed)
    start = datetime(2023, 1, 2, 10)  # Monday
    rows = []
    products = [("SYN001", "SYNTHETIC MUG", 22, 3.50), ("SYN002", "SYNTHETIC BASKET", 35, 6.00), ("SYN003", "SYNTHETIC CANDLE", 15, 4.25), ("SYN004", "SYNTHETIC TOWEL", 28, 8.00)]
    for week in range(44):
        for index, (stock, description, base, price) in enumerate(products):
            season = 5 * math.sin(2 * math.pi * week / 13 + index)
            demand = max(1, round(base + season + week * 0.1 + rng.randint(-3, 3)))
            stamp = start + timedelta(weeks=week, days=index)
            rows.append([str(100000 + week * len(products) + index), stock, description,
                         demand, stamp.strftime("%Y-%m-%d %H:%M:%S"), price,
                         str(20000 + index), "United Kingdom"])
    # A Sunday sale establishes the observed end of the final calendar week.
    rows.append(["999998", "SYN001", "SYNTHETIC MUG", 1,
                 (start + timedelta(weeks=43, days=6)).strftime("%Y-%m-%d %H:%M:%S"),
                 3.50, "20000", "United Kingdom"])
    if include_quality_issues:
        rows.append(rows[20].copy())
        rows.append(["C999999", "SYN001", "SYNTHETIC MUG", -2, "2023-05-01 12:00:00", 3.50, "20000", "United Kingdom"])
        rows.append(["999999", "", "SYNTHETIC BAD ROW", 3, "2023-05-02 12:00:00", 1.00, "20001", "United Kingdom"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/synthetic_online_retail.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-quality-issues", action="store_true", help="Add a duplicate, a return, and a row missing StockCode.")
    args = parser.parse_args()
    count = generate(args.output, args.seed, args.include_quality_issues)
    print(f"Wrote {count} fictional transactions to {args.output}")


if __name__ == "__main__":
    main()
