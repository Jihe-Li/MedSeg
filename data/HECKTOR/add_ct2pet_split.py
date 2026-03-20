#!/usr/bin/env python3
import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def get_center_from_pid(pid: str) -> str:
    pid = str(pid).strip()
    if "-" in pid:
        return pid.split("-", 1)[0]
    return pid


def allocate_train_counts(center_to_indices: dict, train_total: int) -> dict:
    center_counts = {center: len(indices) for center, indices in center_to_indices.items()}
    total = sum(center_counts.values())
    if total == 0:
        return {center: 0 for center in center_to_indices}

    train_total = min(train_total, total)
    if train_total == total:
        return center_counts

    raw_quota = {
        center: (count / total) * train_total for center, count in center_counts.items()
    }
    allocated = {
        center: min(center_counts[center], math.floor(raw_quota[center]))
        for center in center_counts
    }

    remaining = train_total - sum(allocated.values())
    while remaining > 0:
        candidate_centers = [
            center
            for center in center_counts
            if allocated[center] < center_counts[center]
        ]
        if not candidate_centers:
            break

        candidate_centers.sort(
            key=lambda c: (
                raw_quota[c] - math.floor(raw_quota[c]),
                center_counts[c] - allocated[c],
                c,
            ),
            reverse=True,
        )
        for center in candidate_centers:
            if remaining == 0:
                break
            if allocated[center] < center_counts[center]:
                allocated[center] += 1
                remaining -= 1

    return allocated


def add_ct2pet_split(
    input_csv: Path,
    output_csv: Path,
    train_total: int,
    seed: int,
    split_col: str = "ct2pet",
) -> None:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    required_cols = {"PID", "CT", "PT"}
    missing = required_cols - set(fieldnames)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    eligible_by_center = defaultdict(list)
    for idx, row in enumerate(rows):
        if parse_bool(row.get("CT", "")) and parse_bool(row.get("PT", "")):
            center = get_center_from_pid(row.get("PID", ""))
            eligible_by_center[center].append(idx)

    alloc = allocate_train_counts(eligible_by_center, train_total)
    rng = random.Random(seed)
    train_indices = set()
    for center, indices in eligible_by_center.items():
        n_pick = alloc.get(center, 0)
        if n_pick > 0:
            train_indices.update(rng.sample(indices, n_pick))

    for idx, row in enumerate(rows):
        is_eligible = parse_bool(row.get("CT", "")) and parse_bool(row.get("PT", ""))
        if not is_eligible:
            row[split_col] = "None"
        elif idx in train_indices:
            row[split_col] = "train"
        else:
            row[split_col] = "validation"

    output_fields = fieldnames if split_col in fieldnames else fieldnames + [split_col]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    n_eligible = sum(
        1 for row in rows if parse_bool(row.get("CT", "")) and parse_bool(row.get("PT", ""))
    )
    n_train = sum(1 for row in rows if row.get(split_col) == "train")
    n_val = sum(1 for row in rows if row.get(split_col) == "validation")
    n_none = sum(1 for row in rows if row.get(split_col) == "None")

    print(f"Saved: {output_csv}")
    print(f"Eligible(CT/PT=True): {n_eligible}")
    print(f"train: {n_train}, validation: {n_val}, None: {n_none}")
    print("Train counts by center:")
    for center in sorted(eligible_by_center):
        print(f"  {center}: {alloc.get(center, 0)} / {len(eligible_by_center[center])}")


def add_cross_validation_split(
    input_csv: Path,
    output_csv: Path,
    seed: int = 42,
    cv_col: str = "cross-validation",
    n_folds: int = 5,
) -> None:
    if n_folds <= 1:
        raise ValueError(f"n_folds must be > 1, got {n_folds}")

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    required_cols = {"PID", "CT", "PT", "GTV"}
    missing = required_cols - set(fieldnames)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    eligible_by_center = defaultdict(list)
    for idx, row in enumerate(rows):
        if (
            parse_bool(row.get("CT", ""))
            and parse_bool(row.get("PT", ""))
            and parse_bool(row.get("GTV", ""))
        ):
            center = get_center_from_pid(row.get("PID", ""))
            eligible_by_center[center].append(idx)

    rng = random.Random(seed)
    fold_by_index = {}
    for center, indices in eligible_by_center.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for pos, idx in enumerate(shuffled):
            fold_by_index[idx] = pos % n_folds

    for idx, row in enumerate(rows):
        row[cv_col] = str(fold_by_index[idx]) if idx in fold_by_index else ""

    output_fields = fieldnames if cv_col in fieldnames else fieldnames + [cv_col]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    eligible_total = len(fold_by_index)
    fold_counts = defaultdict(int)
    for fold in fold_by_index.values():
        fold_counts[fold] += 1

    print(f"Saved: {output_csv}")
    print(f"Eligible(CT/PT/GTV=True): {eligible_total}")
    print("Fold counts:")
    for fold in range(n_folds):
        print(f"  {fold}: {fold_counts[fold]}")
    print("Eligible counts by center:")
    for center in sorted(eligible_by_center):
        print(f"  {center}: {len(eligible_by_center[center])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add a ct2pet split column to HECKTOR metadata: "
            "sample train cases proportionally by center among CT/PT eligible rows."
        )
    )
    parser.add_argument(
        "--add-cross-validation",
        action="store_true",
        help=(
            "Add a cross-validation column (0..n_folds) for rows with "
            "CT/PT/GTV=True, stratified by center."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/HECKTOR/meta_data_hecktor.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, overwrite input file.",
    )
    parser.add_argument(
        "--train-total",
        type=int,
        default=626,
        help="Number of train samples among CT/PT=True cases.",
    )
    parser.add_argument(
        "--validation-total",
        type=int,
        default=None,
        help=(
            "Optional number of validation samples among CT/PT=True cases. "
            "If provided, train-total will be inferred as eligible - validation-total."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--cv-col",
        type=str,
        default="cross-validation",
        help="Column name for cross-validation assignment.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation split.",
    )
    args = parser.parse_args()

    output_csv = args.output if args.output is not None else args.input

    if args.add_cross_validation:
        add_cross_validation_split(
            input_csv=args.input,
            output_csv=output_csv,
            seed=args.seed,
            cv_col=args.cv_col,
            n_folds=args.n_folds,
        )
        return

    train_total = args.train_total
    if args.validation_total is not None:
        with args.input.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_eligible = sum(
            1
            for row in rows
            if parse_bool(row.get("CT", "")) and parse_bool(row.get("PT", ""))
        )
        if args.validation_total < 0 or args.validation_total > n_eligible:
            raise ValueError(
                f"validation-total must be in [0, {n_eligible}], got {args.validation_total}"
            )
        train_total = n_eligible - args.validation_total

    add_ct2pet_split(args.input, output_csv, train_total, args.seed)


if __name__ == "__main__":
    main()
