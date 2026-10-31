"""Freeze subject folds before changing training-aspect coverage."""
import argparse
from pathlib import Path

from .common import load, save, grouped_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--training-angles", nargs="+", type=float, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--assign-subjects", action="store_true")
    a = p.parse_args()
    records = load(a.input)
    roles = grouped_split([r['subject'] for r in records], a.seed) if a.assign_subjects else [r['split'] for r in records]
    result = []
    subjects = {}
    for row, split in zip(records, roles):
        if row['subject'] in subjects and subjects[row['subject']] != split:
            raise ValueError("Subject crosses folds before angle filtering")
        subjects[row['subject']] = split
        if split == 'train' and row['angle'] not in a.training_angles:
            continue
        path = Path(row['path'])
        result.append(dict(row, split=split, path=str((Path(a.input).parent/path).resolve())))
    if not any(r['split']=='train' for r in result):
        raise ValueError("No training observations at requested aspects")
    save(a.output, result)


if __name__ == "__main__":
    main()
