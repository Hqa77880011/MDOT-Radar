"""Paired radial returns from shared moving-scatterer gesture trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np


def scatterers(rng, gesture, count=1024):
    phase = (np.arange(count) + rng.random(count)) / count
    frequency = [1.0, 2.0, 1.0][gesture]
    velocity = rng.uniform(1.3, 1.9) * np.sin(2 * np.pi * frequency * phase)
    if gesture == 2:
        velocity += 0.45 * np.sin(4 * np.pi * phase + 0.4)
    velocity += rng.normal(0, 0.055, count)
    lateral = 0.18 * np.tanh(velocity)
    intensity = rng.lognormal(0, 0.25, count)
    return velocity, lateral, intensity, phase


def returns(state, angle, rng):
    forward, lateral, intensity, phase = state
    radians = np.deg2rad(angle)
    radial = forward * np.cos(radians) + lateral * np.sin(radians)
    radial += rng.normal(0, 0.02, len(radial))
    return np.column_stack([radial, intensity, phase]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", type=int, default=18)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.subjects < 6:
        parser.error("Use at least six independent subjects")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for subject in range(args.subjects):
        split = "train" if subject < 0.6 * args.subjects else "validation" if subject < 0.8 * args.subjects else "test"
        angles = [-30, 0, 30] if split == "train" else [-45, -30, -15, 0, 15, 30, 45]
        for gesture in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, subject, gesture]))
            state = scatterers(rng, gesture)
            for angle in angles:
                name = f"radar_{subject:03d}_{gesture}_{angle:+03d}.npz"
                np.savez_compressed(output / name, points=returns(state, angle, rng))
                records.append(dict(id=f"{subject}:{gesture}:{angle}", path=name,
                                    subject=f"subject-{subject}", pair=f"{subject}:{gesture}",
                                    angle=angle, label=gesture, split=split))
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(
        source="shared phase/velocity/intensity scatterers projected onto radar line of sight",
        truth="gesture trajectory family; subject-disjoint, canonical zero-angle return retained",
        seed=args.seed, subjects=args.subjects,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
