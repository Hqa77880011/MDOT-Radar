"""Convert Pantomime per-trial pickle clouds with explicit acquisition metadata."""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .common import save


def point_returns(frames, fps, velocity_column=None, intensity_column=3, proxy=False):
    """Keep measured Doppler, or explicitly derive a nearest-point range-rate proxy."""
    if velocity_column is None and not proxy:
        raise ValueError("Public Pantomime xyz/intensity has no Doppler. Supply a measured velocity column or explicitly select --range-rate-proxy.")
    result = []
    previous = None
    for index, frame in enumerate(frames):
        frame = np.asarray(frame, dtype=float)
        if not len(frame):
            continue
        if frame.ndim != 2 or frame.shape[1] < 3:
            raise ValueError("Each pickle frame must be [points, xyz(+intensity)]")
        xyz = frame[:, :3]
        if velocity_column is not None:
            velocity = frame[:, velocity_column]
        elif previous is not None and len(previous):
            _, nearest = cKDTree(previous).query(xyz)
            velocity = (np.linalg.norm(xyz, axis=1) - np.linalg.norm(previous[nearest], axis=1)) * fps
        else:
            previous = xyz
            continue
        intensity = frame[:, intensity_column] if intensity_column is not None else np.ones(len(frame))
        phase = np.full(len(frame), index / max(len(frames)-1, 1))
        result.append(np.column_stack([velocity, np.maximum(intensity, 0), phase]))
        previous = xyz
    if not result:
        raise ValueError("Trial has no usable velocity observations")
    return np.concatenate(result).astype(np.float32)


def convert(metadata, output, fps, velocity_column=None, intensity_column=3, proxy=False):
    """CSV fields: file,id,subject,pair,angle,label,split; paths relative to CSV."""
    metadata, output = Path(metadata), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, subjects = [], {}
    with metadata.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row["subject"] in subjects and subjects[row["subject"]] != row["split"]:
                raise ValueError("Subject crosses partitions")
            subjects[row["subject"]] = row["split"]
            path = metadata.parent / row["file"]
            # Pickle executes code: the CLI requires explicit trust in the dataset.
            with path.open("rb") as handle:
                frames = pickle.load(handle, encoding="latin1")
            points = point_returns(frames, fps, velocity_column, intensity_column, proxy)
            name = f"trial_{index:06d}.npz"
            np.savez_compressed(output / name, points=points)
            records.append(dict(id=row["id"], subject=row["subject"], pair=row["pair"],
                                angle=float(row["angle"]), label=int(row["label"]),
                                split=row["split"], path=name,
                                velocity_source="range_rate_proxy" if proxy else "measured_doppler"))
    save(output / "manifest.json", records)
    save(output / "conversion.json", dict(source="Pantomime raw per-frame pickle", fps=fps,
         velocity_column=velocity_column, intensity_column=intensity_column,
         velocity_source="nearest-point radial-distance finite difference" if proxy else "measured Doppler"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--velocity-column", type=int)
    p.add_argument("--intensity-column", type=int, default=3)
    p.add_argument("--unit-intensity", action="store_true")
    p.add_argument("--range-rate-proxy", action="store_true")
    p.add_argument("--trust-pickle", action="store_true")
    a = p.parse_args()
    if not a.trust_pickle:
        p.error("--trust-pickle is required for trusted upstream dataset pickle files")
    if a.fps <= 0:
        p.error("fps must be positive")
    convert(a.metadata, a.output, a.fps, a.velocity_column,
            None if a.unit_intensity else a.intensity_column, a.range_rate_proxy)


if __name__ == "__main__":
    main()
