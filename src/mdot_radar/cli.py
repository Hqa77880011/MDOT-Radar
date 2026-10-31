import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import Transport, Classifier, ReverseGradient, interpolate, wasserstein
from .common import load, save, table, plot


def measure(points, c):
    grid = np.linspace(-c["vmax"], c["vmax"], c["velocity_bins"])
    result = np.zeros((c["phase_bins"], len(grid)), np.float32)
    velocity, intensity, phase = points.T
    valid = (
        (np.abs(velocity) >= c["clutter_threshold"])
        & (intensity >= 0)
        & (phase >= 0)
        & (phase <= 1)
    )
    velocity, intensity, phase = velocity[valid], intensity[valid], phase[valid]
    for i in range(c["phase_bins"]):
        selected = (
            np.minimum((phase * c["phase_bins"]).astype(int), c["phase_bins"] - 1) == i
        )
        kernel = np.exp(
            -0.5 * ((grid[None] - velocity[selected, None]) / c["bandwidth"]) ** 2
        )
        result[i] = (kernel * intensity[selected, None]).sum(0)
        result[i] /= max(float(result[i].sum()), 1e-12)
    return result


def prepare(a, c):
    records = load(a.input)
    subjects = {}
    result = []
    output = Path(a.data)
    output.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        if r["subject"] in subjects and subjects[r["subject"]] != r["split"]:
            raise ValueError("Subject crosses splits")
        subjects[r["subject"]] = r["split"]
        path = Path(r["path"])
        path = path if path.is_absolute() else Path(a.input).parent / path
        raw = np.load(path)
        array = raw["measure"] if "measure" in raw else measure(raw["points"], c)
        if array.shape != (c["phase_bins"], c["velocity_bins"]):
            raise ValueError("Measure shape differs from configured bins")
        target = output / f"measure_{i:06d}.npz"
        np.savez_compressed(target, measure=array)
        result.append(dict(r, path=str(target.resolve())))
    save(output / "manifest.json", result)
    save(output / "preprocessing.json", c)


def batches(records, c, rng=None):
    order = np.arange(len(records))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), c["batch_size"]):
        selected = [records[i] for i in order[start : start + c["batch_size"]]]
        pp = torch.tensor(
            np.stack([np.load(r["path"])["measure"] for r in selected]),
            device=c["device"],
        )
        angles = torch.tensor(
            [r["angle"] for r in selected], dtype=torch.float32, device=c["device"]
        )
        labels = torch.tensor(
            [r["label"] for r in selected], dtype=torch.long, device=c["device"]
        )
        yield selected, pp, angles, labels


def train(a, c):
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == "train"]
    classes = max(r["label"] for r in records) + 1
    transport = Transport(c).to(c["device"])
    classifier = Classifier(classes).to(c["device"])
    domain_angles = sorted({r["angle"] for r in records})
    discriminator = nn.Linear(256, len(domain_angles)).to(c["device"])
    opt = torch.optim.Adam(
        list(transport.parameters()) + list(classifier.parameters())
        + (list(discriminator.parameters()) if a.variant == "dann" else []),
        lr=c["lr"],
    )
    history = []
    for epoch in range(c["epochs"]):
        total = 0.0
        count = 0
        for selected, pp, angles, labels in batches(records, c, rng):
            canonical, mapping = transport(
                pp, angles, torch.zeros_like(angles), a.variant
            )
            loss = nn.functional.cross_entropy(
                classifier(canonical, angles, a.variant == "angle_token"), labels
            )
            pairs = []
            for r in selected:
                partners = [
                    q
                    for q in records
                    if q["pair"] == r["pair"] and q["angle"] != r["angle"]
                ]
                if not partners and len(domain_angles) == 1:
                    partners = [r]
                if not partners:
                    raise ValueError(
                        "Each training item needs another angle with the same pair ID"
                    )
                pairs.append(partners[int(rng.integers(len(partners)))])
            target = torch.tensor(
                np.stack([np.load(q["path"])["measure"] for q in pairs]),
                device=c["device"],
            )
            target_angle = torch.tensor(
                [q["angle"] for q in pairs], device=c["device"], dtype=torch.float32
            )
            if a.variant == "dann":
                domains = torch.tensor(
                    [domain_angles.index(r["angle"]) for r in selected],
                    device=c["device"],
                )
                loss = loss + c["alignment_weight"] * nn.functional.cross_entropy(
                    discriminator(ReverseGradient.apply(classifier.features(pp))), domains
                )
            elif a.variant == "coral":
                f = classifier.features(pp)
                g = classifier.features(target)
                centered = lambda x: x - x.mean(0, keepdim=True)
                loss = (
                    loss
                    + c["alignment_weight"]
                    * (centered(f).T @ centered(f) - centered(g).T @ centered(g))
                    .square()
                    .mean()
                    / max(len(f) - 1, 1) ** 2
                )
            elif a.variant not in ("identity", "angle_token", "cosine"):
                aligned, t12 = transport(pp, angles, target_angle, a.variant)
                loss = loss + c["alignment_weight"] * wasserstein(
                    aligned, target, 2 * c["vmax"] / (c["velocity_bins"] - 1)
                )
                t23, _ = transport.mapping(
                    target_angle, torch.zeros_like(angles), a.variant
                )
                composed = interpolate(t23, t12, transport.grid)
                if a.variant != "no_composition":
                    loss = (
                        loss
                        + c["composition_weight"] * (composed - mapping).square().mean()
                    )
                loss = (
                    loss
                    + c["smoothness_weight"]
                    * (mapping[..., 2:] - 2 * mapping[..., 1:-1] + mapping[..., :-2])
                    .square()
                    .mean()
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(selected)
            count += len(selected)
        history.append(dict(epoch=epoch + 1, loss=total / count))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            transport=transport.cpu().state_dict(),
            classifier=classifier.cpu().state_dict(),
            classes=classes,
            config=c,
            variant=a.variant,
        ),
        output / "model.pt",
    )
    table(output / "history.csv", history)


def evaluate(a, c):
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = state["config"]
    variant = state["variant"]
    transport = Transport(c).to(c["device"])
    classifier = Classifier(state["classes"]).to(c["device"])
    transport.load_state_dict(state["transport"])
    classifier.load_state_dict(state["classifier"])
    transport.eval()
    classifier.eval()
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == "test"]
    rows = []
    with torch.no_grad():
        for selected, pp, angles, labels in batches(records, c):
            if str(c["device"]).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            used = angles + a.wrong_angle
            canonical, t13 = transport(pp, used, torch.zeros_like(used), variant)
            logits = classifier(canonical, used, variant == "angle_token")
            if str(c["device"]).startswith("cuda"):
                torch.cuda.synchronize()
            latency = (time.perf_counter() - start) * 1000 / len(selected)
            middle = used / 2
            t12, _ = transport.mapping(used, middle, variant)
            t23, _ = transport.mapping(middle, torch.zeros_like(used), variant)
            composition = (
                (interpolate(t23, t12, transport.grid) - t13)
                .square()
                .mean((1, 2))
                .cpu()
                .numpy()
            )
            mass = (canonical.sum(-1) - pp.sum(-1)).abs().mean(1).cpu().numpy()
            for i, r in enumerate(selected):
                partners = [
                    q for q in records if q["pair"] == r["pair"] and q["angle"] == 0
                ]
                w1 = None
                if partners:
                    target = torch.tensor(
                        np.load(partners[0]["path"])["measure"], device=c["device"]
                    )[None]
                    w1 = float(
                        wasserstein(
                            canonical[i : i + 1],
                            target,
                            2 * c["vmax"] / (c["velocity_bins"] - 1),
                        )
                    )
                rows.append(
                    dict(
                        id=r["id"],
                        subject=r["subject"],
                        angle=r["angle"],
                        method=variant,
                        correct=bool(logits[i].argmax() == labels[i]),
                        w1=w1,
                        composition_error=float(composition[i]),
                        mass_error=float(mass[i]),
                        latency_ms=latency,
                    )
                )
    summary = dict(
        method=variant,
        accuracy=np.mean([
            np.mean([r["correct"] for r in rows if r["angle"] == angle])
            for angle in {r["angle"] for r in rows}
        ]),
        sample_accuracy=np.mean([r["correct"] for r in rows]),
        worst_angle_accuracy=min(
            np.mean([r["correct"] for r in rows if r["angle"] == angle])
            for angle in {r["angle"] for r in rows}
        ),
        composition_error=np.mean([r["composition_error"] for r in rows]),
        mass_error=np.mean([r["mass_error"] for r in rows]),
        model_latency_ms=np.mean([r["latency_ms"] for r in rows]),
        w1=np.mean([r["w1"] for r in rows if r["w1"] is not None])
        if any(r["w1"] is not None for r in rows)
        else None,
    )
    table(Path(a.output) / "samples.csv", rows)
    save(Path(a.output) / "metrics.json", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/radar")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--checkpoint")
    p.add_argument("--inputs", nargs="*")
    p.add_argument("--wrong-angle", type=float, default=0.0)
    p.add_argument(
        "--variant",
        choices=[
            "mdot",
            "identity",
            "angle_token",
            "cosine",
            "coral",
            "dann",
            "unconstrained",
            "no_anchor",
            "no_composition",
            "nonconserving",
        ],
        default="mdot",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [load(Path(d) / "metrics.json") for d in a.inputs]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
