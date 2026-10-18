# MDOT-Radar

Implementation of constrained monotone Doppler-measure transport for cross-aspect gesture recognition.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The generator creates radar scattering point clouds for independent subjects and paired aspects. These are generated radar inputs. `source.json` records the scattering process and seed; preparation retains subject and pair IDs.

```bash
python -m mdot_radar.synthetic --output data/generated --subjects 18 --seed 17
mdot-radar prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
```

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
mdot-radar train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
```

## Evaluation

```bash
mdot-radar evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

## Comparisons and ablations

Fit each supported `--variant` into a separate directory, evaluate its own checkpoint, then pass the evaluated run directories to `compare --inputs`.

```bash
mdot-radar compare --inputs runs/reproduce --output outputs/comparison
mdot-radar plot --output outputs/comparison
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `model.pt`, `history.csv`, `samples.csv`, `metrics.json`, `source.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
