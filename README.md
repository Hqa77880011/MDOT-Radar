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

## Method and metric entry points

| Paper component | Implementation / command |
|---|---|
| Phase KDE, clutter filtering, subject isolation | `cli.measure`, `prepare` |
| Positive-increment zero-anchored spline and conservative push-forward | `model.Transport` |
| W1, composition and curvature losses + classification | `train --variant mdot` |
| Identity CNN, angle token, cosine scaling, CORAL, DANN | `train --variant identity/angle_token/cosine/coral/dann` (choose one) |
| Broken constraints | `unconstrained`, `no_anchor`, `no_composition`, `nonconserving` |
| Aspect-macro accuracy, worst angle, W1, mass/path error | `evaluate`; `sample_accuracy` is also retained |
| Wrong metadata | `evaluate --wrong-angle 30` |
| Figures / independent seeds | `compare`, `plot`, `mdot-radar-report --inputs ... --output ...` |

Each `--variant` needs its own training checkpoint. DANN uses a gradient-reversal angle-domain head over the same 256-dimensional CNN features; the domain head is only used during training. All learned transport variants have the same spline parameter count. A single-view training manifest uses self-alignment (there are no measured cross-angle pairs); other protocols require paired returns at different angles. Create single/few-view protocols by filtering only training-aspect records, retaining the subject split and the common test set.

External Pantomime inputs must be converted under its research terms to the same manifest/NPZ format: `points` [N,3] = radial velocity (m/s), nonnegative intensity, normalized phase; manifest fields `id,path,subject,pair,angle,label,split`. Alternatively provide `measure` [phase_bins,velocity_bins] with the frozen preprocessing. The paper's original dataset conversion, experiment checkpoints and locked results are not bundled. The shared classifier here is a compact two-convolution CNN, not the paper's ResNet baseline; widths and numeric hyperparameters absent from the manuscript are explicit implementation choices in the JSON configurations. `model_latency_ms` excludes raw measure construction and disk I/O, and therefore is not the paper's complete-pipeline runtime. No paper performance is claimed by generated data.

## Verification

Run `python tests/smoke.py` from the repository root. On 2026-11-01 this passed with Python 3.13.5, numpy 2.1.3, torch 2.7.1+cu118 (CPU, one thread), scipy 1.15.3 and matplotlib 3.10.0. It reads synthetic point-return NPZs, prepares disjoint subjects, trains and evaluates all ten variants for one epoch, checks finite losses and transport parameter gradients/updates, conservation, monotonicity, zero anchoring, checkpoint output equality and a hand-computed W1=2 case. Generated functional-check outputs stay in ignored `runs/functional-check/`. These checks validate code paths, not Pantomime performance; full training and real-data protocols were not run.
