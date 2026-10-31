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
| Identity ResNet, angle token, cosine scaling, CORAL, DANN | `train --variant identity/angle_token/cosine/coral/dann` (choose one) |
| Broken constraints | `unconstrained`, `no_anchor`, `no_composition`, `nonconserving` |
| Aspect-macro accuracy, worst angle, W1, mass/path error | `evaluate`; `sample_accuracy` is also retained |
| Wrong metadata | `evaluate --wrong-angle 30` |
| Figures / independent seeds | `compare`, `plot`, `mdot-radar-report --inputs ... --output ...` |

Each `--variant` needs its own training checkpoint. DANN uses a gradient-reversal angle-domain head over the same 512-dimensional ResNet features (default width 64); the domain head is only used during training. All learned transport variants have the same spline parameter count. A single-view training manifest uses self-alignment (there are no measured cross-angle pairs); other protocols require paired returns at different angles. Create single/few-view protocols by filtering only training-aspect records, retaining the subject split and the common test set.

## Native Pantomime and aspect protocols

The [Pantomime author data guide](https://dariush-salami.github.io/posts/pantomime-mid-air-gesture-recognition-with-sparse-millimeter-wave-radar-point-clouds/) describes per-trial pickle arrays of ragged frames containing xyz, optionally intensity. That public layout does **not** contain measured Doppler. The converter never silently calls xyz/intensity a velocity measurement. A radar export that retains measured Doppler can supply its zero-based `--velocity-column`; otherwise the explicit `--range-rate-proxy` computes nearest-point radial-distance differences. The proxy changes the measurement definition and cannot reproduce the paper's measured-Doppler experiment by itself.

Create a CSV with `file,id,subject,pair,angle,label,split`. Paths are relative to the CSV, speed units are m/s, angles degrees and labels zero based. `pair` must identify a truly paired acquisition, rather than inventing synchronization between independent gesture repetitions. Only trusted upstream pickle files may use `--trust-pickle`.

```bash
python -m mdot_radar.convert --metadata data/pantomime_trials.csv --output data/pantomime --fps 30 --range-rate-proxy --trust-pickle
python -m mdot_radar.protocol --input data/pantomime/manifest.json --output data/aspect_protocol.json --training-angles -30 0 30 --assign-subjects --seed 17
mdot-radar prepare --config configs/reproduce.json --input data/aspect_protocol.json --data data/reproduce
```

Use the actual recording FPS. Three-column xyz pickles require `--unit-intensity`; missing intensity is then explicitly uniform. For an extended raw export with measured Doppler in column 4, replace `--range-rate-proxy` by `--velocity-column 4`. The protocol assigns subject folds once before filtering training angles. One/three/seven-view protocols change only `--training-angles`; all test views and subjects remain fixed. Wrong-angle tests use the frozen checkpoint. Angle-gap panels can group `samples.csv` by recorded angle without retraining.

All ten variants share an actual ResNet-18 classifier (2/2/2/2 basic residual blocks, default widths 64/128/256/512, a stride-one 3x3 single-channel stem for the short phase axis). `classifier_width` is configurable and set to four only by the tiny functional test. The manuscript does not fix its exact ResNet depth/stem, so these architecture values are explicit implementation choices. The constrained/unconstrained transports share parameter count. `pipeline_latency_ms` times KDE construction, input transfer, transport and classifier in the evaluation process; file decompression is excluded. It is available when raw point returns are retained by `prepare`; `model_latency_ms` separately times transport/classification for every input format.

## Verification

Run `python tests/smoke.py` and `python tests/native_formats.py` from the repository root. On 2026-11-01 this passed with Python 3.13.5, numpy 2.1.3, torch 2.7.1+cu118 (CPU, one thread), scipy 1.15.3 and matplotlib 3.10.0. It reads synthetic point-return NPZs, prepares disjoint subjects, trains and evaluates all ten variants for one epoch, checks finite losses and transport parameter gradients/updates, conservation, monotonicity, zero anchoring, checkpoint output equality and a hand-computed W1=2 case. Generated functional-check outputs stay in ignored `runs/functional-check/`. The native-format check validates xyz/intensity rejection, an explicit known range-rate proxy, measured-velocity extraction and frozen one-view splits. These are functional fixtures; full Pantomime training was not run.
