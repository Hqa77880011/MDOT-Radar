# MDOT-Radar

Constrained monotone Doppler-measure transport for cross-aspect gesture recognition.

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

| Component | Implementation / command |
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

Obtain Pantomime through the [author data guide](https://dariush-salami.github.io/posts/pantomime-mid-air-gesture-recognition-with-sparse-millimeter-wave-radar-point-clouds/). Its public pickle layout contains ragged xyz frames with optional intensity. Use `--velocity-column` for an export containing measured Doppler, or `--range-rate-proxy` for nearest-point radial-distance differences. The two options retain distinct measurement definitions.

Create a CSV with `file,id,subject,pair,angle,label,split`. Paths are relative to the CSV, speed units are m/s, angles degrees and labels zero based. `pair` must identify a truly paired acquisition, rather than inventing synchronization between independent gesture repetitions. Only trusted upstream pickle files may use `--trust-pickle`.

```bash
python -m mdot_radar.convert --metadata data/pantomime_trials.csv --output data/pantomime --fps 30 --range-rate-proxy --trust-pickle
python -m mdot_radar.protocol --input data/pantomime/manifest.json --output data/aspect_protocol.json --training-angles -30 0 30 --assign-subjects --seed 17
mdot-radar prepare --config configs/reproduce.json --input data/aspect_protocol.json --data data/reproduce
```

Use the actual recording FPS. Three-column xyz pickles require `--unit-intensity`; missing intensity is then explicitly uniform. For an extended raw export with measured Doppler in column 4, replace `--range-rate-proxy` by `--velocity-column 4`. The protocol assigns subject folds once before filtering training angles. One/three/seven-view protocols change only `--training-angles`; all test views and subjects remain fixed. Wrong-angle tests use the frozen checkpoint. Angle-gap panels can group `samples.csv` by recorded angle without retraining.

All ten variants share a ResNet-18 classifier with 2/2/2/2 residual blocks, widths 64/128/256/512 and a stride-one 3x3 single-channel stem. `classifier_width` controls width. Constrained and unconstrained transports have the same parameter count. `pipeline_latency_ms` includes KDE construction, input transfer, transport and classification when raw returns are retained by `prepare`; `model_latency_ms` covers transport/classification for all input formats. File decompression is excluded.

## Checks

Generated fixtures check data handling, metrics and checkpoint loading. On Windows with separate MKL/OpenMP runtimes, set `$env:MKL_THREADING_LAYER='SEQUENTIAL'` in the active PowerShell process.

```bash
python tests/smoke.py
python tests/native_formats.py
```
