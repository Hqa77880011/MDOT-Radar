# MDOT-Radar

PyTorch implementation of constrained monotone Doppler-measure transport for cross-aspect gesture recognition.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Radar inputs

Export radar returns to NPZ `points` [return,3] with columns signed radial velocity (m/s), nonnegative intensity and normalized motion phase [0,1]. Alternatively provide precomputed `measure` [phase,velocity] matching the configured bins. The JSON manifest has `id,path,subject,pair,angle,label,split`. `pair` identifies the same subject/gesture/repetition across aspects; class/subject IDs never enter the transport network. Subject groups cannot cross splits. Set held-out angles and few-view training sets **before** writing the manifest. The [Pantomime dataset and source](https://zenodo.org/records/4459969) provide the radar acquisitions. Use the documented seven-angle subset from -45 to 45 degrees in 15-degree steps, retaining participant and repetition IDs. This repository consumes exported point clouds and preserves those grouping fields.

```bash
mdot-radar prepare --input data/manifest.json --data data/radar
mdot-radar train --data data/radar --variant mdot --output runs/full
mdot-radar evaluate --data data/radar --checkpoint runs/full/model.pt --output runs/full
mdot-radar compare --inputs runs/full --output outputs/comparison
mdot-radar plot --output outputs/comparison
```

## Transport and comparisons

Preparation rejects clutter near zero velocity, applies fixed Gaussian kernels and normalizes retained intensity within each phase. The low-rank angle-difference/phase network predicts positive spline increments. Separate negative and positive halves enforce zero anchoring and bounded support. Linear scatter interpolation pushes mass to the fixed grid without renormalizing away mass errors. Training combines canonical classification, paired-aspect W1, three-angle path composition and velocity smoothness.

Train separate controls using `identity`, `angle_token`, `cosine`, `coral`, `unconstrained`, `no_anchor`, `no_composition` and `nonconserving`. They share the classifier, measure grid and optimizer. The unconstrained map retains the same knot-network parameter budget but removes monotonicity. Angle-token receives the same angle metadata; CORAL aligns pre-head covariance. Evaluate the selected model with `--wrong-angle 30` for metadata sensitivity. A source-domain adversarial DANN trainer is not substituted by one of these controls.

## Parameters and outputs

`configs/default.json` controls phase/velocity bins, velocity support, clutter cutoff, kernel bandwidth, knots, conditioning rank, loss weights, optimizer, seed and device. Every training item needs a paired training aspect; canonical diagnostic W1 requires a test item at angle zero with the same pair ID. Missing canonical pairs produce null W1, rather than a synthetic target. Training writes `model.pt` and loss history; evaluation writes `samples.csv` and `metrics.json` with accuracy, worst-angle accuracy, W1, composition MSE, absolute mass error and synchronized canonicalization/classification latency. Latency excludes file reading and offline measure construction.

Create separate manifests for single-view, held-out-angle, angle-gap and one/three/seven-view protocols, preserving subject independence. Each run's per-angle/subject measurements support those groupings. Comparison CSV/JSON and PNG/SVG plots use generated results; measures, model weights and raw acquisitions remain outside Git.

## Registered compact reproduction setting

Shared moving scatterers are projected onto each radar line of sight, producing signed velocity/intensity/phase points. Training uses three aspects; independent test subjects retain all seven aspects and a simulated zero-angle counterpart. The existing point-to-measure preprocessor remains active.

Generate the declared input process, preprocess it, then use the existing method entry points:

```bash
python -m mdot_radar.synthetic --output data/generated --subjects 18 --seed 17
mdot-radar prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
mdot-radar train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
mdot-radar evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

`source.json` records the data-generating assumptions and seed. The configuration records `input_source`. All inputs for this selected setting are generated locally using installed package dependencies.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
mdot-radar-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
