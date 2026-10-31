"""Small CPU pipeline and transport invariants; no paper performance claim."""
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdot_radar.cli import prepare, train, evaluate
from mdot_radar.model import Transport, Classifier, ReverseGradient, wasserstein


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    c = dict(seed=17, epochs=1, batch_size=4, lr=0.001, phase_bins=4,
             velocity_bins=17, vmax=4., bandwidth=.2, clutter_threshold=.03,
             knots=3, rank=4, classifier_width=4, alignment_weight=1., composition_weight=.1,
             smoothness_weight=.01, device="cpu")
    model = Transport(c)
    x = torch.rand(4, 4, 17)
    x /= x.sum(-1, keepdim=True)
    angles = torch.tensor([-30., 0., 30., 45.])
    out, mapping = model(x, angles, torch.zeros_like(angles))
    assert torch.all(mapping.diff(dim=-1) > 0)
    torch.testing.assert_close(mapping[..., 8], torch.zeros(4, 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(out.sum(-1), x.sum(-1))
    before = model.net[0].weight.detach().clone()
    loss = (out * model.grid).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and model.net[0].weight.grad.abs().sum() > 0
    torch.optim.SGD(model.parameters(), lr=.01).step()
    assert not torch.equal(before, model.net[0].weight)
    backbone=Classifier(2,width=4).eval()
    before=backbone.body[0].weight.detach().clone()
    classification_loss=torch.nn.functional.cross_entropy(backbone(x,angles),torch.tensor([0,1,0,1]))
    classification_loss.backward()
    assert backbone.body[0].weight.grad.abs().sum()>0
    torch.optim.SGD(backbone.parameters(),lr=.01).step()
    assert not torch.equal(before,backbone.body[0].weight)
    v = torch.tensor([1.], requires_grad=True)
    ReverseGradient.apply(v).sum().backward()
    assert v.grad.item() == -1
    assert wasserstein(torch.tensor([[[1., 0., 0.]]]),
                       torch.tensor([[[0., 0., 1.]]]), 1.).item() == 2.
    output = Path("runs/functional-check")
    output.mkdir(parents=True, exist_ok=True)
    with nullcontext(output) as tmp:
        root = Path(tmp)
        records = []
        for subject, split in enumerate(["train", "test"]):
            for label in range(2):
                for angle in [-30, 0, 30]:
                    path = root / f"{subject}_{label}_{angle}.npz"
                    points = np.column_stack([np.linspace(-2, 2, 48) + .1 * label,
                                              np.ones(48), np.linspace(0, 1, 48)])
                    np.savez(path, points=points)
                    records.append(dict(id=path.stem, subject=str(subject), pair=f"{subject}:{label}",
                                        split=split, path=path.name, label=label, angle=angle))
        manifest = root / "input.json"
        manifest.write_text(json.dumps(records), encoding="utf-8")
        a = SimpleNamespace(input=str(manifest), data=str(root / "data"),
                            output=str(root / "run"), variant="mdot", wrong_angle=0.)
        prepare(a, c)
        for variant in ["mdot", "identity", "angle_token", "cosine", "coral", "dann",
                        "unconstrained", "no_anchor", "no_composition", "nonconserving"]:
            a.variant = variant
            a.output = str(root / variant)
            train(a, c)
            a.checkpoint = str(Path(a.output) / "model.pt")
            state = torch.load(a.checkpoint, weights_only=True)
            first, second = Transport(c).eval(), Transport(c).eval()
            first.load_state_dict(state["transport"])
            second.load_state_dict(state["transport"])
            torch.testing.assert_close(first(x, angles, angles * 0, variant)[0],
                                       second(x, angles, angles * 0, variant)[0])
            head1,head2=Classifier(2,width=4).eval(),Classifier(2,width=4).eval()
            head1.load_state_dict(state['classifier']);head2.load_state_dict(state['classifier'])
            torch.testing.assert_close(head1(x,angles),head2(x,angles),atol=1e-7,rtol=1e-6)
            evaluate(a, c)
            metrics = json.loads((Path(a.output) / "metrics.json").read_text())
            assert np.isfinite(metrics["accuracy"]) and metrics["mass_error"] is not None
    print("PASS: 10 variants, data/train/eval, gradients/update, checkpoint, W1 and constraints")


if __name__ == "__main__":
    main()
