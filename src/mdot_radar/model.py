import torch
from torch import nn
from torch.nn import functional as F


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def interpolate(values, positions, grid):
    step = grid[1] - grid[0]
    location = ((positions - grid[0]) / step).clamp(0, len(grid) - 1)
    left = location.floor().long().clamp(max=len(grid) - 2)
    fraction = location - left
    return (
        values.gather(-1, left) * (1 - fraction)
        + values.gather(-1, left + 1) * fraction
    )


class Transport(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(
            nn.Linear(4, c["rank"]), nn.Tanh(), nn.Linear(c["rank"], 2 * c["knots"])
        )
        self.register_buffer(
            "grid", torch.linspace(-c["vmax"], c["vmax"], c["velocity_bins"])
        )
        self.register_buffer(
            "knot_grid", torch.linspace(-c["vmax"], c["vmax"], 2 * c["knots"] + 1)
        )

    def mapping(self, source, target, variant="mdot"):
        batch = len(source)
        phase = torch.linspace(0, 1, self.c["phase_bins"], device=source.device)
        angle = (target - source)[:, None].expand(batch, len(phase)) * torch.pi / 180
        phase = phase[None].expand(batch, -1)
        condition = torch.stack(
            [
                angle.sin(),
                angle.cos(),
                (2 * torch.pi * phase).sin(),
                (2 * torch.pi * phase).cos(),
            ],
            dim=-1,
        )
        raw = self.net(condition)
        positive = F.softplus(raw) + 1e-4
        if variant == "no_anchor":
            cumulative = torch.cat(
                [torch.zeros_like(positive[..., :1]), positive.cumsum(-1)], dim=-1
            )
            knots = (
                -self.c["vmax"] + 2 * self.c["vmax"] * cumulative / cumulative[..., -1:]
            )
        elif variant == "unconstrained":
            knots = self.knot_grid + F.pad(raw, (0, 1)) * self.c["vmax"] * 0.2
            knots = knots.clamp(-self.c["vmax"], self.c["vmax"])
        else:
            half = self.c["knots"]
            left = positive[..., :half]
            right = positive[..., half:]
            negative = torch.cat(
                [torch.zeros_like(left[..., :1]), left.cumsum(-1)], dim=-1
            )
            negative = -self.c["vmax"] + self.c["vmax"] * negative / negative[..., -1:]
            positive_part = (
                right.cumsum(-1) / right.sum(-1, keepdim=True) * self.c["vmax"]
            )
            knots = torch.cat([negative, positive_part], dim=-1)
        positions = self.grid[None, None].expand(batch, self.c["phase_bins"], -1)
        result = interpolate(knots, positions, self.knot_grid)
        if variant in ("identity", "angle_token", "coral", "dann"):
            result = positions
        if variant == "cosine":
            scale = (target * torch.pi / 180).cos() / (
                source * torch.pi / 180
            ).cos().clamp_min(0.2)
            result = (positions * scale[:, None, None]).clamp(
                -self.c["vmax"], self.c["vmax"]
            )
        return result, raw

    def forward(self, measure, source, target, variant="mdot"):
        mapping, raw = self.mapping(source, target, variant)
        step = self.grid[1] - self.grid[0]
        location = ((mapping - self.grid[0]) / step).clamp(0, len(self.grid) - 1)
        left = location.floor().long().clamp(max=len(self.grid) - 2)
        fraction = location - left
        result = torch.zeros_like(measure).scatter_add(
            -1, left, measure * (1 - fraction)
        )
        result = result.scatter_add(-1, left + 1, measure * fraction)
        if variant == "nonconserving":
            result = result * (1 + 0.2 * raw.mean(-1).tanh())[..., None]
        return result, mapping


class Classifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
        )
        self.head = nn.Linear(260, classes)

    def features(self, x):
        return self.body(x[:, None])

    def forward(self, x, angles, token=False):
        f = self.features(x)
        radians = angles * torch.pi / 180
        metadata = (
            torch.stack(
                [
                    radians.sin(),
                    radians.cos(),
                    (2 * radians).sin(),
                    (2 * radians).cos(),
                ],
                dim=1,
            )
            if token
            else torch.zeros(len(x), 4, device=x.device)
        )
        return self.head(torch.cat([f, metadata], dim=1))


def wasserstein(left, right, step):
    return (left.cumsum(-1) - right.cumsum(-1)).abs().sum(-1).mean() * step
