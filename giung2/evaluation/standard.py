import torch
import numpy as np
from scipy.optimize import minimize


__all__ = [
    "evaluate_acc",
    "evaluate_nll",
    "evaluate_bs",
    "get_optimal_temperature",
]


@torch.no_grad()
def evaluate_acc(confidences: torch.Tensor,
                 true_labels: torch.Tensor) -> float:

    acc = torch.max(
        confidences, dim=1
    )[1].eq(true_labels).float().mean().item()

    return acc


@torch.no_grad()
def evaluate_nll(confidences: torch.Tensor,
                 true_labels: torch.Tensor) -> float:

    nll = torch.nn.functional.nll_loss(
        torch.log(1e-12 + confidences), true_labels
    ).item()

    return nll


@torch.no_grad()
def evaluate_bs(confidences: torch.Tensor,
                true_labels: torch.Tensor) -> float:

    targets = torch.eye(
        confidences.size(1), device=confidences.device
    )[true_labels].long()

    bs = torch.sum(
        (confidences - targets)**2, dim=1
    ).mean().item()

    return bs


@torch.no_grad()
def get_optimal_temperature(confidences: torch.Tensor,
                            true_labels: torch.Tensor) -> float:

    def obj(t):
        target = true_labels.cpu().numpy()
        return -np.log(
            1e-12 + np.exp(
                torch.log_softmax(
                    torch.log(
                        1e-12 + confidences
                    ) / t, dim=1
                ).data.numpy()
            )[np.arange(len(target)), target]
        ).mean()

    optimal_temperature = minimize(
        obj, 1.0, method="nelder-mead", options={"xtol": 1e-3}
    ).x[0]

    return optimal_temperature
