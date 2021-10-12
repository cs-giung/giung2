import torch


__all__ = [
    "compute_ent",
    "compute_kld",
]


@torch.no_grad()
def compute_ent(confidences: torch.Tensor) -> float:

    ent = -torch.sum(
        confidences * torch.log(1e-12 + confidences), dim=1
    ).mean().item()

    return ent


@torch.no_grad()
def compute_kld(confidences: torch.Tensor) -> float:

    ensemble_size = confidences.size(1)
    if ensemble_size == 1:
        return 0.0

    pairs = []
    for i in range(ensemble_size):
        for j in range(ensemble_size):
            pairs.append((i, j))

    kld = 0.0
    for (i, j) in pairs:
        if i == j:
            continue
        kld += torch.nn.functional.kl_div(
            confidences[:, i, :].log(),
            confidences[:, j, :],
            reduction="sum", log_target=False,
        )

    kld = kld / (ensemble_size * (ensemble_size - 1))
    kld = kld / confidences.size(0)

    return kld
