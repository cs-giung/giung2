import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Tuple


__all__ = [
    "build_sghmc_optimizer",
]


class SGHMC(Optimizer):

    def __init__(self, params, lr, alpha=0.9, lr_scale=1.0, weight_decay=0.0, temperature=1.0) -> None:
        if lr < 0.0:
            raise ValueError("Invalid lr value: {}".format(lr))
        if alpha < 0.0:
            raise ValueError("Inavlid alpha value: {}".format(alpha))
        if lr_scale < 0.0:
            raise ValueError("Invalid lr_scale value: {}".format(lr_scale))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if temperature < 0.0:
            raise ValueError("Invalid temperature value: {}".format(temperature))

        defaults = dict(lr=lr, alpha=alpha, lr_scale=lr_scale,
                        weight_decay=weight_decay, temperature=temperature)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params       = group["params"]
            params_buf   = [torch.zeros_like(p) for p in params]
            lr           = group["lr"]
            alpha        = group["alpha"]
            lr_scale     = group["lr_scale"]
            weight_decay = group["weight_decay"]
            temperature  = group["temperature"]

            for idx, p in enumerate(params):

                if p.grad is None:
                    continue

                d_p = p.grad
                d_p.add_(p, alpha=weight_decay)

                params_buf[idx] = (1 - alpha) * params_buf[idx] - lr * d_p
                params_buf[idx] += (
                    2.0 * lr * alpha * temperature * lr_scale
                ) ** 0.5 * torch.randn_like(p)

                p.data.add_(params_buf[idx])

        return loss


def build_sghmc_optimizer(model: nn.Module, **kwargs) -> Tuple[Optimizer, List]:

    BASE_LR = kwargs.pop("BASE_LR", None)
    BASE_LR_SCALE = kwargs.pop("BASE_LR_SCALE", None)
    WEIGHT_DECAY = kwargs.pop("WEIGHT_DECAY", None)
    MOMENTUM_DECAY = kwargs.pop("MOMENTUM_DECAY", None)
    TEMPERATURE = kwargs.pop("TEMPERATURE", None)

    _cache = set()
    params = list()
    for module in model.modules():
        
        for module_param_name, value in module.named_parameters(recurse=False):

            if not value.requires_grad:
                continue

            if value in _cache:
                continue
            _cache.add(value)

            schedule_params = dict()
            schedule_params["params"] = [value]
            schedule_params["lr"]           = BASE_LR
            schedule_params["lr_scale"]     = BASE_LR_SCALE
            schedule_params["weight_decay"] = WEIGHT_DECAY
            schedule_params["alpha"]        = MOMENTUM_DECAY
            schedule_params["temperature"]  = TEMPERATURE

            params.append(schedule_params)

    return SGHMC(params, lr=BASE_LR), params
