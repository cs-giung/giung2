from torch.nn import Module
from torch.optim import Optimizer, SGD
from typing import List, Tuple


__all__ = [
    "build_sgd_optimizer",
]


def build_sgd_optimizer(model: Module, **kwargs) -> Tuple[Optimizer, List]:

    # basic options 
    BASE_LR         = kwargs.pop("BASE_LR", None)
    WEIGHT_DECAY    = kwargs.pop("WEIGHT_DECAY", None)
    MOMENTUM        = kwargs.pop("MOMENTUM", None)
    NESTEROV        = kwargs.pop("NESTEROV", None)

    # options for BatchEnsemble
    SUFFIX_BE       = kwargs.pop("SUFFIX_BE", tuple())
    BASE_LR_BE      = kwargs.pop("BASE_LR_BE", None)
    WEIGHT_DECAY_BE = kwargs.pop("WEIGHT_DECAY_BE", None)
    MOMENTUM_BE     = kwargs.pop("MOMENTUM_BE", None)
    NESTEROV_BE     = kwargs.pop("NESTEROV_BE", None)

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

            if module_param_name.endswith(tuple(SUFFIX_BE)):
                schedule_params["lr"]           = BASE_LR_BE
                schedule_params["weight_decay"] = WEIGHT_DECAY_BE
                schedule_params["momentum"]     = MOMENTUM_BE
                schedule_params["nesterov"]     = NESTEROV_BE

            else:
                schedule_params["lr"]           = BASE_LR
                schedule_params["weight_decay"] = WEIGHT_DECAY
                schedule_params["momentum"]     = MOMENTUM
                schedule_params["nesterov"]     = NESTEROV

            params.append(schedule_params)
    
    return SGD(params, lr=BASE_LR), params
