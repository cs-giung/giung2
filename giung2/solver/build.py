import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from fvcore.common.config import CfgNode
from giung2.solver.optimizer import *
from giung2.solver.scheduler import *


def build_optimizer(cfg: CfgNode, model: nn.Module) -> Optimizer:
    name = cfg.SOLVER.OPTIMIZER.NAME

    if name == "SGD":
        kwargs = dict()

        # basic options
        kwargs.update({
            "BASE_LR"      : cfg.SOLVER.OPTIMIZER.SGD.BASE_LR,
            "WEIGHT_DECAY" : cfg.SOLVER.OPTIMIZER.SGD.WEIGHT_DECAY,
            "MOMENTUM"     : cfg.SOLVER.OPTIMIZER.SGD.MOMENTUM,
            "NESTEROV"     : cfg.SOLVER.OPTIMIZER.SGD.NESTEROV,
        })

        # options for BatchEnsemble
        if cfg.MODEL.BATCH_ENSEMBLE.ENABLED:
            kwargs.update({
                "SUFFIX_BE"       : cfg.SOLVER.OPTIMIZER.SGD.SUFFIX_BE,
                "BASE_LR_BE"      : cfg.SOLVER.OPTIMIZER.SGD.BASE_LR_BE,
                "WEIGHT_DECAY_BE" : cfg.SOLVER.OPTIMIZER.SGD.WEIGHT_DECAY_BE,
                "MOMENTUM_BE"     : cfg.SOLVER.OPTIMIZER.SGD.MOMENTUM_BE,
                "NESTEROV_BE"     : cfg.SOLVER.OPTIMIZER.SGD.NESTEROV_BE,
            })

        optimizer, params = build_sgd_optimizer(model, **kwargs)

        # Adaptive Gradient Clipping (AGC)
        if cfg.SOLVER.OPTIMIZER.AGC.ENABLED:
            optimizer = AGC(
                params         = params,
                base_optimizer = optimizer,
                clipping       = cfg.SOLVER.OPTIMIZER.AGC.LAMBDA,
                eps            = cfg.SOLVER.OPTIMIZER.AGC.EPSILON,
                model          = model if cfg.SOLVER.OPTIMIZER.AGC.IGNORED_PARAMS else None,
                ignored_params = cfg.SOLVER.OPTIMIZER.AGC.IGNORED_PARAMS,
            )

    else:
        raise NotImplementedError(
            f"Unknown cfg.SOLVER.OPTIMIZER.NAME = \"{name}\""
        )

    return optimizer


def build_scheduler(cfg: CfgNode, optimizer: Optimizer) -> _LRScheduler:
    name = cfg.SOLVER.SCHEDULER.NAME

    if name == "WarmupSimpleCosineLR":
        kwargs = dict()
        kwargs.update({
            "NUM_EPOCHS"    : cfg.SOLVER.NUM_EPOCHS,
            "WARMUP_EPOCHS" : cfg.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_EPOCHS,
            "WARMUP_METHOD" : cfg.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_METHOD,
            "WARMUP_FACTOR" : cfg.SOLVER.SCHEDULER.WARMUP_SIMPLE_COSINE_LR.WARMUP_FACTOR,
        })
        scheduler = build_warmup_simple_cosine_lr(optimizer, **kwargs)

    elif name == "WarmupMultiStepLR":
        kwargs = dict()
        kwargs.update({
            "MILESTONES"    : cfg.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.MILESTONES,
            "WARMUP_METHOD" : cfg.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.WARMUP_METHOD,
            "WARMUP_FACTOR" : cfg.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.WARMUP_FACTOR,
            "GAMMA"         : cfg.SOLVER.SCHEDULER.WARMUP_MULTI_STEP_LR.GAMMA,
        })
        scheduler = build_warmup_multi_step_lr(optimizer, **kwargs)

    elif name == "WarmupLinearDecayLR":
        kwargs = dict()
        kwargs.update({
            "MILESTONES"    : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.MILESTONES,
            "WARMUP_METHOD" : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.WARMUP_METHOD,
            "WARMUP_FACTOR" : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.WARMUP_FACTOR,
            "GAMMA"         : cfg.SOLVER.SCHEDULER.WARMUP_LINEAR_DECAY_LR.GAMMA,
        })
        scheduler = build_warmup_linear_decay_lr(optimizer, **kwargs)

    else:
        raise NotImplementedError(
            f"Unknown cfg.SOLVER.SCHEDULER.NAME = \"{name}\""
        )

    return scheduler
