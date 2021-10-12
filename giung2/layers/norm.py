import torch
import torch.nn as nn

from .utils import initialize_tensor


__all__ = [
    "BatchNorm2d",
    "GroupNorm2d",
    "FilterResponseNorm2d",
]


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class _GroupNorm(nn.GroupNorm):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


def GroupNorm2d(
        num_groups: int,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        device = None,
        dtype = None,
    ) -> nn.Module:
    return _GroupNorm(num_groups, num_features, eps, affine, device, dtype)


class FilterResponseNorm2d(nn.Module):

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-6,
            learnable_eps: bool = False,
            learnable_eps_init: float = 1e-4,
        ) -> None:
        super(FilterResponseNorm2d, self).__init__()
        self.num_features       = num_features
        self.eps                = eps
        self.learnable_eps      = learnable_eps
        self.learnable_eps_init = learnable_eps_init

        self.gamma_frn = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta_frn  = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.tau_frn   = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        if self.learnable_eps:
            self.eps_l_frn = nn.Parameter(torch.Tensor(1))
        else:
            self.register_buffer(
                name="eps_l_frn",
                tensor=torch.zeros(1),
                persistent=False
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.gamma_frn)
        nn.init.zeros_(self.beta_frn)
        nn.init.zeros_(self.tau_frn)
        if self.learnable_eps:
            nn.init.constant_(self.eps_l_frn, self.learnable_eps_init)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def extra_repr(self):
        return '{num_features}, eps={eps}, learnable_eps={learnable_eps}'.format(**self.__dict__)

    def _norm_forward(
            self,
            x: torch.Tensor,
            γ: torch.Tensor,
            β: torch.Tensor,
            τ: torch.Tensor,
            ε: torch.Tensor,
        ) -> torch.Tensor:
        ν2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(ν2 + ε)
        x = γ * x + β
        x = torch.max(x, τ)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self._check_input_dim(x)
        return self._norm_forward(x, self.gamma_frn, self.beta_frn,
                                  self.tau_frn, self.eps + self.eps_l_frn.abs())
