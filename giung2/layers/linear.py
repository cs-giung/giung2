import torch
import torch.nn as nn

from .utils import initialize_tensor


__all__ = [
    "Linear",
    "Linear_BatchEnsemble",
    "Linear_Dropout",
]


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(x)


class Linear_BatchEnsemble(Linear):

    def __init__(self, *args, **kwargs) -> None:
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Linear_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_features)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_features)
            )
        )
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_features)
                )
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        initialize_tensor(self.alpha_be, **self.alpha_initializer)
        initialize_tensor(self.gamma_be, **self.gamma_initializer)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        _, D1 = x.size()
        r_x = x.view(self.ensemble_size, -1, D1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, ensemble_size={}, ensemble_bias={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.ensemble_size, self.ensemble_bias is not None
        )


class Linear_Dropout(Linear):
    
    def __init__(self, *args, **kwargs) -> None:
        drop_p = kwargs.pop("drop_p", None)
        super(Linear_Dropout, self).__init__(*args, **kwargs)
        self.drop_p = drop_p

    def _get_masks(self, x: torch.Tensor, seed: int = None) -> torch.Tensor:
        # TODO: handling random seed...
        probs = torch.ones_like(x) * (1.0 - self.drop_p)
        masks = torch.bernoulli(probs)
        return masks

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs.pop("is_drop", False):
            r = self._get_masks(x)
            x = r * x / (1.0 - self.drop_p)
        return super().forward(x, **kwargs)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, drop_p={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.drop_p
        )
