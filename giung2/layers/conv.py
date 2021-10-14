import torch
import torch.nn as nn

from .utils import initialize_tensor


__all__ = [
    "Conv2d",
    "Conv2d_BatchEnsemble",
    "Conv2d_Dropout",
]


class Conv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(x)


class Conv2d_BatchEnsemble(Conv2d):

    def __init__(self, *args, **kwargs) -> None:
        ensemble_size     = kwargs.pop("ensemble_size", None)
        alpha_initializer = kwargs.pop("alpha_initializer", None)
        gamma_initializer = kwargs.pop("gamma_initializer", None)
        use_ensemble_bias = kwargs.pop("use_ensemble_bias", None)
        super(Conv2d_BatchEnsemble, self).__init__(*args, **kwargs)

        self.ensemble_size     = ensemble_size
        self.alpha_initializer = alpha_initializer
        self.gamma_initializer = gamma_initializer

        # register parameters
        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_channels)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_channels)
            )
        )
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_channels)
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

        _, C1, H1, W1 = x.size()
        r_x = x.view(self.ensemble_size, -1, C1, H1, W1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, C1, 1, 1)
        r_x = r_x.view(-1, C1, H1, W1)

        w_r_x = nn.functional.conv2d(r_x, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups)

        _, C2, H2, W2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, C2, H2, W2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, C2, 1, 1)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, C2, 1, 1)
        s_w_r_x = s_w_r_x.view(-1, C2, H2, W2)

        return s_w_r_x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, ensemble_size={ensemble_size}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            if self.ensemble_bias is None:
                s += ', bias=False, ensemble_bias=False'
            else:
                s += ', bias=False, ensemble_bias=True'
        else:
            if self.ensemble_bias is None:
                s += ', bias=True, ensemble_bias=False'
            else:
                s += ', bias=True, ensemble_bias=True'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Conv2d_Dropout(Conv2d):
    
    def __init__(self, *args, **kwargs) -> None:
        drop_p = kwargs.pop("drop_p", None)
        super(Conv2d_Dropout, self).__init__(*args, **kwargs)
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
