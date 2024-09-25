from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch import jit
from torch import nn
from torch.nn.functional import relu, dropout

from src.models.tools import make_base_weight, get_sequential_linear_biases


class MonotonicLSTMCell(jit.ScriptModule):
    def __init__(
        self,
        n_input_features: int,
        output_size: int,
        hidden_size: int,
        forward_size: int,
        input_dropout: float,
        recurrent_dropout: float,
        z_dropout: float,
        forward_dropout: float,
        sign: int = 1,
        **kwargs
    ):
        super().__init__()
        self.n_input_features: int = n_input_features
        self.output_size: int = output_size
        self.hidden_size: int = hidden_size
        self.forward_size: int = forward_size
        self.input_dropout_rate: float = input_dropout
        self.recurrent_dropout_rate: float = recurrent_dropout
        self.z_dropout_rate: float = z_dropout
        self.forward_dropout_rate: float = forward_dropout

        self.weight_xi = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_xf = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_xc = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_xo = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_hi = make_base_weight(self.hidden_size, self.hidden_size)
        self.weight_hf = make_base_weight(self.hidden_size, self.hidden_size)
        self.weight_hc = make_base_weight(self.hidden_size, self.hidden_size)
        self.weight_ho = make_base_weight(self.hidden_size, self.hidden_size)
        self.weight_zi = make_base_weight(self.output_size, self.hidden_size)
        self.weight_zf = make_base_weight(self.output_size, self.hidden_size)
        self.weight_zc = make_base_weight(self.output_size, self.hidden_size)
        self.weight_zo = make_base_weight(self.output_size, self.hidden_size)
        self.bias_i = make_base_weight(self.hidden_size)
        self.bias_f = make_base_weight(self.hidden_size)
        self.bias_c = make_base_weight(self.hidden_size)
        self.bias_o = make_base_weight(self.hidden_size)
        self.dense_1 = nn.Linear(self.hidden_size, self.forward_size)
        self.dense_2 = nn.Linear(self.forward_size, self.forward_size)
        self.delta = nn.Linear(self.forward_size, self.output_size)
        self.sign = sign

        self.x_dropout: Tensor = self._make_dropout_mask(
            [4, self.n_input_features], self.input_dropout_rate
        )
        self.h_dropout: Tensor = self._make_dropout_mask(
            [4, self.hidden_size], self.recurrent_dropout_rate
        )
        self.z_dropout: Tensor = self._make_dropout_mask([4, self.output_size], self.z_dropout_rate)
        self.d1_dropout: Tensor = self._make_dropout_mask(
            [1, self.forward_size], self.forward_dropout_rate
        )
        self.d2_dropout: Tensor = self._make_dropout_mask(
            [1, self.forward_size], self.forward_dropout_rate
        )

        self.init_recurrent_weights()
        self.init_recurrent_biases()
        self.init_sequential_params()

    def init_recurrent_weights(self):
        for weight in self.weights:
            nn.init.orthogonal_(weight)

    def init_recurrent_biases(self):
        for bias in [self.bias_i, self.bias_c, self.bias_o]:
            nn.init.zeros_(bias)
        # Inizializzo il forget gate a 1. Sembra che sia preferibile
        nn.init.ones_(self.bias_f)

    def init_sequential_params(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for layer in [self.dense_1, self.dense_2, self.delta]:
            weights_init(layer)

    def _make_dropout_mask(self, shape: List[int], dropout_rate: float):
        return dropout(
            torch.ones(Size(shape), dtype=self.weight_hi.dtype, device=self.weight_hi.device),
            p=dropout_rate,
            training=self.training,
        )

    def init_dropouts(self, batch_size: int):
        self.x_dropout = self._make_dropout_mask(
            [4 * batch_size, self.n_input_features], self.input_dropout_rate
        )
        self.h_dropout = self._make_dropout_mask(
            [4 * batch_size, self.hidden_size], self.recurrent_dropout_rate
        )
        self.z_dropout = self._make_dropout_mask([4 * batch_size, self.output_size], self.z_dropout_rate)
        self.d1_dropout = self._make_dropout_mask(
            [batch_size, self.forward_size], self.forward_dropout_rate
        )
        self.d2_dropout = self._make_dropout_mask(
            [batch_size, self.forward_size], self.forward_dropout_rate
        )

    @property
    def weights(self) -> List[Tensor]:
        return [
            self.weight_xi, self.weight_xf, self.weight_xc, self.weight_xo,
            self.weight_hi, self.weight_hf, self.weight_hc, self.weight_ho,
            self.weight_zi, self.weight_zf, self.weight_zc, self.weight_zo,
            self.dense_1.weight, self.dense_2.weight, self.delta.weight
        ]

    @property
    def biases(self) -> List[Tensor]:
        return [self.bias_i, self.bias_f, self.bias_c, self.bias_o, self.dense_1.bias, self.dense_2.bias, self.delta.bias]

    @jit.script_method
    def forward(
        self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        hp, cp, zp = h

        xi, xf, xc, xo = (torch.cat((x, x, x, x), dim=0) * self.x_dropout).chunk(4, dim=0)
        hi, hf, hc, ho = (torch.cat((hp, hp, hp, hp), dim=0) * self.h_dropout).chunk(4, dim=0)
        zi, zf, zc, zo = (torch.cat((zp, zp, zp, zp), dim=0) * self.z_dropout).chunk(4, dim=0)

        It = (
            xi.mm(self.weight_xi)
            + hi.mm(self.weight_hi)
            + zi.mm(self.weight_zi)
            + self.bias_i
        )
        Ft = (
            xf.mm(self.weight_xf)
            + hf.mm(self.weight_hf)
            + zf.mm(self.weight_zf)
            + self.bias_f
        )
        Ct = (
            xc.mm(self.weight_xc)
            + hc.mm(self.weight_hc)
            + zc.mm(self.weight_zc)
            + self.bias_c
        )
        Ot = (
            xo.mm(self.weight_xo)
            + ho.mm(self.weight_ho)
            + zo.mm(self.weight_zo)
            + self.bias_o
        )

        It = F.hardsigmoid(It)
        Ft = F.hardsigmoid(Ft)
        ct = Ft * cp + It * torch.tanh(Ct)
        Ot = F.hardsigmoid(Ot)
        ht = Ot * torch.tanh(ct)
        # Monotonicity-preserving steps
        dt = relu(self.dense_1(ht)) * self.d1_dropout
        dt = relu(self.dense_2(dt)) * self.d2_dropout
        dt = relu(self.delta(dt))
        zt = zp + self.sign*dt
        return ht, ct, zt


class TheirMonotonicLSTMCell(MonotonicLSTMCell):
    def __init__(self, n_input_features: int, dropout: float, **kwargs):
        super().__init__(
            n_input_features=n_input_features,
            output_size=1,
            hidden_size=8,
            forward_size=5,
            input_dropout=dropout,
            recurrent_dropout=0.0,
            z_dropout=dropout,
            forward_dropout=dropout,
            **kwargs
        )
