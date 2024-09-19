from typing import Type, Tuple, List

import torch
from torch import jit, Tensor, nn, Size
from torch.nn import Parameter, functional as F
from torch.nn.functional import dropout

from src.models.pga import MonotonicLSTMCell
from src.models.tools import make_base_weight


class MonotonicLSTM(jit.ScriptModule):
    def __init__(
        self, n_input_features: int, *, cell: Type[MonotonicLSTMCell], **kwargs
    ):
        super().__init__()
        self.cell = cell(n_input_features=n_input_features, **kwargs)

    @jit.script_method
    def forward(
        self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        inputs = x.unbind(1)
        outputs = jit.annotate(List[Tensor], [])
        h0, c0, z0 = h
        h = (h0.squeeze(0), c0.squeeze(0), z0.squeeze(0))
        self.cell.init_dropouts(x.size(0))
        for t in range(len(inputs)):
            h = self.cell(inputs[t], h)
            outputs += [h[2]]
        hf, cf, zf = h
        h = (hf.unsqueeze(0), cf.unsqueeze(0), zf.unsqueeze(0))
        return torch.stack(outputs, dim=1), h


class DropoutLSTMCell(jit.ScriptModule):
    def __init__(
        self,
        n_input_features: int,
        hidden_size: int,
        input_dropout: float,
        recurrent_dropout: float
    ):
        super().__init__()
        self.n_input_features: int = n_input_features
        self.hidden_size: int = hidden_size
        self.input_dropout_rate: float = input_dropout
        self.recurrent_dropout_rate: float = recurrent_dropout

        self.weight_xi: Parameter = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_xf: Parameter = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_xc: Parameter = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_xo: Parameter = make_base_weight(self.n_input_features, self.hidden_size)
        self.weight_hi: Parameter = make_base_weight(self.hidden_size, self.hidden_size)
        self.weight_hf: Parameter = make_base_weight(self.hidden_size, self.hidden_size)
        self.weight_hc: Parameter = make_base_weight(self.hidden_size, self.hidden_size)
        self.weight_ho: Parameter = make_base_weight(self.hidden_size, self.hidden_size)
        self.bias_i: Parameter = make_base_weight(self.hidden_size)
        self.bias_f: Parameter = make_base_weight(self.hidden_size)
        self.bias_c: Parameter = make_base_weight(self.hidden_size)
        self.bias_o: Parameter = make_base_weight(self.hidden_size)

        self.x_dropout: Tensor = self._make_dropout_mask(
            [4, self.n_input_features], self.input_dropout_rate
        )
        self.h_dropout: Tensor = self._make_dropout_mask(
            [4, self.hidden_size], self.recurrent_dropout_rate
        )

        self.init_recurrent_weights()
        self.init_recurrent_biases()

    def init_recurrent_weights(self):
        for weight in [
            self.weight_xi,
            self.weight_xf,
            self.weight_xc,
            self.weight_xo,
            self.weight_hi,
            self.weight_hf,
            self.weight_hc,
            self.weight_ho,
        ]:
            nn.init.orthogonal_(weight)

    def init_recurrent_biases(self):
        for bias in [self.bias_i, self.bias_c, self.bias_o]:
            nn.init.zeros_(bias)
        # Inizializzo il forget gate a 1. Sembra che sia preferibile
        nn.init.ones_(self.bias_f)

    def _make_dropout_mask(self, shape: List[int], dropout_rate: float) -> Tensor:
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

    @jit.script_method
    def forward(
        self, x: Tensor, h: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        hp, cp = h

        xi, xf, xc, xo = (torch.cat((x, x, x, x), dim=0) * self.x_dropout).chunk(4, dim=0)
        hi, hf, hc, ho = (torch.cat((hp, hp, hp, hp), dim=0) * self.h_dropout).chunk(4, dim=0)

        It = (
            xi.mm(self.weight_xi)
            + hi.mm(self.weight_hi)
            + self.bias_i
        )
        Ft = (
            xf.mm(self.weight_xf)
            + hf.mm(self.weight_hf)
            + self.bias_f
        )
        Ct = (
            xc.mm(self.weight_xc)
            + hc.mm(self.weight_hc)
            + self.bias_c
        )
        Ot = (
            xo.mm(self.weight_xo)
            + ho.mm(self.weight_ho)
            + self.bias_o
        )

        It = F.hardsigmoid(It)
        Ft = F.hardsigmoid(Ft)
        ct = Ft * cp.squeeze(0) + It * torch.tanh(Ct)
        Ot = F.hardsigmoid(Ot)
        ht = Ot * torch.tanh(ct)
        return ht, ct


class DropoutLSTM(jit.ScriptModule):
    def __init__(self, n_input_features: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.cell = DropoutLSTMCell(n_input_features=n_input_features, hidden_size=hidden_size, input_dropout=dropout_rate, recurrent_dropout=dropout_rate)

    @jit.script_method
    def forward(
        self, x: Tensor, h: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = x.unbind(1)
        outputs = jit.annotate(List[Tensor], [])
        h0, c0 = h
        h = (h0.squeeze(0), c0.squeeze(0))
        self.cell.init_dropouts(x.size(0))
        for t in range(len(inputs)):
            h = self.cell(inputs[t], h)
            outputs += [h[0]]
        hf, cf = h
        h = (hf.unsqueeze(0), cf.unsqueeze(0))
        return torch.stack(outputs, dim=1), h
