from typing import Tuple, List, Union, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import jit
from torch import nn
from torch.nn.functional import relu, dropout
from torch.nn.parameter import Parameter


def gate_comp(x, weight_x, h, weight_h, z, weight_z, bias):
    return torch.mm(x, weight_x) + torch.mm(h, weight_h) + torch.mm(z, weight_z) + bias

class TheirMonotonicLSTMCell(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 8
        self.forward_size = 5
        self.dropout_rate = dropout_rate

        self.weight_xi = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_xf = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_xc = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_xo = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_hi = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_hf = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_hc = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_ho = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_zi = Parameter(torch.empty(1, self.hidden_size))
        self.weight_zf = Parameter(torch.empty(1, self.hidden_size))
        self.weight_zc = Parameter(torch.empty(1, self.hidden_size))
        self.weight_zo = Parameter(torch.empty(1, self.hidden_size))
        self.bias_i = Parameter(torch.empty(self.hidden_size))
        self.bias_f = Parameter(torch.empty(self.hidden_size))
        self.bias_c = Parameter(torch.empty(self.hidden_size))
        self.bias_o = Parameter(torch.empty(self.hidden_size))
        self.dense_1 = nn.Linear(self.hidden_size, self.forward_size)
        self.dense_2 = nn.Linear(self.forward_size, self.forward_size)
        self.out = nn.Linear(self.forward_size, 1)

        self.dropout_x = None
        self.dropout_z = None
        self.dropout_1 = None
        self.dropout_2 = None

        self.init_recurrent_weights()
        self.init_recurrent_biases()
        self.init_sequential_params()
        self.init_dropouts()

    def init_recurrent_weights(self):
        for weight in [self.weight_xi, self.weight_xf, self.weight_xc, self.weight_xo, self.weight_hi, self.weight_hf, self.weight_hc, self.weight_ho, self.weight_zi, self.weight_zf, self.weight_zc, self.weight_zo]:
            nn.init.orthogonal_(weight)

    def init_recurrent_biases(self):
        for bias in [self.bias_i, self.bias_c, self.bias_o]:
            nn.init.zeros_(bias)
        # Inizializzo il forget gate a 1. Sembra che sia preferibile
        nn.init.ones_(self.bias_f)

    def init_sequential_params(self):
        def weights_init(m):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for layer in [self.dense_1, self.dense_2, self.out]:
            weights_init(layer)

    def init_dropouts(self):
        self.dropout_x = dropout(torch.ones((4, self.input_size), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)
        self.dropout_z = dropout(torch.ones((4, 1), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)
        self.dropout_1 = dropout(torch.ones((1, self.forward_size), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)
        self.dropout_2 = dropout(torch.ones((1, self.forward_size), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)


    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        hp, cp, zp = h
        hp = hp.squeeze(0)
        batch_size = x.size(0)

        xi, xf, xc, xo = (x.repeat((4, 1)) * self.dropout_x.repeat_interleave(batch_size, dim=0, output_size=4*batch_size)).chunk(4, dim=0)
        hi = hp.clone()
        hf = hp.clone()
        hc = hp.clone()
        ho = hp.clone()
        zi, zf, zc, zo = (zp.repeat((4, 1)) * self.dropout_z.repeat_interleave(batch_size, dim=0, output_size=4*batch_size)).chunk(4, dim=0)

        It = xi.mm(self.weight_xi) + hi.mm(self.weight_hi) + zi.mm(self.weight_zi) + self.bias_i
        Ft = xf.mm(self.weight_xf) + hf.mm(self.weight_hf) + zf.mm(self.weight_zf) + self.bias_f
        Ct = xc.mm(self.weight_xc) + hc.mm(self.weight_hc) + zc.mm(self.weight_zc) + self.bias_c
        Ot = xo.mm(self.weight_xo) + ho.mm(self.weight_ho) + zo.mm(self.weight_zo) + self.bias_o

        It = F.hardsigmoid(It)
        Ft = F.hardsigmoid(Ft)
        ct = Ft*cp.squeeze(0) + It*torch.tanh(Ct)
        Ot = F.hardsigmoid(Ot)
        ht = Ot * torch.tanh(ct)
        # Monotonicity-preserving steps
        dt = relu(self.dense_1(ht))*self.dropout_1
        dt = relu(self.dense_2(dt))*self.dropout_2
        dt = relu(self.out(dt))
        zt = zp + dt
        return zt, (ht, ct, zt)

class MonotonicLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forward_size = forward_size
        self.dropout_rate = dropout_rate

        self.weight_xi = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_xf = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_xc = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_xo = Parameter(torch.empty(input_size, self.hidden_size))
        self.weight_hi = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_hf = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_hc = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_ho = Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.weight_zi = Parameter(torch.empty(1, self.hidden_size))
        self.weight_zf = Parameter(torch.empty(1, self.hidden_size))
        self.weight_zc = Parameter(torch.empty(1, self.hidden_size))
        self.weight_zo = Parameter(torch.empty(1, self.hidden_size))
        self.bias_i = Parameter(torch.empty(self.hidden_size))
        self.bias_f = Parameter(torch.empty(self.hidden_size))
        self.bias_c = Parameter(torch.empty(self.hidden_size))
        self.bias_o = Parameter(torch.empty(self.hidden_size))
        self.dense_1 = nn.Linear(self.hidden_size, self.forward_size)
        self.dense_2 = nn.Linear(self.forward_size, self.forward_size)
        self.out = nn.Linear(self.forward_size, 1)

        self.dropout_x = None
        self.dropout_h = None
        self.dropout_z = None
        self.dropout_1 = None
        self.dropout_2 = None

        self.init_recurrent_weights()
        self.init_recurrent_biases()
        self.init_sequential_params()
        self.init_dropouts()

    def init_recurrent_weights(self):
        for weight in [self.weight_xi, self.weight_xf, self.weight_xc, self.weight_xo, self.weight_hi, self.weight_hf, self.weight_hc, self.weight_ho, self.weight_zi, self.weight_zf, self.weight_zc, self.weight_zo]:
            nn.init.orthogonal_(weight)

    def init_recurrent_biases(self):
        for bias in [self.bias_i, self.bias_c, self.bias_o]:
            nn.init.zeros_(bias)
        # Inizializzo il forget gate a 1. Sembra che sia preferibile
        nn.init.ones_(self.bias_f)

    def init_sequential_params(self):
        def weights_init(m):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for layer in [self.dense_1, self.dense_2, self.out]:
            weights_init(layer)

    def init_dropouts(self):
        self.dropout_x = dropout(torch.ones((4, self.input_size), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)
        self.dropout_h = dropout(torch.ones((4, self.hidden_size), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)
        self.dropout_z = dropout(torch.ones((4, 1), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)
        self.dropout_1 = dropout(torch.ones((1, self.forward_size), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)
        self.dropout_2 = dropout(torch.ones((1, self.forward_size), dtype=self.weight_hi.dtype, device=self.weight_hi.device), p=self.dropout_rate, training=self.training)


    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        hp, cp, zp = h
        hp = hp.squeeze(0)
        batch_size = x.size(0)

        xi, xf, xc, xo = (x.repeat((4, 1)) * self.dropout_x.repeat_interleave(batch_size, dim=0, output_size=4*batch_size)).chunk(4, dim=0)
        hi, hf, hc, ho = (hp.repeat((4, 1)) * self.dropout_h.repeat_interleave(batch_size, dim=0, output_size=4*batch_size)).chunk(4, dim=0)
        zi, zf, zc, zo = (zp.repeat((4, 1)) * self.dropout_z.repeat_interleave(batch_size, dim=0, output_size=4*batch_size)).chunk(4, dim=0)

        It = xi.mm(self.weight_xi) + hi.mm(self.weight_hi) + zi.mm(self.weight_zi) + self.bias_i
        Ft = xf.mm(self.weight_xf) + hf.mm(self.weight_hf) + zf.mm(self.weight_zf) + self.bias_f
        Ct = xc.mm(self.weight_xc) + hc.mm(self.weight_hc) + zc.mm(self.weight_zc) + self.bias_c
        Ot = xo.mm(self.weight_xo) + ho.mm(self.weight_ho) + zo.mm(self.weight_zo) + self.bias_o

        It = F.hardsigmoid(It)
        Ft = F.hardsigmoid(Ft)
        ct = Ft*cp.squeeze(0) + It*torch.tanh(Ct)
        Ot = F.hardsigmoid(Ot)
        ht = Ot * torch.tanh(ct)
        # Monotonicity-preserving steps
        dt = relu(self.dense_1(ht))*self.dropout_1
        dt = relu(self.dense_2(dt))*self.dropout_2
        dt = relu(self.out(dt))
        zt = zp + dt
        return zt, (ht, ct, zt)

class MonotonicLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float, cell: Union[Type[TheirMonotonicLSTMCell], Type[MonotonicLSTMCell]]):
        super().__init__()
        self.cell = cell(input_size=input_size, hidden_size=hidden_size, forward_size=forward_size, dropout_rate=dropout_rate)

    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        inputs = x.unbind(1) # dim=1 is the timestep in batch_first=True. Probably batch_first=False (dim=0) is faster
        outputs = jit.annotate(List[Tensor], [])
        self.cell.init_dropouts()
        for t in range(len(inputs)):
            output, h = self.cell(inputs[t], h)
            outputs += [output]
        return torch.stack(outputs, dim=1), h

# class MonotonicLSTM(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float, cell: Union[TheirMonotonicLSTMCell, MonotonicLSTMCell]):
#         super().__init__()
#         self.monotonic_layer = MonotonicLSTMLayer(input_size, hidden_size, forward_size, dropout_rate, cell)
#
#     # @jit.script_method
#     def forward(self, x: Tensor, h0: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
#         return self.monotonic_layer(x, h0)
