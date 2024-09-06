from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import jit
from torch import nn
from torch.nn.parameter import Parameter


class MonotonicLSTMCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float, n_delta_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.weight_zh = Parameter(torch.empty(4 * hidden_size, 1))
        self.bias_h = Parameter(torch.empty(4 * hidden_size))
        modules = []
        in_size = hidden_size
        for _ in range(n_delta_layers-1):
            modules.extend([
                nn.Linear(in_size, forward_size),
                nn.Dropout(dropout_rate),
                nn.ReLU()
            ])
            in_size = forward_size
        self.delta_net = nn.Sequential(*modules, nn.Linear(in_size, 1), nn.ReLU())
        self.init_recurrent_weights()
        self.init_recurrent_biases()
        self.init_sequential_params()

    def init_recurrent_weights(self):
        weights_init_fn = nn.init.orthogonal_
        weights_init_fn(self.weight_ih)
        weights_init_fn(self.weight_hh)
        weights_init_fn(self.weight_zh)

    def init_recurrent_biases(self):
        nn.init.zeros_(self.bias_h)
        # Inizializzo il forget gate a 1. Sembra che sia preferibile
        nn.init.ones_(self.bias_h[self.hidden_size:2*self.hidden_size])

    def init_sequential_params(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.delta_net.apply(weights_init)

    @jit.script_method
    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        hp, cp, zp = h
        gates = (
            torch.mm(x, self.weight_ih.t())
            + torch.mm(hp, self.weight_hh.t())
            + torch.mm(zp, self.weight_zh.t())
            + self.bias_h
        )
        It, Ft, ct, Ot = gates.chunk(4, 1)
        It = F.hardsigmoid(It)
        Ft = F.hardsigmoid(Ft)
        ct = Ft*cp + It*torch.tanh(ct)
        Ot = F.hardsigmoid(Ot)
        ht = Ot * torch.tanh(ct)
        # Monotonicity-preserving step
        dt = self.delta_net(ht)
        zt = zp + dt
        return zt, (ht, ct, zt)

class MonotonicLSTMLayer(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float, n_delta_layers: int):
        super().__init__()
        self.cell = MonotonicLSTMCell(input_size, hidden_size, forward_size, dropout_rate, n_delta_layers)

    @jit.script_method
    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        inputs = x.unbind(1) # dim=1 is the timestep. Probably batch_first=False is faster
        outputs = jit.annotate(List[Tensor], [])
        for t in range(len(inputs)):
            output, h = self.cell(inputs[t], h)
            outputs += [output]
        return torch.stack(outputs, dim=1), h

class MonotonicLSTM(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float, n_delta_layers: int):
        super().__init__()
        self.monotonic_layer = MonotonicLSTMLayer(input_size, hidden_size, forward_size, dropout_rate, n_delta_layers)

    @jit.script_method
    def forward(self, x: Tensor, h0: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        no_batch = x.ndim == 2
        h0, c0, z0 = h0
        if no_batch:
            x = x.unsqueeze(0)
            h0 = h0.unsqueeze(1)
            c0 = c0.unsqueeze(1)
            z0 = z0.unsqueeze(1)
        h0 = h0.squeeze(0)
        c0 = c0.squeeze(0)
        z0 = z0.squeeze(0)
        z, h = self.monotonic_layer(x, (h0, c0, z0))
        if no_batch:
            return z.squeeze(0), h
        h, c, zt = h
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        zt = zt.unsqueeze(0)
        h = (h, c, zt)
        return z, h
