from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import jit
from torch import nn
from torch.nn.parameter import Parameter


class PGADensityCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.weight_zh = Parameter(torch.empty(4 * hidden_size, 1))
        self.bias_h = Parameter(torch.empty(4 * hidden_size))
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_size, forward_size), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(forward_size, forward_size), nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Linear(forward_size, 1), nn.ReLU()
        )
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

class PGADensityLayer(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.cell = PGADensityCell(input_size, hidden_size, forward_size, dropout_rate)

    @jit.script_method
    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        inputs = x.unbind(1) # dim=1 is the timestep. Probably batch_first=False is faster
        outputs = jit.annotate(List[Tensor], [])
        for t in range(len(inputs)):
            output, h = self.cell(inputs[t], h)
            outputs += [output]
        return torch.stack(outputs, dim=1), h

class PGADensityLSTM(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.density_layer = PGADensityLayer(input_size, hidden_size, forward_size, dropout_rate)

    @jit.script_method
    def forward(self, x: Tensor, z0: Tensor) -> Tensor:
        no_batch = x.ndim == 2
        if no_batch:
            x = x.unsqueeze(0)
        zeros = torch.zeros((x.size(0), self.density_layer.cell.hidden_size), dtype=x.dtype, device=x.device)
        # PROBLEMA: la densità iniziale non può essere 0: essendo crescente ed essendo gli input normalizzati sarà sempre < 0!!
        h = (zeros, zeros, z0)
        z = self.density_layer.forward(x, h)[0]
        if no_batch:
            return z.squeeze(0)
        return z

