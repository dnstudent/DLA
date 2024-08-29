from typing import Optional, Tuple, List

import torch
from torch import Tensor
from torch import nn
from torch import jit
from torch.nn import LSTM, LSTMCell, RNNCellBase
from torch.nn.parameter import Parameter


class TorchPGALSTMCell(nn.Module):
    def __init__(self, input_size: int, forward_size: int, hidden_size: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.lstm_cell = LSTMCell(input_size+1, hidden_size, bias, device, dtype)
        self.delta_network = nn.Sequential(
            nn.Linear(hidden_size, forward_size), nn.ReLU(), nn.Dropout(),
            nn.Linear(forward_size, forward_size), nn.ReLU(), nn.Dropout(),
            nn.Linear(forward_size, 1), nn.ReLU()
        )

    def forward(self, x: Tensor, hczx: Optional[Tuple[Tensor, Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor, Tensor]:
        # if input.dim() not in (1, 2):
        #     raise ValueError(f"LSTMCell: Expected input to be 1D or 2D, got {input.dim()}D instead")
        # if hx is not None:
        #     for idx, value in enumerate(hx):
        #         if value.dim() not in (1, 2):
        #             raise ValueError(f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got {value.dim()}D instead")
        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)

        if hczx is None:
            zeros = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
            hczx = (zeros, zeros, torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device))
        else:
            hczx = (hczx[0].unsqueeze(0), hczx[1].unsqueeze(0), hczx[2].unsqueeze(0)) if not is_batched else hczx
        hx, cx, zx = hczx
        x = torch.cat([x, zx], dim=-1)

        hd, cd = self.lstm_cell.forward(x, (hx, cx))
        deltad = self.delta_network(hd)
        zd = deltad + zx

        hczd = (hd, cd, zd)

        if not is_batched:
            hczd = (hczd[0].squeeze(0), hczd[1].squeeze(0), hczd[2].squeeze(0))
        return hczd

class CRNNCell(jit.ScriptModule):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.whh = Parameter(torch.randn(hidden_size, hidden_size))
        self.bh = Parameter(torch.randn(hidden_size))
        self.whi = Parameter(torch.randn(hidden_size, input_size))
        self.bo = Parameter(torch.randn(output_size))
        self.woh = Parameter(torch.randn(output_size, hidden_size))

    @jit.script_method
    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        h = torch.sigmoid(torch.mm(h, self.whh.t()) + torch.mm(x, self.whi.t()) + self.bh)
        y = torch.sigmoid(torch.mm(h, self.woh.t()) + self.bo)
        return y, h

class CRNN(jit.ScriptModule):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.cell = CRNNCell(input_size, output_size, hidden_size)

    @jit.script_method
    def forward(self, x: Tensor, h: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        if h is None:
            h = torch.zeros((x.size(0), self.cell.hidden_size), dtype=x.dtype, device=x.device)
        x = x.unbind(1)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(x)):
            out, h = self.cell(x[i], h)
            outputs += [out]
        return torch.stack(outputs, dim=1), h

class LSTMCellS(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        # input: (batch_size, input_size)
        # hidden: (batch_size, hidden_size)
        # cell_s: (batch_size, hidden_size)
        # gate: (batch_size, hidden_size)
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class PGALSTMCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, forward_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_zh = Parameter(torch.randn(4 * hidden_size, 1))
        self.bias_i = Parameter(torch.randn(4 * hidden_size))
        self.bias_h = Parameter(torch.randn(4 * hidden_size))
        self.bias_z = Parameter(torch.randn(4 * hidden_size))
        self.delta_net = nn.Sequential(nn.Linear(hidden_size, forward_size), nn.Dropout(), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())

    @jit.script_method
    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        hp, cp, zp = h
        gates = (
            torch.mm(x, self.weight_ih.t())
            + self.bias_i
            + torch.mm(hp, self.weight_hh.t())
            + self.bias_h
            + torch.mm(zp, self.weight_zh.t())
            + self.bias_z
        )
        it, ft, ct, ot = gates.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        ct = ft*cp + it*torch.tanh(ct)
        ot = torch.sigmoid(ot)
        ht = ot * torch.tanh(ct)
        # Monotonicity-preserving step
        dt = self.delta_net(ht)
        zt = zp + dt
        return ht, (ht, ct, zt)