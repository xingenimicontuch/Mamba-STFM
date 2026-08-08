from __future__ import annotations

import torch
from torch import Tensor, nn


class SASGScan(nn.Module):
    def __init__(self, dimension: int, descriptor_dimension: int, state_dimension: int | None = None, convolution_width: int = 4) -> None:
        super().__init__()
        state = dimension if state_dimension is None else state_dimension
        self.dimension = dimension
        self.state_dimension = state
        self.input_norm = nn.LayerNorm(dimension)
        self.local = nn.Conv1d(dimension, dimension, convolution_width, padding=convolution_width - 1, groups=dimension)
        joined = dimension + descriptor_dimension
        self.delta_projection = nn.Linear(joined, state)
        self.input_projection = nn.Linear(joined, state * dimension)
        self.output_projection = nn.Linear(state, dimension)
        self.skip = nn.Parameter(torch.ones(dimension))
        self.transition_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float32)))

    def recurrence(self, tokens: Tensor, descriptor: Tensor, reverse: bool = False) -> Tensor:
        if reverse:
            tokens = tokens.flip(1)
        batch, length, _ = tokens.shape
        hidden = tokens.new_zeros(batch, self.state_dimension)
        transition = -torch.exp(self.transition_log).to(tokens.dtype)
        outputs: list[Tensor] = []
        for index in range(length):
            token = tokens[:, index]
            joined = torch.cat((token, descriptor), dim=-1)
            delta = torch.nn.functional.softplus(self.delta_projection(joined))
            matrix = self.input_projection(joined).reshape(batch, self.state_dimension, self.dimension)
            written = torch.bmm(matrix, token.unsqueeze(-1)).squeeze(-1)
            hidden = torch.exp(delta * transition) * hidden + delta * written
            outputs.append(self.output_projection(hidden) + self.skip * token)
        result = torch.stack(outputs, dim=1)
        return result.flip(1) if reverse else result

    def forward(self, tokens: Tensor, descriptor: Tensor) -> Tensor:
        normalized = self.input_norm(tokens)
        local = self.local(normalized.transpose(1, 2))[..., : tokens.shape[1]].transpose(1, 2)
        return self.recurrence(torch.nn.functional.silu(local), descriptor)


class BidirectionalSASG(nn.Module):
    def __init__(self, dimension: int, descriptor_dimension: int, state_dimension: int | None = None, convolution_width: int = 4) -> None:
        super().__init__()
        self.scan = SASGScan(dimension, descriptor_dimension, state_dimension, convolution_width)
        self.merge = nn.Linear(2 * dimension, dimension)

    def forward(self, tokens: Tensor, descriptor: Tensor) -> Tensor:
        forward = self.scan(tokens, descriptor)
        normalized = self.scan.input_norm(tokens)
        local = self.scan.local(normalized.transpose(1, 2))[..., : tokens.shape[1]].transpose(1, 2)
        backward = self.scan.recurrence(torch.nn.functional.silu(local), descriptor, reverse=True)
        return self.merge(torch.cat((forward, backward), dim=-1))


class SelectiveBlock(nn.Module):
    def __init__(self, dimension: int, descriptor_dimension: int, state_dimension: int, expansion: int, convolution_width: int, dropout: float, bidirectional: bool) -> None:
        super().__init__()
        scan_type = BidirectionalSASG if bidirectional else SASGScan
        self.scan = scan_type(dimension, descriptor_dimension, state_dimension, convolution_width)
        self.post_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * expansion, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, descriptor: Tensor) -> Tensor:
        tokens = tokens + self.scan(tokens, descriptor)
        return tokens + self.feed_forward(self.post_norm(tokens))
