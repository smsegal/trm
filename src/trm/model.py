from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dimension by 90 degrees for rotary embeddings."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.reshape_as(x)


def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary positional embeddings to query/key tensors."""
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding cache with on-demand expansion."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    def _maybe_update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        needs_update = (
            self._cos_cached.numel() == 0
            or self._cos_cached.size(0) < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )
        if not needs_update:
            return
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        self._cos_cached = cos
        self._sin_cached = sin

    def forward(
        self,
        position_ids: Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        if position_ids.dim() != 2:
            raise ValueError("position_ids must have shape [batch, seq_len].")
        device = device or position_ids.device
        dtype = dtype or torch.float32
        seq_len = int(position_ids.max().item()) + 1
        self._maybe_update_cache(seq_len, device, dtype)
        cos = self._cos_cached.to(dtype=dtype)[position_ids]
        sin = self._sin_cached.to(dtype=dtype)[position_ids]
        return cos, sin


class MLP(nn.Module):
    """SwiGLU MLP block used inside the transformer layer."""

    def __init__(self, hidden_dim: int, expansion_factor: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * expansion_factor)
        self.fc1 = nn.Linear(hidden_dim, inner_dim * 2, bias=False)
        self.fc2 = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden, gate = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(hidden) * gate
        x = self.fc2(x)
        return self.dropout(x)


class TRMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = dropout

        self.norm_z_attn = nn.RMSNorm(hidden_dim)
        self.norm_z_mlp = nn.RMSNorm(hidden_dim)
        self.norm_y_attn = nn.RMSNorm(hidden_dim)
        self.norm_y_mlp = nn.RMSNorm(hidden_dim)

        self.mlp_z = MLP(hidden_dim, dropout=dropout)
        self.mlp_y = MLP(hidden_dim, dropout=dropout)

    def _self_attention(self, hidden: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, seq_len, _ = hidden.shape
        qkv = self.qkv(hidden)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(t: Tensor) -> Tensor:
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        cos = cos[:, :, :seq_len, :].to(dtype=hidden.dtype)
        sin = sin[:, :, :seq_len, :].to(dtype=hidden.dtype)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        return self.out_proj(attn)

    def forward(self, x: Tensor, y: Tensor, z: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        z_norm = self.norm_z_attn(x + y + z)
        z = z + self._self_attention(z_norm, cos, sin)
        z = z + self.mlp_z(self.norm_z_mlp(z))

        y_norm = self.norm_y_attn(y + z)
        y = y + self._self_attention(y_norm, cos, sin)
        y = y + self.mlp_y(self.norm_y_mlp(y))
        return y, z


class TRM(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_dim: int,
        *,
        num_heads: int | None = None,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads is None:
            num_heads = max(1, hidden_dim // 64)
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.q_head = nn.Linear(hidden_dim, 1)

        head_dim = hidden_dim // num_heads
        self.pos_emb = RotaryEmbedding(head_dim, base=rope_base)
        self.layers = nn.ModuleList(
            [TRMBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.max_position_embeddings = max_position_embeddings

    def _prepare_inputs(
        self,
        tensor: Tensor | None,
        reference: Tensor,
    ) -> Tensor:
        if tensor is None:
            return torch.zeros_like(reference)
        if tensor.dtype == torch.long:
            return self.embedding(tensor)
        return tensor

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        z: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if x.dtype == torch.long:
            x = self.embedding(x)

        batch, seq_len, _ = x.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"input sequence length {seq_len} exceeds maximum {self.max_position_embeddings}."
            )
        y = self._prepare_inputs(y, x)
        z = self._prepare_inputs(z, x)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        if position_ids.size(1) < seq_len:
            raise ValueError("position_ids sequence length is shorter than input sequence length.")

        cos, sin = self.pos_emb(position_ids, device=x.device, dtype=x.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        for layer in self.layers:
            y, z = layer(x, y, z, cos, sin)
        return y, z

    def latent_recursion(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        n: int = 6,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        for _ in range(n):
            y, z = self(x, y, z, position_ids=position_ids)
        return y, z

    def deep_recursion(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        n: int = 6,
        recurse: int = 3,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            for _ in range(max(0, recurse - 1)):
                y, z = self.latent_recursion(x, y, z, n, position_ids=position_ids)
        y, z = self.latent_recursion(x, y, z, n, position_ids=position_ids)
        return y.detach(), z.detach(), self.lm_head(y), self.q_head(y)
