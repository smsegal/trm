from typing import cast

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor, nn

type Tokens = Int[Tensor, "batch seq"]
type HiddenState = Float[Tensor, "batch seq hidden"]
type AttentionTensor = Float[Tensor, "batch heads seq head_dim"]
type RotaryCache = Float[Tensor, "batch seq head_dim"]
type BroadcastRotaryCache = Float[Tensor, "batch 1 seq head_dim"]
type LmLogits = Float[Tensor, "batch seq vocab"]
type HaltingLogits = Float[Tensor, "batch seq 1"]

POSITION_ID_RANK = 2


@beartype
def _rotate_half(x: AttentionTensor) -> AttentionTensor:
    """Rotate the last dimension by 90 degrees for rotary embeddings."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.reshape_as(x)


@beartype
def apply_rotary_pos_emb(
    x: AttentionTensor,
    cos: BroadcastRotaryCache,
    sin: BroadcastRotaryCache,
) -> AttentionTensor:
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
        self.inv_freq: Tensor
        self._cos_cached: Tensor
        self._sin_cached: Tensor

    def _maybe_update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        needs_update = (
            self._cos_cached.numel() == 0
            or self._cos_cached.size(0) < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )
        if not needs_update:
            return
        inv_freq = cast(Tensor, self.inv_freq).to(device=device, dtype=dtype)
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        self._cos_cached = cos
        self._sin_cached = sin

    @beartype
    def forward(
        self,
        position_ids: Tokens,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[RotaryCache, RotaryCache]:
        if position_ids.dim() != POSITION_ID_RANK:
            raise ValueError("position_ids must have shape [batch, seq_len].")
        device = device or position_ids.device
        dtype = dtype or torch.float32
        seq_len = int(position_ids.max().item()) + 1
        self._maybe_update_cache(seq_len, device, dtype)
        cos = self._cos_cached.to(dtype=dtype)[position_ids]
        sin = self._sin_cached.to(dtype=dtype)[position_ids]
        return cast(RotaryCache, cos), cast(RotaryCache, sin)


class MLP(nn.Module):
    """SwiGLU MLP block used inside the transformer layer."""

    def __init__(self, hidden_dim: int, expansion_factor: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * expansion_factor)
        self.fc1 = nn.Linear(hidden_dim, inner_dim * 2, bias=False)
        self.fc2 = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @beartype
    def forward(self, x: HiddenState) -> HiddenState:
        hidden, gate = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(hidden) * gate
        x = self.fc2(x)
        return cast(HiddenState, self.dropout(x))


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

    @beartype
    def _self_attention(
        self,
        hidden: HiddenState,
        cos: BroadcastRotaryCache,
        sin: BroadcastRotaryCache,
    ) -> HiddenState:
        batch, seq_len, _ = hidden.shape
        qkv = self.qkv(hidden)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(t: Tensor) -> AttentionTensor:
            return cast(AttentionTensor, t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2))

        q_heads = reshape(q)
        k_heads = reshape(k)
        v_heads = reshape(v)

        cos_heads = cos[:, :, :seq_len, :].to(dtype=hidden.dtype)
        sin_heads = sin[:, :, :seq_len, :].to(dtype=hidden.dtype)

        q_heads = apply_rotary_pos_emb(q_heads, cos_heads, sin_heads)
        k_heads = apply_rotary_pos_emb(k_heads, cos_heads, sin_heads)

        attn = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        return cast(HiddenState, self.out_proj(attn))

    @beartype
    def forward(
        self,
        x: HiddenState,
        y: HiddenState,
        z: HiddenState,
        cos: BroadcastRotaryCache,
        sin: BroadcastRotaryCache,
    ) -> tuple[HiddenState, HiddenState]:
        z_norm = self.norm_z_attn(x + y + z)
        z = cast(HiddenState, z + self._self_attention(z_norm, cos, sin))
        z = cast(HiddenState, z + self.mlp_z(self.norm_z_mlp(z)))

        y_norm = self.norm_y_attn(y + z)
        y = cast(HiddenState, y + self._self_attention(y_norm, cos, sin))
        y = cast(HiddenState, y + self.mlp_y(self.norm_y_mlp(y)))
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
        self.layers = nn.ModuleList([TRMBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(n_layers)])
        self.max_position_embeddings = max_position_embeddings

    @staticmethod
    @beartype
    def _prepare_inputs(
        tensor: HiddenState | Tokens | None,
        reference: HiddenState,
        embedding: nn.Embedding,
    ) -> HiddenState:
        if tensor is None:
            return torch.zeros_like(reference)
        if tensor.dtype == torch.long:
            return cast(HiddenState, embedding(tensor))
        return cast(HiddenState, tensor)

    @beartype
    def forward(
        self,
        x: Tokens | HiddenState,
        y: Tokens | HiddenState | None = None,
        z: Tokens | HiddenState | None = None,
        position_ids: Tokens | None = None,
    ) -> tuple[HiddenState, HiddenState]:
        if x.dtype == torch.long:
            x_embedded = cast(HiddenState, self.embedding(x))
        else:
            x_embedded = cast(HiddenState, x)

        batch, seq_len, _ = x_embedded.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"input sequence length {seq_len} exceeds maximum {self.max_position_embeddings}."
            )
        y_tensor = self._prepare_inputs(y, x_embedded, self.embedding)
        z_tensor = self._prepare_inputs(z, x_embedded, self.embedding)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x_embedded.device).unsqueeze(0).expand(batch, -1)
        if position_ids.size(1) < seq_len:
            raise ValueError("position_ids sequence length is shorter than input sequence length.")

        cos, sin = self.pos_emb(position_ids, device=x_embedded.device, dtype=x_embedded.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        for layer in self.layers:
            y_tensor, z_tensor = layer(x_embedded, y_tensor, z_tensor, cos, sin)
        return y_tensor, z_tensor

    @beartype
    def latent_recursion(
        self,
        x: HiddenState,
        y: HiddenState,
        z: HiddenState,
        n: int = 6,
        position_ids: Tokens | None = None,
    ) -> tuple[HiddenState, HiddenState]:
        y_state, z_state = y, z
        for _ in range(n):
            y_state, z_state = self(x, y_state, z_state, position_ids=position_ids)
        return y_state, z_state

    @beartype
    def deep_recursion(
        self,
        x: HiddenState,
        y: HiddenState,
        z: HiddenState,
        n: int = 6,
        recurse: int = 3,
        position_ids: Tokens | None = None,
    ) -> tuple[HiddenState, HiddenState, LmLogits, HaltingLogits]:
        y_state, z_state = y, z
        with torch.no_grad():
            for _ in range(max(0, recurse - 1)):
                y_state, z_state = self.latent_recursion(x, y_state, z_state, n, position_ids=position_ids)
        y_state, z_state = self.latent_recursion(x, y_state, z_state, n, position_ids=position_ids)
        lm_logits = cast(LmLogits, self.lm_head(y_state))
        q_logits = cast(HaltingLogits, self.q_head(y_state))
        return y_state.detach(), z_state.detach(), lm_logits, q_logits
