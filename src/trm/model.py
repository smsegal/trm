import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
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
        hidden_state: HiddenState = self.dropout(x)
        return hidden_state


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
        qkv_heads = rearrange(
            self.qkv(hidden),
            "b s (qkv h d) -> qkv b h s d",
            qkv=3,
            h=self.num_heads,
        )
        qk_heads, v_heads = qkv_heads[:2], qkv_heads[2]

        cos_heads = cos[:, :, :seq_len, :].to(dtype=hidden.dtype)
        sin_heads = sin[:, :, :seq_len, :].to(dtype=hidden.dtype)

        qk_heads = rearrange(qk_heads, "qk b h s d -> (qk b) h s d")
        cos_qk = cos_heads.repeat(2, 1, 1, 1)
        sin_qk = sin_heads.repeat(2, 1, 1, 1)
        qk_heads = apply_rotary_pos_emb(qk_heads, cos_qk, sin_qk)
        q_heads, k_heads = rearrange(qk_heads, "(qk b) h s d -> qk b h s d", qk=2, b=batch).unbind(dim=0)

        attn = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")
        hidden_state: HiddenState = self.out_proj(attn)
        return hidden_state

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
        z_attn: HiddenState = z + self._self_attention(z_norm, cos, sin)
        z_mlp: HiddenState = z_attn + self.mlp_z(self.norm_z_mlp(z_attn))

        y_norm = self.norm_y_attn(y + z_mlp)
        y_attn: HiddenState = y + self._self_attention(y_norm, cos, sin)
        y_mlp: HiddenState = y_attn + self.mlp_y(self.norm_y_mlp(y_attn))
        return y_mlp, z_mlp


class TRM(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        *,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.q_head = nn.Linear(hidden_dim, 1)

        head_dim = hidden_dim // num_heads
        cos_cache, sin_cache = self._build_rotary_cache(
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            base=rope_base,
        )
        self.register_buffer("rotary_cos", cos_cache, persistent=False)
        self.register_buffer("rotary_sin", sin_cache, persistent=False)
        self.rotary_cos: Tensor
        self.rotary_sin: Tensor
        self.layers = nn.ModuleList([TRMBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(n_layers)])
        self.max_position_embeddings = max_position_embeddings

    @staticmethod
    def _build_rotary_cache(
        *,
        max_position_embeddings: int,
        head_dim: int,
        base: float,
    ) -> tuple[Tensor, Tensor]:
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        return cos, sin

    @beartype
    def forward(
        self,
        x: HiddenState,
        y: HiddenState,
        z: HiddenState | None = None,
        position_ids: Tokens | None = None,
    ) -> tuple[HiddenState, HiddenState]:
        batch, seq_len, _ = x.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError(f"input sequence length {seq_len} exceeds maximum {self.max_position_embeddings}.")
        
        if z is None:
            z = torch.zeros_like(x)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        if position_ids.size(1) < seq_len:
            raise ValueError("position_ids sequence length is shorter than input sequence length.")

        if position_ids.dim() != POSITION_ID_RANK:
            raise ValueError("position_ids must have shape [batch, seq_len].")
        if position_ids.device != x.device:
            position_ids = position_ids.to(x.device)
        if position_ids.dtype != torch.long:
            position_ids = position_ids.to(dtype=torch.long)
        if int(position_ids.max().item()) >= self.max_position_embeddings:
            raise ValueError("position_ids contain indices beyond the rotary cache.")

        cos_cache: RotaryCache = self.rotary_cos[position_ids]
        sin_cache: RotaryCache = self.rotary_sin[position_ids]
        target_dtype = x.dtype
        if cos_cache.dtype != target_dtype:
            cos_cache = cos_cache.to(dtype=target_dtype)
            sin_cache = sin_cache.to(dtype=target_dtype)

        cos_broadcast = rearrange(cos_cache, "b s d -> b 1 s d")
        sin_broadcast = rearrange(sin_cache, "b s d -> b 1 s d")

        for layer in self.layers:
            y, z = layer(x, y, z, cos_broadcast, sin_broadcast)
        return y, z

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
        lm_logits: LmLogits = self.lm_head(y_state)
        q_logits: HaltingLogits = self.q_head(y_state)
        return y_state.detach(), z_state.detach(), lm_logits, q_logits
