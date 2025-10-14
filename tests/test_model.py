import torch
import pytest

from trm.model import TRM


@pytest.fixture()
def small_trm() -> TRM:
    torch.manual_seed(0)
    model = TRM(
        n_layers=2,
        vocab_size=32,
        hidden_dim=16,
        num_heads=4,
        max_position_embeddings=32,
        dropout=0.0,
    )
    model.eval()
    return model


def test_forward_accepts_token_and_hidden_inputs(small_trm: TRM) -> None:
    torch.manual_seed(1)
    batch, seq_len = 2, 5
    x_hidden = small_trm.embedding(torch.randint(0, small_trm.embedding.num_embeddings, (batch, seq_len), dtype=torch.long))
    y_init, z_init = small_trm.get_init_state(batch, seq_len)

    y_state, z_state = small_trm(x_hidden, y_init, z_init)

    expected_shape = (batch, seq_len, small_trm.embedding.embedding_dim)
    assert y_state.shape == expected_shape
    assert z_state.shape == expected_shape
    assert y_state.dtype == small_trm.embedding.weight.dtype
    assert torch.all(torch.isfinite(y_state))
    assert torch.all(torch.isfinite(z_state))


def test_deep_recursion_returns_stable_logits(small_trm: TRM) -> None:
    torch.manual_seed(2)
    batch, seq_len = 2, 6
    x_tokens = torch.randint(0, small_trm.embedding.num_embeddings, (batch, seq_len), dtype=torch.long)
    x_hidden = small_trm.embedding(x_tokens)
    y_init, z_init = small_trm.get_init_state(batch, seq_len)

    y_state, z_state, lm_logits, q_logits = small_trm.deep_recursion(
        x_hidden,
        y_init,
        z_init,
        n=2,
        recurse=2,
    )

    hidden_shape = (batch, seq_len, small_trm.embedding.embedding_dim)
    assert y_state.shape == hidden_shape
    assert z_state.shape == hidden_shape
    assert not y_state.requires_grad and not z_state.requires_grad

    vocab_size = small_trm.embedding.num_embeddings
    assert lm_logits.shape == (batch, seq_len, vocab_size)
    assert q_logits.shape == (batch, seq_len, 1)
    assert torch.all(torch.isfinite(lm_logits))
    assert torch.all(torch.isfinite(q_logits))
