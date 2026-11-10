import pytest
import torch

from tinyserve import ModelConfig, TinyTransformer


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(max_seq_len=64)


@pytest.fixture
def model(config: ModelConfig) -> TinyTransformer:
    torch.manual_seed(1234)
    return TinyTransformer(config).eval()


@pytest.fixture
def prompts(config: ModelConfig) -> list[list[int]]:
    generator = torch.Generator().manual_seed(7)
    return torch.randint(1, config.vocab_size, (4, 6), generator=generator).tolist()
