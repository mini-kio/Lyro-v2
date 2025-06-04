import os
import sys
import importlib
import pytest

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.tokenizer import LyroTokenizer


def test_encode_without_text_tokenizer():
    tokenizer = LyroTokenizer(text_tokenizer_path="nonexistent.model")
    assert tokenizer.text_tokenizer is None
    tokens = tokenizer.encode_text("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
