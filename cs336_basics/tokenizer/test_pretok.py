import re
import pytest
import time

from .pretok import pretokenize_chunk, pretokenize_multi


@pytest.fixture
def short_file(tmp_path):
    """
    Short file content includes:
    - repeated tokens
    - special tokens in between
    - an invalid UTF-8 byte sequence to ensure errors='ignore' is exercised
    """
    content = b"poop <|S|>poop!\nabc<|S|>abc\nbad:\xff\xfeend\n"
    p = tmp_path / "short.txt"
    p.write_bytes(content)
    return p


@pytest.fixture
def rep_short_file(tmp_path):
    """
    Short file content includes:
    - repeated tokens
    - special tokens in between
    - an invalid UTF-8 byte sequence to ensure errors='ignore' is exercised
    """
    content = 8 * b"poop <|S|>poop!\nabc<|S|>abc\nbad:\xff\xfeend\n"
    p = tmp_path / "short.txt"
    p.write_bytes(content)
    return p


def test_counts_ignore_special_tokens(short_file):
    start = time.time()
    out = pretokenize_chunk(
        file=short_file,
        start=0,
        end=short_file.stat().st_size,
        special_tokens=["<|S|>"],
    )
    end = time.time()
    print("time: ", end - start)

    assert out[tuple("poop")] == 2
    assert out[tuple("abc")] == 2
    assert out[tuple("bad")] == 1
    assert out[tuple("end")] == 1


def test_counts_ignore_special_tokens_multi(rep_short_file):
    start = time.time()
    out = pretokenize_multi(
        file=rep_short_file,
        special_tokens=["<|S|>"],
        ncpu=1,
    )
    end = time.time()
    print("time: ", end - start)

    assert out[tuple("poop")] == 16
    assert out[tuple("abc")] == 16
    assert out[tuple("bad")] == 8
    assert out[tuple("end")] == 8