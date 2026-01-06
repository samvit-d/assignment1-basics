import pytest
import time

from .pretok import pretokenize_chunk, pretokenize_multi


@pytest.fixture
def large_file():
    return "../../data/TinyStoriesV2-GPT4-valid.txt"


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


def byte_tuple(string):
    return tuple(string.encode("utf-8"))


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

    assert out[byte_tuple("poop")] == 2
    assert out[byte_tuple("abc")] == 2
    assert out[byte_tuple("bad")] == 1
    assert out[byte_tuple("end")] == 1


def test_counts_ignore_special_tokens_multi(rep_short_file):
    start = time.time()
    out = pretokenize_multi(
        file=rep_short_file,
        special_tokens=["<|S|>"],
        ncpu=1,
    )
    end = time.time()
    print("time: ", end - start)

    assert out[byte_tuple("poop")] == 16
    assert out[byte_tuple("abc")] == 16
    assert out[byte_tuple("bad")] == 8
    assert out[byte_tuple("end")] == 8


def test_large_file_time(large_file, trials=1):
    start = time.time()
    for _ in range(trials):
        _ = pretokenize_multi(
            file=large_file,
            special_tokens=["<|endoftext|>"],
            ncpu=1,
        )
    end = time.time()
    print("time: ", (end - start) / trials)

    assert True


def test_large_file_time_multi(large_file, trials=8):
    start = time.time()
    for _ in range(trials):
        _ = pretokenize_multi(
            file=large_file,
            special_tokens=["<|endoftext|>"],
            ncpu=8,
        )
    end = time.time()
    print("time: ", (end - start) / trials)

    assert True
