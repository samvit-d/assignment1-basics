import os
import regex as re
import itertools
import collections

from typing import BinaryIO
from multiprocessing import Pool

# Pattern for regex based tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_multi(
    file: str,
    special_tokens: list[str],
    ncpu: int = 8,
) -> dict[str, int]:
    """
    Return counts of tokens as parsed by a regex-based pretokenizer for a file, parallelized
    across multiple cores.
    """
    with open(file, "rb") as f:
        boundaries = find_chunk_boundaries(f, ncpu, b"<|endoftext|>")

    # Pretokenize each chunk
    args = [
        (file, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with Pool(processes=ncpu) as p:
        chunked_counts = p.starmap(pretokenize_chunk, args)

    # Merge chunked counts
    counts = collections.Counter()
    for chunked_count in chunked_counts:
        counts += chunked_count

    return counts


def pretokenize_chunk(
    file: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[str, int]:
    """
    Return counts of tokens as parsed by a regex-based pretokenizer for a file chunk section.
    Example output: {('p', 'o', 'o', 'p') : 1}
    """
    # Escaped regex pattern for all special tokens
    special_tokens_regex = "|".join([re.escape(tok) for tok in special_tokens])

    # Construct iterator of pretokens in the chunk
    mini_iters = []
    with open(file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        mini_chunks = re.split(special_tokens_regex, chunk)  # Split on special characters
        for mini_chunk in mini_chunks:
            mini_iters.append(re.finditer(PAT, mini_chunk))
    pretokens = itertools.chain.from_iterable(mini_iters)  # Chain together to create one iterator

    # Count pretokens
    counter = collections.Counter()
    for pretoken in pretokens:
        counter[tuple(pretoken.group(0))] += 1

    return counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
