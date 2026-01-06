import os
import regex as re
import itertools
import collections

from typing import BinaryIO, Union
from multiprocessing import Pool


def bpe_merge(
    pretoken_counts: dict,
    max_merges: int,
) -> tuple[dict, list]:
    """
    Take in counts of pretokens, and merge until the vocabulary size limit is reached.
    Return both the final vocabulary and an ordered list of merges.
    """
    # First pass, find initial byte-pair counts
    pair_counts = collections.Counter()
    pair_insts = collections.defaultdict(list)
    pretoken_ct_list = list(pretoken_counts.items())
    for idx, (pretoken, count) in enumerate(pretoken_ct_list):
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count
            inst_append(pair_insts, pair, idx)

    # Iteratively merge
    merge_vocab = {}
    merge_list = []
    num_merges = 0
    while num_merges < max_merges:
        num_merges += 1

        # Merge the most common
        pair = pair_counts.most_common(n=1)
        new_code = pair[0] + pair[1]
        merge_vocab[num_merges + 256] = new_code
        merge_list.append(pair[0], pair[1])

        # Update data structures
        update_structures(pair_counts, pair_insts)

    return merge_vocab, merge_list


def update_structures(
    pair: tuple,
    pair_counts: collections.Counter,
    pair_insts: collections.defaultdict[tuple, list],
    pretoken_ct_list: list[tuple[tuple, int]],
):
    """
    Update 'in place' the pair counts + instances and pretoken counts.
    """
    del pair_counts[pair]
    inst_indices = pair_insts[pair]
    concat_pair = pair[0] + pair[1]

    # Update each instance of this token pair
    for idx in inst_indices:
        pretoken, pretoken_ct = pretoken_ct_list[idx]
        new_pretoken = []

        # Construct the new pretoken
        i = 0
        n = len(pretoken)
        while i < n - 1:
            next_tok = pretoken[i]
            i += 1

            # If there's a match, make updates
            if pair == (pretoken[i], pretoken[i + 1]):
                next_tok += pretoken[i + 1]
                i += 1
                if i != 0:
                    new_pair = (pretoken[i], concat_pair)
                    pair_counts[new_pair] += 1
                    inst_append(pair_insts, new_pair, idx)
                if i != n - 2:
                    new_pair = (concat_pair, pretoken[i])
                    pair_counts[new_pair] += 1
                    inst_append(pair_insts, new_pair, idx)

            new_pretoken.append(next_tok)

        # If we missed the last token, add it
        if i == n - 1:
            new_pretoken.append(pretoken[-1])

        pretoken_ct_list[idx] = tuple(new_pretoken), pretoken_ct


def inst_append(pair_insts, pair, idx):
    """
    Modified append to avoid duplication of indices in the instance list.
    """
    if not (pair_insts[pair] and pair_insts[pair][-1] == idx):
        pair_insts[pair].append(idx)
