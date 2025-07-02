#!/usr/bin/env python3
import os
print('testing torchtext...')

from collections import OrderedDict
from typing import Dict, List, Optional

from fairseq.data.dictionary import Dictionary
from torchtext.vocab import Vocab


def build_fairseq_vocab(
    vocab_file: str,
    dictionary_class: Dictionary = Dictionary,
    special_token_replacements: Dict[str, str] = None,
    unk_token: str = "<unk>",
    max_vocab: int = -1,
    min_count: int = -1,
    tokens_to_add: Optional[List[str]] = None,
):
    """Function builds a torchtext Vocab for models pre-trained using Fairseq
    modules.

    The dictionary class can take any Fairseq Dictionary class and is
    used to load the vocab file.

    """
    if not special_token_replacements:
        special_token_replacements = {
            "<pad>": "__PAD__",
            "<s>": "__BEGIN_OF_SENTENCE__",
            "</s>": "__END_OF_SENTENCE__",
            "<unk>": "__UNKNOWN__",
            "<mask>": "__MASK__",
        }
        unk_replacement = (
            special_token_replacements[unk_token] if unk_token in special_token_replacements else unk_token
        )
        special_tokens_to_remove = [special_pair[0] for special_pair in special_token_replacements]
        special_tokens_to_add = tuple(
            special_pair[1] for special_pair in special_token_replacements if special_pair[0] != unk_token
        )

    with open(vocab_file) as f:
        dictionary = dictionary_class.load(f)
        # finalize will sort the dict based on frequency so only do this if
        # a min_count or max_vocab size is specified
        if min_count > 0 or max_vocab > 0:
            dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
        if tokens_to_add:
            for token in tokens_to_add:
                dictionary.add_symbol(token)

        dictionary_items = list(zip(dictionary.symbols, dictionary.count))

        ordered_dict = OrderedDict()
        # add special tokens to beginning of ordered_dict
        for s in special_tokens_to_add:
            ordered_dict[s] = 1

        # add all other tokens from dictionary_items
        for token, freq in dictionary_items:
            ordered_dict[token] = freq

        # remove special_tokens_to_remove from dict
        for s in special_tokens_to_remove:
            if s in ordered_dict:
                del ordered_dict[s]

        return Vocab(dictionary_items, unk_token=unk_replacement)

print("OK Torchtext")
