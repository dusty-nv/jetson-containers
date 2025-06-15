#!/usr/bin/env python3
print('testing xgrammar...')
"""
XGrammar Demo
=============
This script demonstrates how to compile an integrated JSON grammar with XGrammar and
validate (simulate) a generation from an LLM step by step. It is designed as a
self-contained demonstration that can be run with:

    python test.py

Requirements:
    pip install xgrammar transformers torch numpy

No authentication on Hugging Face is required: we use the public tokenizer for
"gpt2", so you don't need to log in or create a token. No large model is downloaded
and no GPU is used; logits are simply simulated and the IDs of a valid JSON output are fed manually.
"""

import xgrammar as xgr
import torch
from transformers import AutoTokenizer, AutoConfig


def demo_simulated_generation() -> None:
    """Compiles the default JSON grammar and checks a simulated generation."""

    # MODEL AND TOKENIZER -------------------------------------------------------
    model_id = "gpt2"  # Public tokenizer (does not require login)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    vocab_size = config.vocab_size  # May be > tokenizer.vocab_size

    # COMPILE GRAMMAR ----------------------------------------------------------
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_builtin_json_grammar()

    # MATCHER + BITMASK --------------------------------------------------------
    matcher = xgr.GrammarMatcher(compiled_grammar)
    token_bitmask = xgr.allocate_token_bitmask(batch_size=1, vocab_size=vocab_size)

    # SIMULATED RESPONSE -------------------------------------------------------
    simulated_response = '{"name": "John", "age": 30}<|endoftext|>'
    simulated_ids = tokenizer.encode(simulated_response, add_special_tokens=False)

    # Simulate the autoregressive loop token by token
    for token_id in simulated_ids:
        # Fill in the valid tokens according to the grammar
        matcher.fill_next_token_bitmask(token_bitmask)

        # Simulate model logits with noise (not actually used)
        fake_logits = torch.randn(vocab_size)
        xgr.apply_token_bitmask_inplace(fake_logits, token_bitmask)

        # The token from the simulated response *must* be accepted by the grammar
        assert matcher.accept_token(token_id), f"Token {token_id} is invalid for the grammar"

    # When <|endoftext|> is processed, the generation must be marked as finished
    assert matcher.is_terminated(), "Generation was not marked as terminated"

    print("✅ XGrammar demo completed successfully. All checks passed!")


if __name__ == "__main__":
    demo_simulated_generation()

    
print('xgrammar OK\n')
