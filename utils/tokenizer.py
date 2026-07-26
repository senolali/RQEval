# -*- coding: utf-8 -*-
"""utils/tokenizer.py -- Shared token-counting utility for the Efficiency
(ES) metric.

Previously, every model adapter counted tokens via `len(response.split())`
(whitespace word count), which is a crude approximation: it undercounts
subword-tokenized text and is not comparable across models with different
vocabularies. API adapters now use provider-reported output/completion
token usage whenever it is available. This module provides the deterministic
fallback, based on tiktoken's cl100k_base encoding (the same byte-pair-
encoding family used by GPT-3.5/4-class models).

The fallback remains an approximation for providers that omit usage
metadata, but it is far closer to true token counts than raw whitespace
word count. Local HuggingFace models pass their own tokenizer via the
`hf_tokenizer` argument, which gives an exact count.

If tiktoken's encoding cannot be loaded (e.g. no network access to fetch
its vocabulary file on first use), this falls back to whitespace word
count and emits a one-time warning, rather than raising -- token counting
should never be the reason an evaluation run fails.
"""

import warnings

_tiktoken_encoding = None
_tiktoken_failed = False
_warned = False


def count_tokens(text: str, hf_tokenizer=None) -> int:
    """Count tokens in `text`.

    If `hf_tokenizer` is given (a HuggingFace tokenizer instance, as used
    by local models), it is used directly for an exact count. Otherwise,
    falls back to a shared tiktoken (cl100k_base) estimate, and finally to
    whitespace word count if tiktoken is unavailable.
    """
    global _tiktoken_encoding, _tiktoken_failed, _warned

    if not text:
        return 0

    if hf_tokenizer is not None:
        try:
            return len(hf_tokenizer.encode(text))
        except Exception:
            pass  # fall through to the shared estimate below

    if not _tiktoken_failed:
        if _tiktoken_encoding is None:
            try:
                import tiktoken
                _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                _tiktoken_failed = True
                if not _warned:
                    warnings.warn(
                        f"tiktoken encoding unavailable ({e}); falling back "
                        f"to whitespace word count for token_count / ES. "
                        f"This is a coarser approximation -- install "
                        f"tiktoken and ensure network access to its "
                        f"vocabulary file if you want the closer estimate.",
                        stacklevel=2,
                    )
                    _warned = True
        if _tiktoken_encoding is not None:
            try:
                return len(_tiktoken_encoding.encode(text))
            except Exception:
                pass  # fall through to word count

    return len(text.split())
