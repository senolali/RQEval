# -*- coding: utf-8 -*-
"""metrics/answer_extraction.py -- Type-aware canonical answer extraction.

This module implements the corrected answer-matching protocol described in
the paper's Experimental Setup / Reproducibility note. It replaces the
earlier "gold substring anywhere in prediction" strategy, which could match
incidentally inside an unrelated token for short gold answers (e.g. gold
"D" matching the "d" in "drastic"; gold "5" matching inside "25").

Canonical equality by gold type:
    numeric   -> compared as parsed floating-point values (respecting a
                 "####"-delimited final answer when present, GSM8K-style)
    yes/no    -> compared against {"yes", "no"} using an unambiguous
                 lexical signal (negation markers etc.), not raw substring
                 presence
    multiple-choice (A-D) -> requires a structurally anchored option
                 letter (e.g. following "answer is", "correct option is",
                 or as the first token of the response), not any bare
                 occurrence of the letter
    free text -> normalized string equality, or containment of the
                 normalized gold string within the normalized prediction

If no canonical answer can be extracted from a response, extract_answer()
returns None; callers should treat this as non-matching (incorrect /
inconsistent), not as an automatic pass.

Correctness (CQ) and Robustness (RS) call is_correct() below; Consistency
(CS) calls extract_answer() directly to compare canonicalized answers
across repeated runs (see metrics/consistency.py).
"""
import re

_num_re  = re.compile(r'-?[\d,]+\.?\d*')
_hash_re = re.compile(r'####\s*(-?[\d,.]+)')
_choice_res = [
    # "Answer: C", "# Answer: **B**", "answer is (C)", "correct option is B" ...
    re.compile(r'\banswer\s*(?:is)?\s*:?\s*\**\(?([ABCD])\)?(?![A-Za-z])', re.I),
    re.compile(r'\bcorrect\s+(?:answer|option|choice|conclusion|interpretation)\s*(?:is)?\s*:?\s*\**\(?([ABCD])\)?(?![A-Za-z])', re.I),
    # "The correct conclusion is **D. ..." / "...interpretation ... is: D"
    re.compile(r'\b(?:conclusion|interpretation|option|choice)\b[^.\n]{0,80}?\bis\s*:?\s*\**\(?([ABCD])(?![A-Za-z])', re.I),
    # generic "is D" / "is: **B**" -- case-SENSITIVE so "is a/an" never matches
    re.compile(r'\bis\s*:?\s*\**\(?([ABCD])\)?(?![A-Za-z])'),
    re.compile(r'^\s*#*\s*\**\(?([ABCD])\)?[\).:\s]'),
    re.compile(r'\(([ABCD])\)'),
]


def _last_line_choice(p):
    for ln in reversed([l.strip() for l in str(p).splitlines() if l.strip()]):
        m = re.match(r'^\**\(?([ABCD])\)?[\).:\s\*]', ln + ' ')
        return m.group(1) if m else None
    return None


_yes_words = ("yes", "true", "correct", "indeed")
_no_words  = ("no", "false", "incorrect", "not correct", "untrue",
              "not", "never", "unlikely", "doubtful", "cannot", "impossible")


def _norm(t):
    return re.sub(r'\s+', ' ', str(t).lower().strip().rstrip('.,;:!?*'))


def _gold_kind(gold):
    g = _norm(gold)
    if g in ("yes", "no", "true", "false"):
        return "yn"
    if len(g) == 1 and g.upper() in "ABCD":
        return "choice"
    if re.match(r'^-?[\d,.]+$', g.replace(' ', '')):
        return "num"
    return "text"


def extract_answer(pred, gold):
    """Extract the canonical final answer from a raw model response.

    Returns None if no canonical answer can be extracted.
    """
    p = str(pred or "")
    if not p.strip():
        return None                      # empty response -> invalid

    kind = _gold_kind(gold)

    if kind == "num":
        m = _hash_re.search(p)
        if m:
            s = m.group(1)
        else:
            nums = _num_re.findall(p)
            if not nums:
                return None
            s = nums[-1]
        try:
            return repr(float(s.replace(',', '')))
        except ValueError:
            return None

    if kind == "yn":
        t = _norm(p)
        first = t.split('.')[0][:80]     # the first sentence/clause is decisive

        # find the EARLIEST occurrence of any yes/no signal word in the
        # first clause, and decide based on whichever comes first in the
        # text -- not "is any no-word present anywhere", which previously
        # made "Yes, ... not fish" resolve to "no" just because a no-word
        # loop ran before the yes-word loop, regardless of word order
        def _earliest(words):
            best = None
            for w in words:
                pat = r'(?:^|\s)' + re.escape(w) + r'(?:\s|$|[.,;:!?])'
                m = re.search(pat, first)
                if m and (best is None or m.start() < best):
                    best = m.start()
            return best

        no_pos  = _earliest(_no_words)
        yes_pos = _earliest(_yes_words)
        if no_pos is not None or yes_pos is not None:
            if yes_pos is None:
                return "no"
            if no_pos is None:
                return "yes"
            return "no" if no_pos < yes_pos else "yes"

        # no signal in the first sentence -> look for one unambiguous
        # signal across the whole response
        has_y = any(re.search(r'\b' + w + r'\b', t) for w in ("yes", "true"))
        has_n = any(re.search(r'\b' + w + r'\b', t) for w in ("no", "false"))
        if has_y != has_n:
            return "yes" if has_y else "no"
        return None

    if kind == "choice":
        for rx in _choice_res:
            m = rx.search(p)
            if m:
                return m.group(1).upper()
        return _last_line_choice(p)

    # free text: first check whether the response contains the gold answer
    # (answer-level equivalence), then fall back to normalized short-text
    # equality
    t = _norm(p)
    g = _norm(gold)
    if g and g in t:
        return f"contains:{g}"
    if len(t) <= 60:
        return t
    return None


def canonical_gold(gold):
    """Canonicalize a gold answer into the same representation space used
    by extract_answer(), so the two can be compared with plain equality."""
    kind = _gold_kind(gold)
    g = _norm(gold)
    if kind == "num":
        try:
            return repr(float(g.replace(',', '').replace(' ', '')))
        except ValueError:
            return g
    if kind == "yn":
        return "yes" if g in ("yes", "true") else "no"
    if kind == "choice":
        return g.upper()
    return f"contains:{g}"


def is_correct(pred, gold):
    """True if the canonical answer extracted from `pred` matches the
    canonical form of `gold`. This is the single source of truth for
    Correctness (CQ) and, via AccuracyMetric, Robustness (RS).

    Numeric answers are compared with a tolerance of 1e-3, per the
    paper's Methodology; all other answer types require exact
    canonical-form equality.
    """
    extracted = extract_answer(pred, gold)
    gold_canon = canonical_gold(gold)
    if extracted is None:
        return False

    if _gold_kind(gold) == "num":
        try:
            # both are repr(float(...)) strings; compare the underlying
            # values with tolerance instead of exact string equality
            return abs(float(extracted) - float(gold_canon)) < 1e-3
        except ValueError:
            return extracted == gold_canon

    return extracted == gold_canon
