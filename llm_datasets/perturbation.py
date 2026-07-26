# -*- coding: utf-8 -*-
"""llm_datasets/perturbation.py -- Semantic-preserving perturbation pipeline.

Implements the three-strategy rule-based perturbation methodology
described in the paper's Methodology section (Robustness, RS):

    (i)   Synonym substitution using WordNet synsets, restricted to
          content words (NOUN/VERB/ADJ/ADV), accepted only if the
          candidate's cosine similarity to the original word (via
          spaCy word vectors) exceeds 0.85.
    (ii)  Syntactic reordering via dependency-parse-based clause
          transposition: a fronted subordinate clause (advcl) is moved
          to the end of the sentence, or two coordinate clauses joined
          by a conjunction are swapped. Falls back to the original
          question if no reorderable clause structure is found.
    (iii) Surface paraphrasing via back-translation
          (English -> French -> English) using NLLB-200
          (facebook/nllb-200-distilled-600M).

generate_perturbations(question, n=3) returns up to `n` perturbations,
one per strategy, in the fixed order above.

Requirements:
    pip install nltk spacy
    python -m nltk.downloader wordnet omw-1.4
    python -m spacy download en_core_web_md
    (facebook/nllb-200-distilled-600M is downloaded automatically on
    first use via the `transformers` package, and requires `torch`,
    `transformers`, and `sentencepiece`.)

en_core_web_md (not _sm) is required: the small model has no static
word vectors, so cosine-similarity filtering would be meaningless.

Cohen's kappa agreement (paper: kappa = 0.91) reflects a human
double-annotation validation step performed on a sample of generated
perturbations to confirm the gold answer is preserved; it is a manual
review protocol, not something this module reproduces automatically.
If you regenerate perturbations for a new dataset, we recommend
manually reviewing a sample (e.g. 5-10% of items) to confirm the gold
answer still holds under each perturbation before using them for
evaluation, and reporting inter-annotator agreement on that sample.
"""

import warnings

_nlp = None                 # spaCy pipeline (en_core_web_md), lazy-loaded
_translator_en_fr = None    # NLLB pipeline, lazy-loaded
_translator_fr_en = None

_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
_WN_POS = None  # set on first use, after nltk import


def _get_spacy():
    """Lazy-load spaCy with the medium English model (has real word
    vectors, required for meaningful cosine-similarity filtering)."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_md")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_md' not found. Install with:\n"
                "  python -m spacy download en_core_web_md\n"
                "(en_core_web_sm will NOT work here -- it has no static "
                "word vectors, so cosine-similarity filtering would be "
                "meaningless.)"
            )
    return _nlp


# ---------------------------------------------------------------------
# Strategy (i): WordNet synonym substitution, content words only,
# accepted only if cosine similarity > 0.85
# ---------------------------------------------------------------------

def _wordnet_synonym_perturbation(question: str, threshold: float = 0.85) -> str:
    from nltk.corpus import wordnet as wn
    from lemminflect import getInflection
    global _WN_POS
    if _WN_POS is None:
        _WN_POS = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

    nlp = _get_spacy()
    doc = nlp(question)
    tokens = [t.text_with_ws for t in doc]

    for i, tok in enumerate(doc):
        if tok.pos_ not in _CONTENT_POS or not tok.is_alpha:
            continue
        wn_pos = _WN_POS.get(tok.pos_)
        if wn_pos is None:
            continue

        candidates = set()
        # restrict to the single most frequent WordNet sense (synsets are
        # ordered by usage frequency); searching all senses of a
        # polysemous word (e.g. "go" has 30) picks up synonyms of rare
        # senses that don't fit the sentence's actual meaning
        senses = wn.synsets(tok.lemma_, pos=wn_pos)
        for syn in senses[:1]:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                # skip multi-word WordNet lemmas (e.g. "go bad", "working
                # capital") -- substituting these for a single token
                # breaks grammar/meaning far more often than it helps
                if " " in name:
                    continue
                # skip trivial morphological variants of the same word
                if name.lower() == tok.text.lower() or name.lower() == tok.lemma_.lower():
                    continue
                candidates.add(name)
        if not candidates:
            continue

        best, best_sim = None, 0.0
        for cand in candidates:
            cand_doc = nlp(cand)
            if not tok.has_vector or not cand_doc.has_vector or cand_doc.vector_norm == 0:
                continue
            sim = tok.similarity(cand_doc)
            if sim > best_sim:
                best, best_sim = cand, sim
        if best is None or best_sim <= threshold:
            continue

        # inflect the candidate to match the original token's exact
        # grammatical form (tense/number/etc.), not its bare lemma --
        # otherwise e.g. "has" -> "get" breaks subject-verb agreement
        inflected = getInflection(best, tag=tok.tag_)
        replacement = inflected[0] if inflected else best
        if tok.text[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        tokens[i] = replacement + (tok.whitespace_ or "")
        return "".join(tokens)   # one substitution per perturbation

    return ""   # unavailable slot; evaluator keeps it in P with score zero


# ---------------------------------------------------------------------
# Strategy (ii): dependency-parse-based syntactic reordering
# (clause transposition preserving logical scope)
# ---------------------------------------------------------------------

def _syntactic_reorder_perturbation(question: str) -> str:
    nlp = _get_spacy()
    doc = nlp(question)
    sents = list(doc.sents)
    if not sents:
        return ""
    sent = sents[0]
    end_punct = question.rstrip()[-1] if question.rstrip() and question.rstrip()[-1] in "?." else "."

    # Case A: a fronted subordinate clause (e.g. "If X, Y?") -> move it
    # to the end ("Y, if X?"), preserving logical scope.
    advcl = [t for t in sent if t.dep_ == "advcl"]
    if advcl:
        clause_root = advcl[0]
        clause_tokens = sorted(clause_root.subtree, key=lambda t: t.i)
        clause_start, clause_end = clause_tokens[0].i, clause_tokens[-1].i
        if clause_start <= sent.start + 1:
            clause_text = doc[clause_start:clause_end + 1].text.rstrip(",").strip()
            main_text = doc[clause_end + 1:sent.end].text.lstrip(",").strip()
            if main_text and clause_text:
                main_text = main_text[0].upper() + main_text[1:]
                clause_lower = clause_text[0].lower() + clause_text[1:]
                return f"{main_text.rstrip('?.')}, {clause_lower}{end_punct}"

    # Case B: two coordinate clauses joined by a conjunction -> swap them.
    for tok in sent:
        if tok.dep_ == "cc" and tok.head.dep_ in ("ROOT", "conj"):
            left = doc[sent.start:tok.i].text.strip().rstrip(",")
            right = doc[tok.i + 1:sent.end].text.strip().rstrip("?.")
            if left and right:
                right_cap = right[0].upper() + right[1:]
                left_lower = left[0].lower() + left[1:]
                return f"{right_cap}, {tok.text} {left_lower}{end_punct}"

    return ""   # unavailable slot; never pass the original off as perturbed


# ---------------------------------------------------------------------
# Strategy (iii): back-translation via NLLB-200
# (English -> French -> English)
# ---------------------------------------------------------------------

def _get_nllb_translators():
    global _translator_en_fr, _translator_fr_en
    if _translator_en_fr is None:
        from transformers import pipeline
        model_id = "facebook/nllb-200-distilled-600M"
        _translator_en_fr = pipeline(
            "translation", model=model_id,
            src_lang="eng_Latn", tgt_lang="fra_Latn", max_length=400,device=0,
        )
        _translator_fr_en = pipeline(
            "translation", model=model_id,
            src_lang="fra_Latn", tgt_lang="eng_Latn", max_length=400,device=0,
        )
    return _translator_en_fr, _translator_fr_en


def _back_translation_perturbation(question: str) -> str:
    try:
        en_fr, fr_en = _get_nllb_translators()
        fr = en_fr(question)[0]["translation_text"]
        back = fr_en(fr)[0]["translation_text"]
        back = back.strip()
        return back if back and back != question.strip() else ""
    except Exception as e:
        warnings.warn(
            f"NLLB back-translation unavailable ({e}); this perturbation "
            f"slot will use the original question unchanged. Install "
            f"torch/transformers/sentencepiece and ensure "
            f"facebook/nllb-200-distilled-600M can be downloaded."
        )
        return ""


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

_STRATEGIES = [
    _wordnet_synonym_perturbation,
    _syntactic_reorder_perturbation,
    _back_translation_perturbation,
]


def generate_perturbations(question: str, n: int = 3) -> list:
    """Generate up to `n` semantic-preserving perturbations of `question`:
    (1) WordNet synonym substitution, (2) dependency-parse-based
    syntactic reordering, (3) NLLB-200 back-translation, in that order.

    A strategy that cannot produce a distinct valid perturbation returns an
    empty slot.  The evaluator retains that slot in the fixed P denominator
    with zero contribution.  It never substitutes the original question,
    which would turn RS into a repeated-original test and inflate the score.
    """
    n = min(n, len(_STRATEGIES))
    out = []
    for strategy in _STRATEGIES[:n]:
        try:
            candidate = strategy(question)
            out.append(
                candidate
                if candidate and candidate.strip() != question.strip()
                else ""
            )
        except Exception as e:
            warnings.warn(
                f"Perturbation strategy '{strategy.__name__}' failed "
                f"({e}); leaving this fixed-P slot empty (zero contribution)."
            )
            out.append("")
    return out
