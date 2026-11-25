"""Entity-swap augmentation for WebNLG supervised fine-tuning.

Motivation
----------
A model fine-tuned on WebNLG tends to memorise the entity names it saw during
training and to reproduce them even when the prompt names a different entity.
This augmentation adds renamed copies of *training* pairs: an entity name is
replaced by a new name consistently in the triples and in every reference.
The only way to produce the renamed reference is to copy the name from the
prompt, so the model learns to copy names instead of recalling them.

Properties
----------
* Uses training records only (dev/test are never touched).
* Keeps the fixed prompt format: the triples are still ``subject | relation |
  object`` with WebNLG-style underscores; only the entity *values* change.
* An entity is renamed only if its surface form (underscores -> spaces) occurs
  verbatim in **every** reference, so no reference is left mentioning the old
  name. Numbers, dates and quoted literals are never renamed.
* Deterministic for a given ``seed``.
"""
from __future__ import annotations

import copy
import random
import re
from typing import Iterable

_LITERAL = re.compile(r'^["\']|^[\d\s.,:+\-/%$€£]+$')
_SYLLABLES = (
    "al ar bel bor cal dor el en fa gar hal ka kel lan lor mar mel mor na nor "
    "or pal ra ren ros sal sel tan tar tor val ven vor wen zar ri li sa da te"
).split()


def _surface(entity: str) -> str:
    """WebNLG entity value -> how it usually appears in text."""
    return entity.replace("_", " ").strip()


def _is_swappable(entity: str) -> bool:
    entity = entity.strip()
    if not entity or _LITERAL.search(entity):
        return False
    # Require at least one capitalised alphabetic word (a name, not a common noun/value).
    return any(word[:1].isupper() and word[:1].isalpha() for word in _surface(entity).split())


def _pseudo_word(rng: random.Random) -> str:
    word = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3)))
    return word.capitalize()


def _new_name(original: str, rng: random.Random, pool: list[str], pseudo_prob: float,
              avoid: set[str]) -> str:
    """A replacement with as many words as ``original``; never reuses a word in ``avoid``."""
    n_words = len(original.split())
    words = []
    while len(words) < n_words:
        if pool and rng.random() >= pseudo_prob:
            word = rng.choice(pool)
        else:
            word = _pseudo_word(rng)
        if word not in avoid and word not in words:
            words.append(word)
    return " ".join(words)


def _split_parenthetical(surface: str) -> tuple[str, str]:
    """'Echoes (The Winterlights album)' -> ('Echoes', ' (The Winterlights album)')."""
    match = re.match(r"^(.*?)(\s*\(.*\))$", surface)
    return (match.group(1), match.group(2)) if match else (surface, "")


def _word_pool(records: Iterable[dict]) -> list[str]:
    """Capitalised words used in training entity names (for realistic-looking names)."""
    pool = set()
    for record in records:
        for triple in record["triples"]:
            parts = [p.strip() for p in triple.split("|")]
            for entity in (parts[0], parts[2]):
                if _is_swappable(entity):
                    for word in _surface(entity).split():
                        if word.isalpha() and word[:1].isupper() and len(word) > 2:
                            pool.add(word)
    return sorted(pool)


def _replace_in_text(text: str, old: str, new: str) -> str:
    pattern = re.compile(r"(?<![\w])" + re.escape(old) + r"(?![\w])")
    return pattern.sub(new, text)


def swap_record(record: dict, rng: random.Random, pool: list[str],
                pseudo_prob: float = 0.5, swap_prob: float = 1.0) -> dict | None:
    """Return a renamed copy of ``record``, or ``None`` if no entity can be renamed."""
    references = record["references"]
    entities = []
    for triple in record["triples"]:
        parts = [p.strip() for p in triple.split("|")]
        for entity in (parts[0], parts[2]):
            if entity not in entities:
                entities.append(entity)

    def in_all(text_form: str) -> bool:
        return all(re.search(r"(?<![\w])" + re.escape(text_form) + r"(?![\w])", ref) for ref in references)

    # Words of this record's own names are never reused as replacement words.
    avoid = {w for e in entities for w in _surface(e).split()}
    mapping = {}            # entity -> (old text form, new text form, new triple value)
    for entity in entities:
        if not _is_swappable(entity) or rng.random() >= swap_prob:
            continue
        surface = _surface(entity)
        if in_all(surface):
            old_text, suffix = surface, ""
        else:
            # 'Echoes_(The_Winterlights_album)' is usually written 'Echoes' in text.
            base, suffix = _split_parenthetical(surface)
            if not suffix or not base or not in_all(base):
                continue
            old_text = base
        new_text = _new_name(old_text, rng, pool, pseudo_prob, avoid)
        avoid.update(new_text.split())
        mapping[entity] = (old_text, new_text, (new_text + suffix).replace(" ", "_"))
    if not mapping:
        return None

    new_triples = []
    for triple in record["triples"]:
        parts = [p.strip() for p in triple.split("|")]
        for index in (0, 2):
            if parts[index] in mapping:
                parts[index] = mapping[parts[index]][2]
        new_triples.append(" | ".join(parts))

    # replace longer names first ("Aarhus Airport" before "Aarhus"), via placeholders
    # so a newly inserted name is never replaced again.
    ordered = sorted(mapping.values(), key=lambda item: -len(item[0]))
    new_references = []
    for ref in references:
        text = ref
        for k, (old_text, _, _) in enumerate(ordered):
            text = _replace_in_text(text, old_text, f"\x00{k}\x00")
        for k, (_, new_text, _) in enumerate(ordered):
            text = text.replace(f"\x00{k}\x00", new_text)
        new_references.append(text)

    augmented = copy.deepcopy(record)
    augmented["triples"] = new_triples
    augmented["references"] = new_references
    return augmented


def augment_with_entity_swaps(records: list[dict], copies: int = 1, seed: int = 11785,
                              pseudo_prob: float = 0.5, swap_prob: float = 1.0) -> list[dict]:
    """Return renamed copies of training records (the originals are not included).

    Args:
        records: WebNLG *training* records with ``id``, ``triples`` and ``references``.
        copies: renamed copies to try per record.
        seed: random seed (deterministic output).
        pseudo_prob: probability that a replacement word is invented rather than
            drawn from words used in training entity names.
        swap_prob: probability of renaming each eligible entity in a copy.
    """
    if copies < 0:
        raise ValueError("copies must be >= 0")
    rng = random.Random(seed)
    pool = _word_pool(records)
    augmented = []
    for record in records:
        for k in range(copies):
            new = swap_record(record, rng, pool, pseudo_prob=pseudo_prob, swap_prob=swap_prob)
            if new is not None:
                new["id"] = f"{record.get('id', 'train')}-swap{k + 1}"
                augmented.append(new)
    return augmented