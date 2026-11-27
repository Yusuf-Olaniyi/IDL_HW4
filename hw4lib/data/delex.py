"""Delexicalisation for WebNLG data-to-text.

Entity names in the triples are replaced by placeholders (ENT1, ENT2, ...).
The model learns to write e.g. "ENT1 serves the city of ENT2." and the names
are restored deterministically after generation.

Rules (identical for train, dev and test, computed from the triples only):
* Every subject/object value is an entity, except purely numeric/date-like
  values (e.g. "1945-04-27", "2777.0", "27400"), which stay as they are.
* The same value always gets the same placeholder; placeholders are numbered
  by first appearance in the triples.
* The prompt template is unchanged (format_webnlg_prompt); only the entity
  values inside the triples are replaced.
"""
from __future__ import annotations

import copy
import re

PLACEHOLDER = "ENT"
_NUMERIC = re.compile(r"^[\d\s.,:+\-/%$€£]+$")
_PLACEHOLDER_RE = re.compile(r"\bENT(\d+)\b")


def _split(triple: str) -> list[str]:
    return [part.strip() for part in triple.split("|")]


def _is_entity(value: str) -> bool:
    stripped = value.strip().strip('"').strip()
    return bool(stripped) and not _NUMERIC.match(stripped)


def _surface(value: str) -> str:
    """'Aarhus_Airport' -> 'Aarhus Airport'; '"Aarhus, Denmark"' -> 'Aarhus, Denmark'."""
    return value.replace("_", " ").strip().strip('"').strip()


def _restore_text(value: str) -> str:
    """Text written back for an entity: the surface form without '(disambiguation)'."""
    surface = _surface(value)
    base = re.sub(r"\s*\(.*\)$", "", surface).strip()
    return base or surface


def _variants(value: str) -> list[str]:
    surface = _surface(value)
    base = _restore_text(value)
    return sorted({surface, base}, key=len, reverse=True)


def delex_triples(triples: list[str]) -> tuple[list[str], dict[str, str]]:
    """Replace entity values by placeholders.

    Returns the new triples and ``mapping`` placeholder -> text to restore.
    """
    value_to_ph: dict[str, str] = {}
    mapping: dict[str, str] = {}
    new_triples = []
    for triple in triples:
        parts = _split(triple)
        for index in (0, 2):
            value = parts[index]
            if not _is_entity(value):
                continue
            if value not in value_to_ph:
                placeholder = f"{PLACEHOLDER}{len(value_to_ph) + 1}"
                value_to_ph[value] = placeholder
                mapping[placeholder] = _restore_text(value)
            parts[index] = value_to_ph[value]
        new_triples.append(" | ".join(parts))
    return new_triples, mapping


def _value_to_placeholder(triples: list[str]) -> dict[str, str]:
    value_to_ph = {}
    for triple in triples:
        parts = _split(triple)
        for value in (parts[0], parts[2]):
            if _is_entity(value) and value not in value_to_ph:
                value_to_ph[value] = f"{PLACEHOLDER}{len(value_to_ph) + 1}"
    return value_to_ph


def delex_reference(reference: str, triples: list[str]) -> str | None:
    """Delexicalise one reference; ``None`` if some entity is not found in it."""
    value_to_ph = _value_to_placeholder(triples)
    # Longest names first so "Aarhus Airport" is replaced before "Aarhus".
    order = sorted(value_to_ph, key=lambda v: -len(_variants(v)[0]))
    text = reference
    for k, value in enumerate(order):
        found = False
        for variant in _variants(value):
            pattern = re.compile(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", re.IGNORECASE)
            text, count = pattern.subn(f"\x00{k}\x00", text)
            if count:
                found = True
                break
        if not found:
            return None
    for k, value in enumerate(order):
        text = text.replace(f"\x00{k}\x00", value_to_ph[value])
    return text


def delex_record(record: dict) -> dict | None:
    """Delexicalised copy of a training/dev record, or ``None`` if no reference survives."""
    new_triples, mapping = delex_triples(record["triples"])
    references = [delex_reference(ref, record["triples"]) for ref in record["references"]]
    kept = [ref for ref in references if ref is not None]
    if not kept:
        return None
    out = copy.deepcopy(record)
    out["triples"] = new_triples
    out["references"] = kept
    out["delex_map"] = mapping
    return out


def delex_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Delexicalise a list of records; also return coverage statistics."""
    out, refs_in, refs_kept = [], 0, 0
    for record in records:
        refs_in += len(record["references"])
        new = delex_record(record)
        if new is not None:
            refs_kept += len(new["references"])
            out.append(new)
    stats = {
        "records_in": len(records), "records_kept": len(out),
        "references_in": refs_in, "references_kept": refs_kept,
    }
    return out, stats


def _is_common_noun(name: str) -> bool:
    """DBpedia style 'Test pilot' / 'Harbor rock': only the first word capitalised."""
    words = name.split()
    return (len(words) > 1 and words[0][:1].isupper() and words[0][1:].islower()
            and all(w.islower() for w in words[1:]))


def relex(text: str, mapping: dict[str, str]) -> str:
    """Put names back; placeholders the input does not have are removed.

    Values like 'Test pilot' are lower-cased ('test pilot') unless they start a sentence.
    """
    def replace(match):
        name = mapping.get(match.group(0), "")
        before = text[:match.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?"
        if name and _is_common_noun(name) and not sentence_start:
            name = name[0].lower() + name[1:]
        return name

    restored = _PLACEHOLDER_RE.sub(replace, text)
    return re.sub(r"\s{2,}", " ", restored).replace(" .", ".").replace(" ,", ",").strip()