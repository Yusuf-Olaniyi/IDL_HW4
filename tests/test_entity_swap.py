from hw4lib.data.entity_swap import augment_with_entity_swaps, swap_record
from hw4lib.utils.webnlg import format_webnlg_prompt
import random

RECORDS = [
    {"id": "t1", "triples": ["Aarhus_Airport | cityServed | Aarhus"],
     "references": ["Aarhus Airport serves the city of Aarhus.", "The city of Aarhus is served by Aarhus Airport."]},
    {"id": "t2", "triples": ["Alan_Bean | birthDate | 1932-03-15", "Alan_Bean | occupation | Test_pilot"],
     "references": ["Alan Bean, born on 1932-03-15, was a test pilot."]},
    {"id": "t3", "triples": ["Ajax | numberOfMembers | 17000"], "references": ["The club has 17000 members."]},
]

def test_consistent_rename_and_format():
    aug = augment_with_entity_swaps(RECORDS, copies=2, seed=1)
    assert aug and all(a["id"].split("-swap")[0] in {"t1", "t2"} for a in aug)
    for a in aug:
        format_webnlg_prompt(a["triples"])                        # prompt format still valid
        src = next(r for r in RECORDS if r["id"] == a["id"].split("-swap")[0])
        for ref in a["references"]:
            assert "Aarhus" not in ref and "Alan Bean" not in ref  # no stale names
        for t in a["triples"]:
            name = t.split(" | ")[0].replace("_", " ")
            if not name[0].isdigit():
                assert all(name in ref for ref in a["references"])  # new name is copyable
    # literals untouched
    t2 = [a for a in aug if a["id"].startswith("t2")][0]
    assert "1932-03-15" in t2["triples"][0] and "1932-03-15" in t2["references"][0]
    # 'Test_pilot' not verbatim in references ("test pilot") -> kept as is
    assert t2["triples"][1].endswith("Test_pilot")

def test_skips_when_name_not_in_every_reference_and_deterministic():
    assert swap_record(RECORDS[2], random.Random(0), ["Velo"]) is None
    assert augment_with_entity_swaps(RECORDS, seed=3) == augment_with_entity_swaps(RECORDS, seed=3)
    assert RECORDS[0]["triples"] == ["Aarhus_Airport | cityServed | Aarhus"]   # originals unchanged

def test_parenthetical_entity_and_contained_names():
    rec = {"id": "p", "triples": ["Echoes_(The_Winterlights_album) | followedBy | Aarhus_Airport",
                                  "Aarhus_Airport | cityServed | Aarhus"],
           "references": ["Echoes was followed by Aarhus Airport, which serves Aarhus."]}
    out = swap_record(rec, random.Random(2), ["Velo", "Marn", "Kest", "Dorin"])
    subj = out["triples"][0].split(" | ")[0]
    assert subj.endswith("_(The_Winterlights_album)") and not subj.startswith("Echoes")
    ref = out["references"][0]
    assert "Echoes" not in ref and "Aarhus" not in ref
    for t in out["triples"]:
        for ent in (t.split(" | ")[0], t.split(" | ")[2]):
            base = ent.replace("_", " ").split(" (")[0]
            assert base in ref