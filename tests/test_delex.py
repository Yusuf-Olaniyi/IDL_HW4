from hw4lib.data.delex import delex_triples, delex_reference, delex_record, delex_records, relex
from hw4lib.utils.webnlg import format_webnlg_prompt

def test_triples_and_restore():
    t = ["Aarhus_Airport | cityServed | \"Aarhus, Denmark\"", "Aarhus_Airport | runwayLength | 2777.0",
         "Echoes_(The_Winterlights_album) | followedBy | Aarhus_Airport"]
    new, mapping = delex_triples(t)
    assert new == ["ENT1 | cityServed | ENT2", "ENT1 | runwayLength | 2777.0", "ENT3 | followedBy | ENT1"]
    assert mapping == {"ENT1": "Aarhus Airport", "ENT2": "Aarhus, Denmark", "ENT3": "Echoes"}
    format_webnlg_prompt(new)                      # template still valid
    assert relex("ENT3 was followed by ENT1, which serves ENT2.", mapping) == \
        "Echoes was followed by Aarhus Airport, which serves Aarhus, Denmark."
    assert relex("ENT1 serves ENT7 .", mapping) == "Aarhus Airport serves."   # unknown placeholder dropped

def test_reference_rules():
    t = ["Aarhus_Airport | cityServed | Aarhus", "Aarhus_Airport | runwayLength | 2777.0"]
    assert delex_reference("Aarhus Airport serves the city of Aarhus and is 2777.0 long.", t) == \
        "ENT1 serves the city of ENT2 and is 2777.0 long."
    assert delex_reference("aarhus airport serves AARHUS.", t) == "ENT1 serves ENT2."   # case-insensitive
    assert delex_reference("The airport serves Aarhus.", t) is None                   # an entity missing
    rec = {"id": "x", "triples": t, "references": ["The airport serves Aarhus.", "Aarhus Airport serves Aarhus."]}
    out = delex_record(rec)
    assert out["references"] == ["ENT1 serves ENT2."] and rec["references"][0] == "The airport serves Aarhus."
    assert delex_record({"id": "y", "triples": t, "references": ["The airport serves the city."]}) is None
    recs, stats = delex_records([rec, rec])
    assert stats == {"records_in": 2, "records_kept": 2, "references_in": 4, "references_kept": 2}

def test_parenthetical_names():
    t = ["Echoes_(The_Winterlights_album) | genre | Harbor_rock"]
    assert delex_reference("Echoes is a Harbor rock album.", t) == "ENT1 is a ENT2 album."
    assert delex_reference("Echoes (The Winterlights album) is Harbor rock.", t) == "ENT1 is ENT2."

def test_common_noun_casing():
    _, mapping = delex_triples(["Alan_Bean | occupation | Test_pilot", "Alan_Bean | birthPlace | Kingdom_of_Veloria"])
    assert relex("ENT1 was a ENT2 born in ENT3.", mapping) == "Alan Bean was a test pilot born in Kingdom of Veloria."
    assert relex("ENT2 was the job of ENT1.", mapping) == "Test pilot was the job of Alan Bean."