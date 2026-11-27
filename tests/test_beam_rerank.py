import torch
from hw4lib.model import DecoderOnlyTransformer
from hw4lib.decoding.beam_rerank import beam_search, rerank, _banned_tokens, entity_surfaces
from hw4lib.decoding.sequence_generator import SequenceGenerator

class Tok:
    eos_id = 0

def _model(rope=True):
    torch.manual_seed(0)
    return DecoderOnlyTransformer(num_layers=2, d_model=32, num_heads=4, num_kv_heads=2, d_ff=64,
                                  dropout=0.0, max_len=64, num_classes=50, use_rope=rope).eval()

def test_width1_equals_greedy_and_cache_matches_full():
    m = _model(); prompt = [5, 7, 9, 11]
    greedy, _ = SequenceGenerator(m.score, Tok(), 4 + 20, "cpu", cache_score_fn=m.score_cached).generate_greedy(
        torch.tensor([prompt]))
    g = greedy[0, 4:].tolist(); g = g[:g.index(0)] if 0 in g else g
    b = beam_search(m, prompt, 0, beam_width=1, max_new_tokens=20)[0][0]
    assert b == g
    # KV-cached beam search == full-prefix beam search on the same model
    class NoCache:                      # same weights, exposes only full-prefix scoring
        use_rope = False
        max_len = m.max_len
        score = staticmethod(m.score)
    cached = beam_search(m, prompt, 0, beam_width=4, max_new_tokens=15)
    plain = beam_search(NoCache(), prompt, 0, beam_width=4, max_new_tokens=15)
    assert [c[0] for c in cached] == [p[0] for p in plain]
    assert all(abs(a[1] - b[1]) < 1e-4 for a, b in zip(cached, plain))

def test_constraints():
    m = _model(); prompt = [5, 7, 9, 11]
    for ids, _ in beam_search(m, prompt, 0, beam_width=3, max_new_tokens=30,
                              no_repeat_ngram_size=2, min_new_tokens=3):
        assert len(ids) >= 3
        grams = [tuple(ids[i:i + 2]) for i in range(len(ids) - 1)]
        assert len(grams) == len(set(grams))
    assert _banned_tokens([1, 2, 3, 1, 2], 3) == {3}

def test_rerank_prefers_coverage():
    triples = ["Aarhus_Airport | cityServed | Aarhus", "Echoes_(album) | genre | Rock"]
    assert entity_surfaces(triples) == ["Aarhus Airport", "Aarhus", "Echoes", "Rock"]
    texts = [("Aarhus Airport serves a city.", -0.1), ("Aarhus Airport serves Aarhus; Echoes is Rock.", -0.9)]
    assert rerank(texts, triples).startswith("Aarhus Airport serves Aarhus")