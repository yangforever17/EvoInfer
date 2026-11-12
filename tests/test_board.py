import time

from evo import board as B


def test_overlaps():
    assert B.overlaps(["m.py:A.f"], ["m.py:A"])                         # a class contains its methods
    assert B.overlaps(["m.py"], ["m.py:A.f"])                           # a bare file overlaps everything in it
    assert not B.overlaps(["m.py:A.f"], ["m.py:A.g"])
    assert not B.overlaps(["m.py:A"], ["n.py:A"])
    assert B.overlaps(["./m.py:A.f"], ["m.py:A.f"])


def test_active_claims_lifecycle():
    now = time.time()
    evs = [{"type": "claim", "agent": "a1", "cid": "c1", "ts": now, "where": ["m.py:f"], "hypothesis": "x"},
           {"type": "claim", "agent": "a2", "cid": "c2", "ts": now, "where": ["m.py:g"], "hypothesis": "y"},
           {"type": "claim", "agent": "a3", "cid": "c3", "ts": now - 5 * 3600, "where": ["m.py:h"], "hypothesis": "old"},
           {"type": "claim", "agent": "queue:e9", "cid": "q9", "ts": now - 9 * 3600, "where": ["m.py:k"], "hypothesis": "sibling"},
           {"type": "release", "agent": "a1", "cid": None, "ts": now},
           {"type": "end", "agent": "a2", "ts": now}]
    live = {c["cid"] for c in B.active_claims(evs, now)}
    assert live == {"q9"}   # released, episode ended, expired (TTL); standing queue claims stay until integrated
