"""Coordination board shared by concurrently running agents.

Append-only JSONL at records/board.jsonl. Event types:
  claim   {agent, ts, cid, hypothesis, where: ["path" | "path:symbol", ...]}
  release {agent, ts, cid | None, reason}     (None = all of the agent's claims)
  note    {agent, ts, text}                   (measurements / findings others can reuse)
  end     {agent, ts}                         (written by the orchestrator when an episode stops)
A claim is active until released, until its agent's episode ends, or for at most CLAIM_TTL_S.
Standing claims (agent "queue:<episode>") are posted by the orchestrator for accepted siblings that are not yet
merged into the best engine; they have no TTL and are released when the sibling is integrated.
Two claims overlap when they name the same file and either names no symbol, or one symbol equals or contains the
other ("GatedDeltaNet" contains "GatedDeltaNet.forward").
Ablation arms (EVO_ARM) use their own board file so arms cannot see each other.
"""
import fcntl, json, os, time, uuid
from evo.paths import RECORDS

ARM = os.environ.get("EVO_ARM", "")
BOARD = os.path.join(RECORDS, f"board-{ARM}.jsonl" if ARM else "board.jsonl")
CLAIM_TTL_S = 4 * 3600


def append(ev):
    ev = dict(ev, ts=ev.get("ts") or time.time())
    with open(BOARD, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(ev) + "\n")
    return ev


def events():
    if not os.path.exists(BOARD):
        return []
    with open(BOARD) as f:
        return [json.loads(l) for l in f if l.strip()]


def active_claims(evs=None, now=None):
    evs = evs if evs is not None else events()
    now = now or time.time()
    ended = {e["agent"] for e in evs if e["type"] == "end"}
    claims = {}
    for e in evs:
        if e["type"] == "claim":
            claims[e["cid"]] = e
        elif e["type"] == "release":
            for cid in [c for c, v in claims.items() if v["agent"] == e["agent"] and e.get("cid") in (None, c)]:
                claims.pop(cid)
    return [c for c in claims.values()
            if is_queue(c["agent"]) or (c["agent"] not in ended and now - c["ts"] < CLAIM_TTL_S)]


def is_queue(agent):
    return agent.startswith("queue:")


def _split(w):
    path, _, sym = w.partition(":")
    return path.strip().lstrip("./"), sym.strip()


def overlaps(where_a, where_b):
    hits = []
    for a in where_a:
        pa, sa = _split(a)
        for b in where_b:
            pb, sb = _split(b)
            if pa == pb and (not sa or not sb or sa == sb or sa.startswith(sb + ".") or sb.startswith(sa + ".")):
                hits.append((a, b))
    return hits


def new_claim(agent, hypothesis, where):
    return append({"type": "claim", "agent": agent, "cid": uuid.uuid4().hex[:6], "hypothesis": hypothesis, "where": where})


def ago(ts):
    m = int((time.time() - ts) / 60)
    return f"{m} min ago" if m < 120 else f"{m // 60} h ago"
