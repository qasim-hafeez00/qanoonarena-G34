"""Runtime citation monitor (Step 4, item 2) - Group G34 policy: NORMALISE, then DROP.

The agent calls check() in the cite_check node, after generation and before memory_write.
`allowed_ids` holds every document id the bot saw this turn (retrieved + memory).

Policy, in order, for every citation the model produced:
  1. kept        - it is exactly an id the bot saw this turn (or the version-free form of one);
  2. normalised  - it is a sloppy spelling of an id the bot saw ("PPC 379", "ppc-379",
                   "Section 379 PPC", "S. 151 CA", "SHC 140894", "PPC-379#2") and is rewritten
                   to that id;
  3. dropped     - anything else (not seen this turn, or not in the corpus at all) is removed
                   from the answer's citation list, because the scoring and the user can only
                   trust sources the bot actually read.

The answer text is never changed, and an answer is never blocked: dropping a bad citation
costs nothing on a correct answer, while blocking would turn many correct answers into
refusals (E7). The action taken is written to the cite_check span of every trace:
"none" (all grounded), "normalised", "dropped", or "normalised+dropped".
"""
from __future__ import annotations

import re
from dataclasses import replace

_ACT_ALIASES = {
    "PPC": "PPC", "PAKISTANPENALCODE": "PPC", "PENALCODE": "PPC",
    "CPC": "CPC", "CODEOFCIVILPROCEDURE": "CPC", "CIVILPROCEDURECODE": "CPC",
    "CA": "CA", "CONTRACTACT": "CA", "CONTRACT": "CA",
    "LA": "LA", "LIMITATIONACT": "LA", "LIMITATION": "LA",
    "SRA": "SRA", "SPECIFICRELIEFACT": "SRA", "SPECIFICRELIEF": "SRA",
    "TPA": "TPA", "TRANSFEROFPROPERTYACT": "TPA", "TRANSFEROFPROPERTY": "TPA",
    "AA": "AA", "ARMSACT": "AA", "ARMS": "AA",
    "SHC": "SHC",
}


def _candidates(raw: str) -> list[str]:
    """Possible canonical ids for a sloppy citation string."""
    c = raw.strip().strip("[]()").split("#")[0].strip()
    out = [c, c.upper()]
    u = re.sub(r"[\s_]+", "-", c.upper()).replace("--", "-")
    out.append(u)
    # "SECTION 379 PPC", "S. 379 OF THE PAKISTAN PENAL CODE", "PPC S.379", "PPC SECTION 379"
    flat = re.sub(r"[^A-Z0-9]", " ", c.upper())
    flat = re.sub(r"\b(SECTION|SEC|S|OF|THE|U)\b", " ", flat)
    toks = flat.split()
    nums = [t for t in toks if re.fullmatch(r"\d+[A-Z]?(?:\d+)?", t)]
    words = "".join(t for t in toks if not re.fullmatch(r"\d+[A-Z]?(?:\d+)?", t) and not t.isdigit())
    words = re.sub(r"\d{4}$", "", words)
    act = _ACT_ALIASES.get(words)
    if act and nums:
        out.append(f"{act}-{nums[0]}")
    return list(dict.fromkeys(x for x in out if x))


def check(answer, allowed_ids: set[str], corpus) -> tuple[object, dict]:
    allowed = set(allowed_ids)
    allowed_base = {a.split("@")[0]: a for a in allowed}

    kept, normalised, dropped = [], [], []
    for c in answer.citations:
        if c in allowed or c.split("@")[0] in allowed_base:
            kept.append(c)
            continue
        hit = next((x for x in _candidates(c) if x in allowed or x.split("@")[0] in allowed_base), None)
        if hit is not None:
            normalised.append({"from": c, "to": hit})
            kept.append(hit)
        else:
            dropped.append(c)

    kept = list(dict.fromkeys(kept))
    action = "+".join(a for a, v in (("normalised", normalised), ("dropped", dropped)) if v) or "none"
    flags = {
        "ungrounded": dropped,                 # citations the bot never saw this turn (or that do not exist)
        "n_ungrounded": len(dropped),
        "not_in_corpus": [c for c in dropped if not corpus.exists(c)],
        "normalised": normalised,
        "n_cited_raw": len(answer.citations),
        "n_cited_final": len(kept),
        "abstain": bool(answer.abstain),
        "action": action,
    }
    if kept != list(answer.citations):
        answer = replace(answer, citations=kept)
    return answer, flags
