"""Submission 2 helper: evidence for reviewing another group's 10 sessions.

    python scripts/review_helper.py contrib_received/G12.jsonl > review_notes_G12.md

For every session it prints what a reviewer must judge (the decision itself stays human):
  - the official checker's messages for that session;
  - the gold answer, keywords, citations, and each evidence quote with its context in the corpus;
  - whether the keywords appear in the gold answer AND in the cited document;
  - T3/T4: whether the final question alone already retrieves the gold document (if it does, the
    turn may not really need memory), and the gap between the fact turn and the question;
  - T4: the amendment notes of the cited section, and whether the keyword year appears in them;
  - T5: corpus hits for the Act/section named in the question (if any, it may be answerable).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.common import Corpus, norm_text, read_jsonl  # noqa: E402
from harness.retriever import BM25Retriever  # noqa: E402
from harness.validate_sessions import validate  # noqa: E402


def ctx(doc_text: str, quote: str, width: int = 160) -> str:
    t, q = norm_text(doc_text), norm_text(quote)
    i = t.find(q)
    if i < 0:
        return "(quote NOT found)"
    return "…" + t[max(0, i - width): i + len(q) + width] + "…"


def main():
    path = sys.argv[1]
    sessions = read_jsonl(path)
    corpus = Corpus()
    ret = BM25Retriever(corpus)
    errs = validate(sessions, corpus, contrib=True)
    print(f"# Review notes for {Path(path).name}\n")
    print(f"Checker: {'OK' if not errs else f'{len(errs)} problem(s)'}")
    for e in errs:
        print(f"- {e}")
    alltext = {d.doc_id: norm_text(d.text) for d in corpus.docs.values()}
    for s in sessions:
        sid, typ = s.get("session_id"), s.get("type")
        print(f"\n## {sid} ({typ})\n")
        for i, t in enumerate(s["turns"], 1):
            print(f"{i}. U: {t['user']}")
        for i, t in enumerate(s["turns"], 1):
            g = t.get("gold")
            if not g:
                continue
            print(f"\n**Gold turn {i}**: {g.get('answer')}")
            print(f"- keywords {g.get('answer_keywords')}, citations {g.get('citations')}, abstain {g.get('abstain')}, "
                  f"memory_dependent {g.get('memory_dependent')} {g.get('depends_on_turns')}")
            for kw in g.get("answer_keywords") or []:
                alts = [norm_text(a) for a in kw.split("|") if a.strip()]
                in_ans = any(a in norm_text(g.get("answer", "")) for a in alts)
                in_doc = any(any(a in alltext.get(c, "") for a in alts) for c in g.get("citations") or [])
                print(f"- keyword `{kw}`: in gold answer={in_ans}, in cited doc={in_doc}")
            for e in g.get("evidence") or []:
                d = corpus.get(e.get("doc_id", ""))
                print(f"- evidence {e.get('doc_id')}: \"{e.get('quote')}\"")
                if d:
                    print(f"  - title: {d.title}; context: {ctx(d.text, e.get('quote', ''))}")
                    if d.meta.get("outcome"):
                        print(f"  - judgment metadata: outcome={d.meta.get('outcome')}, date={d.meta.get('date')}")
            hits = [h.doc_id for h in ret.search(t["user"])]
            gold_c = g.get("citations") or []
            print(f"- question ALONE retrieves: {hits}  -> gold retrieved without memory: "
                  f"{any(c in hits for c in gold_c) if gold_c else 'n/a'}")
            if typ in ("T3", "T4"):
                dep = g.get("depends_on_turns") or []
                print(f"- gap between fact turn(s) {dep} and question turn {i}: {i - max(dep) if dep else '?'} "
                      f"(baseline window = 2 turns, so a gap of 3 or more is needed)")
            if typ == "T4":
                for c in gold_c:
                    d = corpus.get(c)
                    if d:
                        notes = d.meta.get("amended_by") or []
                        print(f"- amendment notes of {c}: {notes}")
                        years = set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", " ".join(notes)))
                        print(f"  - years in notes: {sorted(years)}")
            if typ == "T5" or g.get("abstain"):
                q = t["user"]
                m = re.search(r"section\s+(\d+[A-Z]?)", q, re.I)
                act = re.search(r"(?:the\s+)?([A-Z][\w\-’' ]+?(?:Act|Code|Ordinance|Order)),?\s*(\d{4})?", q)
                if act:
                    name = norm_text(act.group(1))
                    n = sum(1 for v in alltext.values() if name in v)
                    print(f"- corpus documents mentioning '{act.group(1)}': {n}")
                if m:
                    same = [d for d in corpus.docs if d.split("-", 1)[-1].upper() == m.group(1).upper()]
                    print(f"- corpus ids with section {m.group(1)}: {same[:10]}")
                print(f"- top retrieved for the question: {hits}")


if __name__ == "__main__":
    main()
