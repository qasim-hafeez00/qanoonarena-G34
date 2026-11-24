"""M1: Summary memory (Group G34).

After every turn the bot rewrites ONE short running summary of what the USER has told it,
and the answer node sees that summary under MEMORY instead of raw turns.

Design (full rules in MEMORY.md at the repository root):

  1. LLM rewrite. The same backbone model rewrites the summary from (old summary + new
     user message) with a fixed prompt. When a LoRA adapter is loaded, the adapter is
     switched off for this call, so the summariser is the same for A0 and A1 and the
     answer-tuned adapter cannot turn the summary into a JSON answer.
  2. Pinned anchors. Legally important details are also extracted by rules from every user
     message: statute sections (mapped to corpus ids such as PPC-460), dates, ages, money
     amounts and case numbers. They are kept in a small table outside the LLM text and
     always printed under "Key facts", so a rewrite can never silently drop them.
  3. Corrections. If a user message signals a correction ("sorry", "actually", "I meant",
     "not X but Y", ...) the new anchors of each type REPLACE the older anchors of that
     type, the old values are listed as superseded, and lines of the LLM summary that still
     contain a superseded value are removed (anti-drift).
  4. Never stored: retrieved documents, tool output, the bot's answer text. Only the ids of
     citations that passed the citation checker are kept, in a separate "discussed" list.
  5. Budget. The block is cut to memory_budget_tokens: older "questions asked" go first,
     then LLM summary lines from the top; key facts are dropped last.

The summariser falls back to a rule-based (extractive) summary when there is no real model
(mock backend), when the model output is unusable, or when QA_M1_MODE=rules.

Ablation knobs (environment variables, so harness/ stays unchanged):
    QA_M1_MAX_WORDS   target length of the LLM summary in words      (default 120)
    QA_M1_MODE        "llm" (default) or "rules"
    QA_M1_HINT        "auto" (default), "always" or "off": retrieval hint policy
"""
from __future__ import annotations

import os
import re
from contextlib import nullcontext
from dataclasses import dataclass, field

from .base import Memory, MemoryRead

# ---------------------------------------------------------------------------- acts
# (short code, regex that names the Act in free text, display name)
ACTS = [
    ("PPC", r"pakistan\s+penal\s+code|\bp\.?\s?p\.?\s?c\b|penal\s+code", "Pakistan Penal Code, 1860"),
    ("CPC", r"code\s+of\s+civil\s+procedure|civil\s+procedure\s+code|\bc\.?\s?p\.?\s?c\b", "Code of Civil Procedure, 1908"),
    ("CA", r"contract\s+act", "Contract Act, 1872"),
    ("LA", r"limitation\s+act", "Limitation Act, 1908"),
    ("SRA", r"specific\s+relief\s+act", "Specific Relief Act, 1877"),
    ("TPA", r"transfer\s+of\s+property\s+act|\bt\.?\s?p\.?\s?a\b", "Transfer of Property Act, 1882"),
    ("AA", r"\barms\s+act", "Arms Act, 1878"),
]
ACT_NAME = {code: name for code, _, name in ACTS}
_ACT_RES = [(code, re.compile(rx, re.I)) for code, rx, _ in ACTS]

MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_MONTH_NUM = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}

_SECTION = re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d{1,4}[a-z]?(?:-[a-z])?)\b", re.I)
_SECTION_SHORT = re.compile(r"\b(\d{1,4}[A-Z]?)\s*(PPC|CPC|TPA|P\.P\.C)\b")  # "379 PPC"
_DATE_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTHS})\.?,?\s+(\d{{4}})\b", re.I)
_DATE_MDY = re.compile(rf"\b({MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DATE_NUM = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_YEAR = re.compile(rf"\b({MONTHS})\.?\s+(\d{{4}})\b", re.I)
_YEAR = re.compile(r"\b(?:in|since|year)\s+((?:19|20)\d{2})\b", re.I)
_AGE = re.compile(r"\b(\d{1,3})[\s-]*(?:years?|yrs?)[\s-]*old\b|\baged?\s+(\d{1,3})\b", re.I)
_MONEY = re.compile(r"(?:\brs\.?|\bpkr|rupees?)\s*([\d,]+(?:\.\d+)?(?:\s*(?:lakh|lac|million|crore|thousand))?)"
                    r"|\b([\d,]+(?:\.\d+)?\s*(?:lakh|lac|million|crore|thousand)?)\s*(?:rupees|rs\b|pkr)", re.I)
_CASE_NO = re.compile(r"\b((?:Cr\.?\s?|Crl\.?\s?|Civil\s|C\.?\s?)?(?:Appeal|Acq\.?\s?A|Revision|Rev|Misc|Petition|"
                      r"Suit|Bail|J\.?M|II\.?\s?A|R\.?A)\.?\s*(?:No\.?\s*)?[A-Z]?-?\d+\s*(?:/|of)\s*\d{4})", re.I)

_CORRECTION = re.compile(
    r"\b(sorry|correction|i\s+meant|i\s+mean\b|actually|mistake|i\s+was\s+wrong|not\s+.{1,30}\s+but|"
    r"instead\s+of|let\s+me\s+correct|to\s+correct|i\s+misspoke|wrongly|update:)", re.I)
_BACKREF = re.compile(
    r"\b(this|that|these|those|the\s+same|it|its|my\s+(?:matter|case|situation|problem)|what\s+i\s+(?:told|said|mentioned)|"
    r"earlier|before|above|at\s+the\s+start|first\s+message|previous|mentioned|the\s+section|the\s+case|"
    r"the\s+act|he|she|they|him|her|them)\b", re.I)
_OLD_MARK = re.compile(r"\b(?:not|instead\s+of|rather\s+than|no\s+longer)\s+(?:on\s+|the\s+|in\s+)?", re.I)
_FILLER = re.compile(r"^(?:ok(?:ay)?|thanks?(?: you)?|please .{0,60}|noted|yes|no|sure)[.!]?$", re.I)


@dataclass
class Anchor:
    kind: str      # section | date | age | money | case | act
    value: str     # display value
    key: str       # normalised value used for de-duplication / supersession
    turn: int
    doc_id: str = ""
    pos: int = -1  # character offset in the user message (used to spot "not X but Y")


@dataclass
class SessionState:
    summary: str = ""                                     # LLM-written text (or extractive fallback)
    anchors: list[Anchor] = field(default_factory=list)   # pinned key facts (current values)
    superseded: list[Anchor] = field(default_factory=list)
    discussed: list[str] = field(default_factory=list)    # doc ids the bot cited and that passed cite_check
    questions: list[str] = field(default_factory=list)    # short record of questions asked
    user_log: list[tuple[int, str]] = field(default_factory=list)  # for the extractive fallback only
    last_act: str = ""                                    # most recent Act named by the user
    n_turns: int = 0


SUMMARY_SYSTEM = """You maintain a short running case summary for a legal assistant.
You receive the CURRENT SUMMARY and the user's NEW MESSAGE. Rewrite the summary so that it covers everything the user has said so far.
Rules:
- Keep every legally important fact: what happened, parties and their relationship, dates, places, amounts, ages, documents, the Act and section numbers the user named, and case names or numbers.
- Use only what the user said. Never add legal analysis, guesses, outcomes or sections the user did not name.
- If the new message corrects an earlier fact, replace the old fact with the new one and do not keep the old value.
- Keep the user's own words for names, dates and section numbers.
- Write at most {max_words} words as short lines starting with "- ". Reply with the summary lines only."""


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class SummaryMemory(Memory):
    code = "M1"
    backend = None
    seed = 13

    def __init__(self, max_words: int | None = None, mode: str | None = None, hint: str | None = None):
        self.max_words = int(max_words or os.environ.get("QA_M1_MAX_WORDS", 120))
        self.mode = (mode or os.environ.get("QA_M1_MODE", "llm")).lower()
        self.hint_mode = (hint or os.environ.get("QA_M1_HINT", "auto")).lower()
        self.state: dict[str, SessionState] = {}
        self._corpus_ids: set[str] | None = None

    # ------------------------------------------------------------------ interface
    def reset(self, session_id):
        self.state[session_id] = SessionState()

    def read(self, session_id, query, budget_tokens, count_tokens) -> MemoryRead:
        st = self.state.get(session_id)
        if st is None or st.n_turns == 0:
            return MemoryRead()
        block_lines = self._render(st)
        text = self._fit(block_lines, budget_tokens, count_tokens)
        doc_ids = [a.doc_id for a in st.anchors if a.doc_id] + list(st.discussed)
        doc_ids = list(dict.fromkeys(doc_ids))
        items = [{"kind": a.kind, "value": a.value, "turn": a.turn, "doc_id": a.doc_id} for a in st.anchors]
        items += [{"kind": "superseded", "value": a.value, "turn": a.turn} for a in st.superseded]
        if st.summary:
            items.append({"kind": "summary", "value": st.summary[:600]})
        return MemoryRead(text=text, doc_ids=doc_ids, items=items,
                          retrieval_hint=self._hint(st, query))

    def write(self, session_id, turn, user_msg, answer, state):
        st = self.state.setdefault(session_id, SessionState())
        st.n_turns += 1
        written: list[dict] = []
        msg = _norm_space(user_msg)
        is_corr = bool(_CORRECTION.search(msg))

        # 1) pinned anchors from the USER message only
        new = self._extract(msg, turn, st)
        if is_corr and new:
            # in "not X but Y" / "instead of X", X is the OLD value: never re-pin it
            old_starts = [m.end() for m in _OLD_MARK.finditer(msg)]
            stale = [a for a in new if any(0 <= a.pos - s <= 3 for s in old_starts)]
            new = [a for a in new if a not in stale]
            kinds = {a.kind for a in new}
            keep, gone = [], []
            for a in st.anchors:
                (gone if a.kind in kinds and a.kind != "act" else keep).append(a)
            st.anchors = keep
            st.superseded.extend(gone)
            written += [{"op": "supersede", "kind": a.kind, "value": a.value, "turn": a.turn} for a in gone]
        for a in new:
            if not any(b.kind == a.kind and b.key == a.key for b in st.anchors):
                st.anchors.append(a)
                written.append({"op": "pin", "kind": a.kind, "value": a.value, "doc_id": a.doc_id, "turn": turn})

        # 2) questions asked (short) and the extractive log
        if "?" in msg:
            st.questions.append(f"turn {turn}: {msg[:140]}")
        if not _FILLER.match(msg):
            st.user_log.append((turn, msg))

        # 3) citations that survived the citation checker (ids only, never the answer text)
        flags = (state or {}).get("cite_flags") or {}
        bad = set(flags.get("ungrounded") or [])
        if answer is not None and not getattr(answer, "abstain", False):
            for c in getattr(answer, "citations", []) or []:
                if c not in bad and c not in st.discussed:
                    st.discussed.append(c)
                    written.append({"op": "discussed", "doc_id": c, "turn": turn})
            st.discussed = st.discussed[-8:]

        # 4) rewrite the running summary
        info = self._rewrite(st, msg, turn)
        written.append({"op": "summary", "mode": info["mode"], "words": len(st.summary.split()),
                        "tokens_in": info.get("tokens_in", 0), "tokens_out": info.get("tokens_out", 0),
                        "text": st.summary[:600]})
        if is_corr:
            written.append({"op": "correction_detected", "turn": turn})
        return written

    # ------------------------------------------------------------------ extraction
    def _ids(self) -> set[str]:
        if self._corpus_ids is None:
            try:
                from harness.common import Corpus
                self._corpus_ids = set(Corpus().docs)
            except Exception:  # pragma: no cover
                self._corpus_ids = set()
        return self._corpus_ids

    def _acts_in(self, text: str) -> list[tuple[int, str]]:
        out = []
        for code, rx in _ACT_RES:
            for m in rx.finditer(text):
                out.append((m.start(), code))
        return sorted(out)

    def _extract(self, msg: str, turn: int, st: SessionState) -> list[Anchor]:
        found: list[Anchor] = []
        acts = self._acts_in(msg)
        for _, code in acts:
            found.append(Anchor("act", ACT_NAME[code], code, turn))
        if acts:
            st.last_act = acts[-1][1]
        ids = self._ids()

        def add_section(num: str, pos: int, explicit_act: str = ""):
            num = num.upper().replace("-", "")
            act = explicit_act
            if not act:
                after = [c for p, c in acts if p >= pos]
                before = [c for p, c in acts if p < pos]
                act = (after[0] if after else (before[-1] if before else st.last_act))
            doc_id = f"{act}-{num}" if act else ""
            if doc_id and ids and doc_id not in ids:
                doc_id = ""
            label = f"section {num} of the {ACT_NAME[act]}" if act else f"section {num}"
            found.append(Anchor("section", label + (f" [{doc_id}]" if doc_id else ""), doc_id or label.lower(),
                                turn, doc_id, pos))

        for m in _SECTION.finditer(msg):
            add_section(m.group(1), m.start())
        for m in _SECTION_SHORT.finditer(msg):
            add_section(m.group(1), m.start(), "PPC" if "P" in m.group(2).upper()[:1] else m.group(2).upper())

        for iso, shown, pos in self._dates(msg):
            found.append(Anchor("date", shown, iso, turn, "", pos))
        for m in _AGE.finditer(msg):
            v = m.group(1) or m.group(2)
            found.append(Anchor("age", f"{v} years old", v, turn, "", m.start()))
        for m in _MONEY.finditer(msg):
            v = (m.group(1) or m.group(2) or "").strip()
            if v and re.search(r"\d", v):
                found.append(Anchor("money", f"Rs. {v}", v.replace(",", "").lower(), turn, "", m.start()))
        for m in _CASE_NO.finditer(msg):
            v = _norm_space(m.group(1))
            found.append(Anchor("case", v, v.lower().replace(" ", ""), turn, "", m.start()))
        # de-duplicate within the message
        seen, out = set(), []
        for a in found:
            if (a.kind, a.key) not in seen:
                seen.add((a.kind, a.key))
                out.append(a)
        return out

    @staticmethod
    def _dates(msg: str) -> list[tuple[str, str, int]]:
        """(iso value, text as written, offset) for every date-like expression."""
        out, spans = [], []

        def mon(name: str) -> int:
            return next((v for k, v in _MONTH_NUM.items() if k.startswith(name.lower()[:3])), 0)

        def take(m, iso):
            if any(s <= m.start() < e for s, e in spans):
                return
            spans.append(m.span())
            out.append((iso, m.group(0), m.start()))

        for m in _DATE_DMY.finditer(msg):
            take(m, f"{m.group(3)}-{mon(m.group(2)):02d}-{int(m.group(1)):02d}")
        for m in _DATE_MDY.finditer(msg):
            take(m, f"{m.group(3)}-{mon(m.group(1)):02d}-{int(m.group(2)):02d}")
        for m in _DATE_ISO.finditer(msg):
            take(m, m.group(0))
        for m in _DATE_NUM.finditer(msg):
            take(m, f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}")
        for m in _MONTH_YEAR.finditer(msg):
            take(m, f"{m.group(2)}-{mon(m.group(1)):02d}")
        for m in _YEAR.finditer(msg):
            take(m, m.group(1))
        return out

    # ------------------------------------------------------------------ summary rewrite
    def _rewrite(self, st: SessionState, msg: str, turn: int) -> dict:
        be = self.backend
        use_llm = self.mode == "llm" and be is not None and getattr(be, "name", "") != "mock"
        if use_llm and not _FILLER.match(msg):
            try:
                text, tin, tout = self._llm(st.summary, msg, turn)
                cleaned = self._clean(text)
                if cleaned:
                    st.summary = cleaned
                    self._drop_superseded(st)
                    return {"mode": "llm", "tokens_in": tin, "tokens_out": tout}
            except Exception as e:  # never let memory break a turn
                st.summary = self._extractive(st)
                return {"mode": f"rules (llm error: {type(e).__name__})"}
        elif use_llm:
            return {"mode": "skipped (filler message)"}
        st.summary = self._extractive(st)
        return {"mode": "rules"}

    def _llm(self, old: str, msg: str, turn: int) -> tuple[str, int, int]:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM.format(max_words=self.max_words)},
            {"role": "user", "content": f"CURRENT SUMMARY:\n{old or '(empty)'}\n\nNEW MESSAGE (turn {turn}):\n{msg}"},
        ]
        be = self.backend
        model = getattr(be, "model", None)
        ctx = model.disable_adapter() if model is not None and hasattr(model, "disable_adapter") \
            and hasattr(model, "peft_config") else nullcontext()
        with ctx:
            if model is not None and hasattr(be, "tok"):
                return self._hf_generate(be, messages)
            return be.generate(messages, self.seed)

    def _hf_generate(self, be, messages) -> tuple[str, int, int]:
        """Same model, seed and sampling settings as the harness backend, but a shorter output
        cap: a summary of max_words words never needs the 384 tokens reserved for answers."""
        torch = be.torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        d = be.cfg["decoding"]
        cap = min(d["max_new_tokens"], int(self.max_words * 2) + 40)
        ids = be.tok.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt").to(be.model.device)
        with torch.no_grad():
            out = be.model.generate(ids, max_new_tokens=cap, do_sample=d["temperature"] > 0,
                                    temperature=d["temperature"], top_p=d["top_p"],
                                    pad_token_id=be.tok.eos_token_id)
        new = out[0][ids.shape[1]:]
        return be.tok.decode(new, skip_special_tokens=True), int(ids.shape[1]), int(new.shape[0])

    def _clean(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
        text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
        if not text or text.lstrip().startswith("{"):
            return ""  # the model answered instead of summarising: fall back
        lines = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or re.match(r"^(current summary|summary|new message)\s*:?$", ln, re.I):
                continue
            ln = re.sub(r"^[*•\-]\s*", "", ln)
            lines.append("- " + ln)
        words, kept = 0, []
        for ln in lines:  # hard cap at 1.5x the target so drift cannot grow the summary
            n = len(ln.split())
            if words + n > int(self.max_words * 1.5):
                break
            kept.append(ln)
            words += n
        return "\n".join(kept)

    def _drop_superseded(self, st: SessionState) -> None:
        if not st.superseded or not st.summary:
            return
        current = {a.value.lower() for a in st.anchors}
        olds = [a.value for a in st.superseded if a.value.lower() not in current and len(a.value) >= 3]
        if not olds:
            return
        lines = [ln for ln in st.summary.splitlines()
                 if not any(o.lower() in ln.lower() for o in olds)]
        st.summary = "\n".join(lines)

    def _extractive(self, st: SessionState) -> str:
        """Rule-based fallback: the user's own fact-bearing messages, newest kept when long."""
        lines = [f"- (turn {t}) {m[:220]}" for t, m in st.user_log]
        out, words = [], 0
        for ln in reversed(lines):
            n = len(ln.split())
            if words + n > self.max_words and out:
                break
            out.append(ln)
            words += n
        return "\n".join(reversed(out))

    # ------------------------------------------------------------------ read helpers
    def _render(self, st: SessionState) -> list[tuple[int, str]]:
        """Lines with a drop priority (lower = dropped first when over budget)."""
        lines: list[tuple[int, str]] = [(9, "CONVERSATION SUMMARY (what the user has said so far):")]
        for ln in st.summary.splitlines():
            lines.append((2, ln))
        key = [a for a in st.anchors if a.kind != "act"]
        if key:
            lines.append((9, "KEY FACTS (pinned from the user's messages; latest values):"))
            for a in key:
                lines.append((5, f"- {a.kind}: {a.value} (turn {a.turn})"))
        if st.superseded:
            olds = "; ".join(f"{a.value} (turn {a.turn})" for a in st.superseded[-4:])
            lines.append((4, f"- corrected by the user, do NOT use: {olds}"))
        if st.discussed:
            lines.append((3, "Sources cited earlier in this conversation: " + ", ".join(st.discussed[-6:])))
        if st.questions:
            for q in st.questions[-3:]:
                lines.append((1, f"Earlier question {q}"))
        return lines

    @staticmethod
    def _fit(lines: list[tuple[int, str]], budget: int, count_tokens) -> str:
        keep = list(lines)

        def total():
            return sum(count_tokens(t) + 1 for _, t in keep)

        # drop lowest-priority lines first; within a priority drop the OLDEST (earliest) line
        while keep and total() > budget:
            low = min(p for p, _ in keep if p < 9) if any(p < 9 for p, _ in keep) else None
            if low is None:
                break
            idx = next(i for i, (p, _) in enumerate(keep) if p == low)
            keep.pop(idx)
        return "\n".join(t for _, t in keep)

    def _hint(self, st: SessionState, query: str) -> str:
        """Words added to the BM25 query. Back-references ("this section", "my matter") need the
        remembered context; a self-contained question (it names its own section or case) does not."""
        if self.hint_mode == "off":
            return ""
        q_has_section = bool(_SECTION.search(query) or _SECTION_SHORT.search(query))
        q_has_case = bool(re.search(r"\bv\.?\s|\bvs\.?\s|versus", query, re.I))
        backref = bool(_BACKREF.search(query))
        if self.hint_mode == "auto" and (q_has_section or q_has_case) and not backref:
            return ""
        parts = []
        for a in st.anchors:
            if a.kind == "section" and a.doc_id:
                act, num = a.doc_id.split("-", 1)
                parts.append(f"section {num} {ACT_NAME[act]}")
            elif a.kind == "section":
                parts.append(a.value)
            elif a.kind == "case":
                parts.append(a.value)
        # topical words of the first fact-bearing user message (the "matter")
        if st.user_log:
            parts.append(st.user_log[0][1][:200])
        # plus the first two lines of an LLM-written summary (the extractive one repeats user_log)
        if st.summary and not st.summary.startswith("- (turn "):
            parts += [ln.lstrip("- ") for ln in st.summary.splitlines()[:2]]
        hint = " ".join(parts)
        return " ".join(hint.split()[:70])
