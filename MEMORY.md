# MEMORY.md — Group G34, memory type M1 (summary memory)

Code: [agent/memory/m1_summary.py](agent/memory/m1_summary.py). It plugs into the fixed graph
`memory_read → retrieve → generate → cite_check → memory_write`, so `--memory M0` switches it off and
nothing else changes. M0 is the unchanged 2-turn window.

## 1. What our memory stores
For each conversation we keep one `SessionState`:

| Part | Content | Written by |
| --- | --- | --- |
| `summary` | A running summary of what the **user** has said: at most `QA_M1_MAX_WORDS` words (default 120), as `- ` lines | the backbone LLM rewrites it after every turn |
| `anchors` (key facts) | Typed facts pinned by rules from the user's words: `section` (mapped to a corpus id, e.g. *section 460 of the PPC → PPC-460*), `date` (normalised to ISO), `age`, `money`, `case` number, `act` | regex rules |
| `superseded` | Values the user later corrected | correction rule |
| `discussed` | Ids the bot cited that **passed** the citation checker (ids only, at most 8) | `memory_write` |
| `questions` | The last 3 questions the user asked (≤140 chars each) | `memory_write` |

## 2. Write policy
| Question | Our rule |
| --- | --- |
| What is written to memory? | The user's facts, as (a) an LLM-rewritten summary and (b) pinned key facts extracted by rules. Also the ids of citations that survived the citation checker. |
| When is it written? | After **every** answer (`memory_write` node). The LLM rewrite is skipped only for pure filler ("ok", "thanks", "please keep it short"). If the LLM output is unusable (empty, or a JSON answer instead of a summary) the rule-based extractive summary is used. |
| What is never written? | Retrieved documents and search results, tool output, and the bot's answer **text** (its guesses). Ids of citations the checker dropped. The summariser prompt forbids legal analysis and any section the user did not name. |
| How is a new item kept? | Key facts are added when a rule matches. They are de-duplicated on a normalised key (ISO date, doc id, digits of the amount) and are never dropped by the LLM, because they are printed by code, not by the model. |
| What happens when the user corrects a fact? | A message with a correction cue (*sorry, actually, I meant, not X but Y, instead of, correction…*) **replaces** all older key facts of the same type as the new value. In "not X but Y", X is recognised as the old value and never re-pinned. Old values move to `superseded`, every summary line that still contains a superseded value is deleted, and the prompt shows "corrected by the user, do NOT use: …". The summariser is also told to replace rather than keep both values. |

## 3. Read policy
| Question | Our rule |
| --- | --- |
| What is read back for a new message? | One block: `CONVERSATION SUMMARY`, then `KEY FACTS` (latest values with their turn), the "do NOT use" line for corrected values, the sources cited earlier, and the last questions. The doc ids of pinned sections and discussed sources go into `MemoryRead.doc_ids`, so citing them counts as grounded. |
| How is it ordered or ranked? | A fixed order. Each line has a drop priority: headers 9, key facts 5, corrections 4, earlier sources 3, summary lines 2, earlier questions 1. |
| What is dropped first when it does not fit `memory_budget_tokens` (768)? | The lowest priority first, and within a priority the oldest line first. So the order is: earlier questions, then the oldest summary lines, then earlier sources. Key facts go last. The summary is also capped at 1.5 × the word target when it is written, so drift cannot grow it. |
| What do we add to `retrieval_hint`, and why? | The pinned sections spelled as "section N <Act name>" (the fixed BM25 retriever gives +5 to "section N" matches), case numbers, the user's first fact-bearing message, and the first two summary lines, capped at 70 words. The hint is left empty when the new message names its own section or case and has no back-reference ("this section", "my matter", "what I told you"…), so a new topic is not polluted. On the dev set, with the mock backend, this raises gold-document retrieval for T3 from 0/20 to 18/20 and for T4 from 0/20 to 20/20 (T1/T2 unchanged at 19/20 and 20/20). |

## 4. State versus memory
Retrieved chunks, scores, the raw model output, token counts and cite-check flags live only in the
LangGraph `TurnState` and in the trace spans for that turn. They are evidence for one answer, not facts
about the user. Persisting them would let a wrong retrieval or a hallucinated section come back as
"memory" (error E4).

## 5. Three example conversations
*Copied from the `memory_write` / `memory_read` spans. Replace with traces from the final hf runs
(`traces/G34_M1+A0_dev_s13_*.jsonl`) before submitting.*

### Example 1: memory working normally
### Example 2: the user corrects an earlier fact
### Example 3: a case where our memory still fails, and why
