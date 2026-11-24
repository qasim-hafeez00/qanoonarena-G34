"""Builds contrib/G34.jsonl (Step 1). Every quote was checked by hand against harness/corpus.

    python scripts/make_contrib_g34.py
    python -m harness.validate_sessions contrib/G34.jsonl --contrib
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def g(answer, kws, cites, ev, mem=False, dep=None, abstain=False):
    return {"answer": answer, "answer_keywords": kws, "citations": cites, "exact_version": False,
            "abstain": abstain, "memory_dependent": mem, "depends_on_turns": dep or [], "as_of": None,
            "evidence": [{"doc_id": d, "quote": q} for d, q in ev]}


def T(u, gold=None):
    return {"user": u, "gold": gold}


S = []
S.append({"session_id": "G34-T1-01", "type": "T1", "author": "G34", "turns": [
    T("I signed as guarantor for my friend's bank loan. Under the Contract Act, 1872, how far does my "
      "liability as a surety extend compared to his?",
      g("Under section 128 of the Contract Act, 1872 the liability of the surety is co-extensive with that of "
        "the principal debtor, unless the contract provides otherwise.",
        ["128", "co-extensive|coextensive|co extensive"], ["CA-128"],
        [("CA-128", "The liability of the surety is co-extensive with that of the principal debtor, "
                    "unless it is otherwise provided by the contract.")]))]})

S.append({"session_id": "G34-T1-02", "type": "T1", "author": "G34", "turns": [
    T("Under the Arms Act, 1878, what is the maximum punishment for breaking a condition of an arms licence, "
      "when the act is not punishable under section 19 or section 20?",
      g("Section 21 of the Arms Act, 1878 (breach of licence) provides imprisonment up to six months, or a fine "
        "up to five hundred rupees, or both.",
        ["six months|6 months", "five hundred|500"], ["AA-21"],
        [("AA-21", "be punished with imprisonment for a term which may extend to six months, or with fine "
                   "which may extend to five hundred rupees, or with both")]))]})

S.append({"session_id": "G34-T2-01", "type": "T2", "author": "G34", "turns": [
    T("In Taj Muhammad v. The State (Criminal Appeal 234/2016), decided by the Sindh High Court, Circuit at "
      "Hyderabad on 23 December 2016, how much Daman was the appellant required to deposit before he was "
      "released after the compromise?",
      g("Rs. 25,000 as Daman. The offences were compounded under section 345 Cr.P.C. after the complainant "
        "forgave the appellant; the Daman was to be deposited with the Additional Registrar and the appellant "
        "was ordered to be released.",
        ["25,000|25000|twenty-five thousand|twenty five thousand"], ["SHC-112685"],
        [("SHC-112685", "the amount of Rs.25,000/- towards Daman will be deposited by the appellant with "
                        "the learned Additional Registrar of this Court")]))]})

S.append({"session_id": "G34-T2-02", "type": "T2", "author": "G34", "turns": [
    T("In Khalid Ali v. Mst Amnat (II.A. 10/2008), decided by the Sindh High Court, Circuit at Hyderabad on "
      "24 November 2017, did the court set aside the concurrent findings of the lower courts, and why or why not?",
      g("No. The second appeals were dismissed with no order as to costs, because at the stage of second appeal "
        "it is not open to re-appraise the evidence and the two concurrent findings were not in direct conflict "
        "with the record.",
        ["dismissed", "re-appraise|reappraise|re-appraisal|reappraisal|concurrent"], ["SHC-124973"],
        [("SHC-124973", "at the stage of second appeal it is not open to re-appraise of the evidence"),
         ("SHC-124973", "The appeals as such, are dismissed however, in the circumstances with no orders as to costs.")]))]})

S.append({"session_id": "G34-T3-01", "type": "T3", "author": "G34", "turns": [
    T("My cousin is only 16 years old. Last month he signed a written agreement to sell me his motorcycle."),
    T("I already paid him 40,000 rupees as advance."),
    T("Now his father says the deal is off and he will not hand over the motorcycle."),
    T("Our families are not on speaking terms since then."),
    T("Considering the detail about him I gave you in my first message, which section of the Contract Act, 1872 "
      "decides whether he could make this agreement at all?",
      g("Section 11 of the Contract Act, 1872: only a person who is of the age of majority, of sound mind and not "
        "disqualified by law is competent to contract. A 16-year-old has not reached majority, so he was not "
        "competent to contract.",
        ["11"], ["CA-11"],
        [("CA-11", "Every person is competent to contract who is of the age of majority according to the law "
                   "to which he is subject")],
        mem=True, dep=[1]))]})

S.append({"session_id": "G34-T3-02", "type": "T3", "author": "G34", "turns": [
    T("The trial court dismissed my civil suit for recovery of money as time-barred, even though the defendant "
      "never raised limitation as a defence."),
    T("I filed the suit in Hyderabad in 2023."),
    T("My lawyer says he was ill for several months before filing."),
    T("I still have the bank receipts for the original loan."),
    T("Based on what I told you at the start, was the court right to do that even though nobody objected? "
      "Which section supports your answer?",
      g("Yes. Under section 3 of the Limitation Act, 1908, every suit instituted after the prescribed period of "
        "limitation shall be dismissed, although limitation has not been set up as a defence (subject to "
        "sections 4 to 25).",
        ["section 3|s. 3|s.3|LA-3"], ["LA-3"],
        [("LA-3", "every suit institute, appeal preferred, and application made, after the period of limitation "
                  "prescribed therefore by the first schedule shall be dismissed, although limitation has not "
                  "been set up as a defence")],
        mem=True, dep=[1]))]})

S.append({"session_id": "G34-T4-01", "type": "T4", "author": "G34", "turns": [
    T("I am writing an assignment on section 460 of the Pakistan Penal Code, 1860, about persons jointly "
      "concerned in house-breaking by night."),
    T("My teacher wants the answer in simple language."),
    T("The submission is due on Friday."),
    T("Please also include the exact citation."),
    T("Has this section been amended? Which law made the change, and in which year?",
      g("Yes. The amendment note for section 460 of the Pakistan Penal Code, 1860 reads: Subs. by Act II of "
        "1997, ss. 8 & 9. So it was substituted by Act II of 1997.",
        ["1997"], ["PPC-460"],
        [("PPC-460", "Subs. by Act II of 1997, ss. 8 & 9.")], mem=True, dep=[1]))]})

S.append({"session_id": "G34-T4-02", "type": "T4", "author": "G34", "turns": [
    T("My contract dispute turns on section 54 of the Contract Act, 1872, on reciprocal promises where one "
      "promise must be performed first."),
    T("The other side has hired a senior advocate."),
    T("The hearing is next month in Karachi."),
    T("Has the text of that section ever been changed by a later law? Name the law and the year.",
      g("Yes. The amendment note for section 54 of the Contract Act, 1872 reads: Subs. by the Federal Laws "
        "(Revision and Declaration) Ordinance, 1981 (XXVII of 1981), s. 3 and 2nd Sch.",
        ["1981"], ["CA-54"],
        [("CA-54", "Subs. by the Federal Laws (Revision and Declaration) Ordinance, 1981 (XXVII of 1981), "
                   "s. 3 and 2nd Sch.")],
        mem=True, dep=[1]))]})

ABS = "I am not sure. The documents provided do not contain this law."
S.append({"session_id": "G34-T5-01", "type": "T5", "author": "G34", "turns": [
    T("My husband and I have been living apart for a year."),
    T("Under the Muslim Family Laws Ordinance, 1961, within how many days must a husband notify the Union "
      "Council chairman after pronouncing talaq?",
      g(ABS + " The Muslim Family Laws Ordinance, 1961 is not in the corpus.", [], [], [], abstain=True))]})

S.append({"session_id": "G34-T5-02", "type": "T5", "author": "G34", "turns": [
    T("I rent a shop in Karachi and my landlord wants me out."),
    T("Which section of the Sindh Rented Premises Ordinance, 1979 allows a landlord to seek eviction of a "
      "tenant for his personal bona fide need?",
      g(ABS + " The Sindh Rented Premises Ordinance, 1979 is not in the corpus.", [], [], [], abstain=True))]})

out = ROOT / "contrib" / "G34.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for s in S:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"wrote {len(S)} sessions to {out}")
