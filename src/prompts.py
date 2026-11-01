"""Prompt builder and controls.

Conditions
  multi      k questions in one prompt (k=1 is the baseline)
  padding    k=1 question preceded by neutral filler matched in length to the k=4 prompt
  statement  k=1 question followed by k-1 declarative statements (not questions)
  empty      k=1 question plus k-1 numbered slots with no content
  concise    k=1 question with one added instruction asking for concise reasoning (our wording, not ConPress's)

NOTE: the statement and empty controls are OUR reading of the ConPress ablation design.
Verify against the ConPress PDF (arXiv 2602.01472) before describing them as theirs.
"""

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
CONCISE_LINE = "Keep your reasoning concise."
MULTI_LINE = "Answer each question separately and put each final answer in its own \\boxed{}, in order."

# Neutral, question-free filler sentences (deterministic, cycled to the needed length).
FILLER = [
    "The library opens at nine in the morning and closes at six in the evening.",
    "A river flows through the valley and joins the sea near a small harbour.",
    "Autumn leaves covered the path behind the old school building.",
    "The committee met on Thursday to review the schedule for the coming month.",
    "Several types of bread were displayed on the wooden shelves of the bakery.",
    "The train from the northern station usually arrives a few minutes early.",
    "A gentle wind moved the curtains in the quiet room upstairs.",
    "The museum added a new gallery with paintings from the last century.",
]

# Declarative statements for the statement control (no question, no task).
STATEMENTS = [
    "The sky is often blue on clear days.",
    "Water boils at a lower temperature at high altitude.",
    "Many cities have a central park.",
    "Libraries lend books to their members.",
]


def system_text(k_slots: int) -> str:
    return SYSTEM if k_slots <= 1 else SYSTEM + "\n" + MULTI_LINE


def make_filler(n_chars: int) -> str:
    out, i = [], 0
    while sum(len(s) + 1 for s in out) < n_chars:
        out.append(FILLER[i % len(FILLER)])
        i += 1
    return " ".join(out)[:n_chars].rsplit(" ", 1)[0] + "."


def user_text(questions: list[str]) -> str:
    return "\n\n".join(f"Question {i + 1}: {q}" for i, q in enumerate(questions))


def build_messages(condition: str, questions: list[str], k_ref: int = 4, ref_chars: int = 0):
    """Return (messages, k_expected). k_expected = number of boxed answers expected.

    condition: 'multi' | 'padding' | 'statement' | 'empty'
    For 'padding', ref_chars is the character length of the matching k=k_ref user text.
    """
    if condition == "multi":
        k = len(questions)
        return [{"role": "system", "content": system_text(k)},
                {"role": "user", "content": user_text(questions)}], k
    assert len(questions) == 1, "controls apply to k=1 questions"
    q = questions[0]
    if condition == "concise":
        return [{"role": "system", "content": system_text(1) + "\n" + CONCISE_LINE},
                {"role": "user", "content": user_text([q])}], 1
    if condition == "padding":
        base = user_text([q])
        filler = make_filler(max(0, ref_chars - len(base) - 2))
        return [{"role": "system", "content": system_text(1)},
                {"role": "user", "content": filler + "\n\n" + base}], 1
    if condition == "statement":
        extra = "\n\n".join(f"Statement {i + 2}: {STATEMENTS[i % len(STATEMENTS)]}" for i in range(k_ref - 1))
        return [{"role": "system", "content": system_text(1)},
                {"role": "user", "content": user_text([q]) + "\n\n" + extra}], 1
    if condition == "empty":
        extra = "\n\n".join(f"Question {i + 2}:" for i in range(k_ref - 1))
        return [{"role": "system", "content": system_text(1)},
                {"role": "user", "content": user_text([q]) + "\n\n" + extra}], 1
    raise ValueError(condition)
