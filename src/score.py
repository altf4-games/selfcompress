"""Scoring: Math-Verify per question index, omission / cap / loop / empty-think detectors."""
import logging
import re
from collections import Counter

logging.getLogger().setLevel(logging.ERROR)  # Math-Verify warns on every call when timeouts are off

try:
    from math_verify import parse, verify
except ImportError:  # keeps the detectors importable without math-verify
    parse = verify = None


def split_think(raw: str):
    """Return (think_text, answer_text, has_close). Handles templates that prepend <think>."""
    if "</think>" in raw:
        think, ans = raw.split("</think>", 1)
        return think.replace("<think>", "").strip(), ans.strip(), True
    return raw.replace("<think>", "").strip(), "", False


def extract_boxed(text: str) -> list[str]:
    """All \\boxed{...} contents in order, with brace matching."""
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            return out
        k, depth = j + len("\\boxed{"), 1
        while k < len(text) and depth:
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
        if depth == 0:
            out.append(text[j + len("\\boxed{"):k - 1])
        i = k


def gold_parsed(g: str):
    """Parse a gold answer: plain numbers (GSM8K) directly, LaTeX answers (MATH) inside $...$ if the bare parse is empty."""
    if re.fullmatch(r"-?[\d,\.]+", g.strip()):  # plain number (all GSM8K golds): bare parse, as before
        r = parse(g, parsing_timeout=None)
        if r:
            return r
    # LaTeX gold: parse as math. A bare parse of e.g. "2k+2" would return only the number 2 and mark correct answers wrong.
    return parse("$" + g + "$", parsing_timeout=None)


def gold_answer(gsm_answer: str) -> str:
    return gsm_answer.split("####")[-1].strip().replace(",", "")


def is_loop(text: str, n: int = 10, thresh: int = 15) -> bool:
    """Repeated word n-gram detector. Threshold is a tunable guess; hand-check validates it."""
    w = text.split()
    if len(w) < n * 2:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return c.most_common(1)[0][1] >= thresh


def is_degenerate(text: str, n: int = 10, tail_words: int = 1500, thresh: int = 50) -> bool:
    """Degenerate repetition: one word n-gram repeated >= thresh times within the last tail_words words.
    Threshold chosen from the 76 capped GSM8K generations of the main grid: tail counts are 2-21 for ordinary long
    reasoning and 249-1490 for true loops (nothing between), so any threshold in 22..248 gives the same split."""
    w = text.split()[-tail_words:]
    if len(w) < n * 2:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return c.most_common(1)[0][1] >= thresh


def score_generation(raw: str, golds: list[str], finish_reason: str) -> dict:
    """golds: gold answer string per question, in order. Answers are taken from the text after </think>."""
    k = len(golds)
    think, ans, closed = split_think(raw)
    boxed = extract_boxed(ans) if closed else []
    extra = len(boxed) > k
    used = boxed[-k:] if extra else boxed  # if over-boxed, take the last k (documented choice)
    correct, omitted = [], []
    for i, g in enumerate(golds):
        if i < len(used):
            omitted.append(False)
            ok = False
            if verify is not None:
                try:
                    # Timeouts are disabled: Math-Verify's multiprocessing timeout fails on Windows.
                    ok = bool(verify(gold_parsed(g),
                                     parse("\\boxed{" + used[i] + "}", parsing_timeout=None), timeout_seconds=None))
                except Exception:
                    ok = False
            correct.append(ok)
        else:
            omitted.append(True)
            correct.append(False)
    return {
        "k": k, "correct": correct, "omitted": omitted, "n_boxed": len(boxed), "extra_boxed": extra,
        "cap_hit": finish_reason == "length", "think_closed": closed,
        "empty_think": closed and len(think) < 10,
        # repeat = repeated n-gram anywhere (includes harmless re-reading of the problem);
        # loop = cap hit AND degenerate repetition in the tail (see is_degenerate).
        "repeat": is_loop(raw), "loop": finish_reason == "length" and is_degenerate(raw), "think_chars": len(think), "answer_chars": len(ans),
    }
