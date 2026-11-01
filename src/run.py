"""Resumable async client for a llama.cpp server. Usage: python src/run.py configs/<name>.yaml

Appends one JSON line per generation to runs/<run_id>.jsonl. Finished units are skipped on restart.
Raw logs are never rewritten.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_messages, user_text  # noqa: E402
from score import split_think  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_units(cfg, items):
    n, cond, k = cfg["n_questions"], cfg["condition"], cfg["k"]
    items = items[:n]
    units = []
    if cond == "multi":
        assert n % k == 0
        for g in range(0, n, k):
            grp = items[g:g + k]
            units.append({"unit_id": f"multi_k{k}_g{g // k}", "items": grp,
                          "messages_args": ("multi", [x["question"] for x in grp], k, 0)})
    else:
        kref = cfg.get("k_ref", 4)
        for i, it in enumerate(items):
            g0 = (i // kref) * kref
            ref = len(user_text([x["question"] for x in items[g0:g0 + kref]]))
            units.append({"unit_id": f"{cond}_q{i}", "items": [it],
                          "messages_args": (cond, [it["question"]], kref, ref)})
    return units


async def one(client, cfg, u, props, out_f, lock, sem):
    cond, qs, kref, ref = u["messages_args"]
    msgs, k = build_messages(cond, qs, k_ref=kref, ref_chars=ref)
    s = cfg["sampling"]
    async with sem:
        t0 = time.time()
        rendered = (await client.post("/apply-template", json={"messages": msgs})).json().get("prompt")
        body = {"model": cfg["model"], "messages": msgs, "max_tokens": cfg["max_tokens"],
                "temperature": s["temperature"], "top_p": s["top_p"], "top_k": s["top_k"], "min_p": s["min_p"],
                "seed": cfg["seed"] * 1_000_003 + sum(map(ord, u["unit_id"]))}  # deterministic per unit
        r = (await client.post("/v1/chat/completions", json=body)).json()
        if "choices" not in r:  # server error: skip, the next pass of the queue retries this unit
            print("ERROR", u["unit_id"], str(r)[:200])
            return
        ch = r["choices"][0]
        raw = ch["message"].get("content") or ""
        think, _, _ = split_think(raw)
        think_tok = len((await client.post("/tokenize", json={"content": think})).json().get("tokens", []))
        rec = {
            "run_id": cfg["run_id"], "unit_id": u["unit_id"], "model": cfg["model"], "gguf": cfg["gguf"],
            "precision": cfg["precision"], "condition": cond, "k": k, "seed": cfg["seed"],
            "question_pos": [x["pos"] for x in u["items"]], "test_idx": [x["test_idx"] for x in u["items"]],
            "gold": [x["gold"] if "gold" in x else x["answer"].split("####")[-1].strip().replace(",", "")
                     for x in u["items"]],
            "rendered_prompt": rendered, "raw_output": raw, "finish_reason": ch["finish_reason"],
            "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
            "think_tokens": think_tok, "timings": r.get("timings"), "wall_s": time.time() - t0,
        }
    async with lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()


async def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8-sig"))
    items = json.loads((ROOT / "data" / cfg.get("data_file", "gsm8k_300.json")).read_text())["items"]
    units = make_units(cfg, items)
    out_path = ROOT / "runs" / f"{cfg['run_id']}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["unit_id"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [u for u in units if u["unit_id"] not in done]
    print(f"{cfg['run_id']}: {len(done)} done, {len(todo)} to run")
    async with httpx.AsyncClient(base_url=cfg["server"], timeout=None) as client:
        props = (await client.get("/props")).json()
        meta_path = ROOT / "runs" / f"{cfg['run_id']}.meta.json"
        if not meta_path.exists():
            ex = build_messages(*todo[0]["messages_args"][:2], k_ref=todo[0]["messages_args"][2],
                                ref_chars=todo[0]["messages_args"][3])[0] if todo else []
            rendered = (await client.post("/apply-template", json={"messages": ex})).json().get("prompt")
            meta_path.write_text(json.dumps({"config": cfg, "server_props": props, "rendered_example": rendered,
                                             "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
            print("--- rendered example prompt ---\n" + str(rendered) + "\n-------------------------------")
        sem, lock = asyncio.Semaphore(cfg["slots"]), asyncio.Lock()
        with open(out_path, "a", encoding="utf-8") as f:
            res = await asyncio.gather(*(one(client, cfg, u, props, f, lock, sem) for u in todo),
                                       return_exceptions=True)
            for e in res:
                if isinstance(e, Exception):
                    print("ERROR", repr(e))
    print("finished", cfg["run_id"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
