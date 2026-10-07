#!/usr/bin/env python3
"""Choose baseline (b)'s prompt on the DEV split, before NB2 freezes it.

Why this exists. Rubric §"Trước khi nộp": making (b) stronger is welcome, weakening it
is cheating. But "stronger" has to be decided by a measurement, and that measurement
must NOT be taken on `data/eval_target.jsonl` — tuning a prompt on the set it is later
graded on is the same leak as editing the eval set after seeing results.

So candidates are scored on `data/split/val.jsonl`: the 25 records NB1 holds out of
`train_seed.jsonl`. They are never trained on (NB3/NB4 read `split/train.jsonl` only)
and never graded on. Few-shot examples and the labeling guide are derived from
`split/train.jsonl` only — the same information the fine-tune gets, no more.

    python scripts/prompt_dev.py                 # GPU (Colab T4), uses the tier's dtype
    python scripts/prompt_dev.py --cpu-bf16      # CPU fallback, bf16 weights (~8 GB RAM)
    python scripts/prompt_dev.py --print-only    # just print the candidate prompts

Writes `results/prompt_dev.json`. Copy the winner into `config.OPTIMIZED_PROMPT` and
only then run NB2.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report      # noqa: E402
from labkit.config import OPTIMIZED_PROMPT, get_tier      # noqa: E402

FIELDS = ("intent", "urgency", "sentiment")


def load_jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def cue_table(train_rows: list[dict]) -> dict[str, dict[str, list[str]]]:
    """label -> cue phrases, per field, mined from the TRAIN split.

    Tickets are `<opener> mình đặt <product> mã đơn <id>. <intent>. <urgency>. <sentiment>.`
    A cue is only kept if it maps to ONE label everywhere in train; an ambiguous cue in
    a prompt would be a guess presented as a rule.
    """
    seen: dict[str, dict[str, collections.Counter]] = {f: collections.defaultdict(collections.Counter)
                                                       for f in FIELDS}
    for r in train_rows:
        parts = [p.strip().rstrip(".") for p in r["input"].split(". ")]
        for field, phrase in zip(FIELDS, parts[1:4]):
            seen[field][phrase.lower()][r["label"][field]] += 1
    table: dict[str, dict[str, list[str]]] = {}
    for field, cues in seen.items():
        table[field] = collections.defaultdict(list)
        for phrase, labels in sorted(cues.items()):
            if len(labels) == 1:
                table[field][next(iter(labels))].append(phrase)
    return table


def fewshot_examples(train_rows: list[dict], k: int, seed: int = 42) -> list[dict]:
    """k train examples, round-robin over intents so every class is shown."""
    rng = random.Random(seed)
    by_intent: dict[str, list[dict]] = collections.defaultdict(list)
    for r in train_rows:
        by_intent[r["label"]["intent"]].append(r)
    for v in by_intent.values():
        rng.shuffle(v)
    out, keys = [], sorted(by_intent)
    while len(out) < k:
        for key in keys:
            if by_intent[key] and len(out) < k:
                out.append(by_intent[key].pop())
    return out


HEADER = (
    "Bạn là hệ thống phân loại ticket CSKH. Trả về DUY NHẤT một object JSON, không kèm "
    "giải thích, không kèm markdown fence.\n\n"
    "Schema bắt buộc — đúng 4 khóa:\n"
    '{"intent": ..., "urgency": ..., "product": ..., "sentiment": ...}\n\n'
    "intent    ∈ doi_tra | van_chuyen | hoan_tien | san_pham_loi | hoi_thong_tin\n"
    "urgency   ∈ cao | trung_binh | thap\n"
    "sentiment ∈ tieu_cuc | trung_tinh | tich_cuc\n"
    "product   = tên sản phẩm xuất hiện nguyên văn trong ticket (giữ nguyên chữ như trong "
    "ticket, không kèm mã đơn)"
)


def render_examples(rows: list[dict]) -> str:
    return "\n".join(
        f'Ticket: "{r["input"]}"\nJSON: {json.dumps(r["label"], ensure_ascii=False)}'
        for r in rows)


def render_guide(table: dict[str, dict[str, list[str]]]) -> str:
    lines = ["Hướng dẫn gán nhãn nội bộ — áp dụng theo CỤM TỪ trong ticket, kể cả khi "
             "nghĩa thông thường gợi ý nhãn khác:"]
    for field in FIELDS:
        lines.append(f"{field}:")
        for label, cues in sorted(table[field].items()):
            lines.append(f"  - {label}: " + ", ".join(f'"{c}"' for c in cues))
    lines.append('Lưu ý: "hoàn lại" là trả hàng (doi_tra), KHÔNG phải hoan_tien. Mỗi trường '
                 "được quyết định độc lập: sentiment chỉ theo câu thể hiện thái độ, không theo "
                 "việc sản phẩm có lỗi hay không.")
    return "\n".join(lines)


def build_candidates(train_rows: list[dict]) -> dict[str, str]:
    table = cue_table(train_rows)
    shots = fewshot_examples(train_rows, k=8)
    guide = render_guide(table)
    return {
        "shipped": OPTIMIZED_PROMPT,
        "fewshot8": f"{HEADER}\n\nVí dụ:\n{render_examples(shots)}",
        "guide": f"{HEADER}\n\n{guide}\n\nVí dụ:\n{render_examples(shots[:3])}",
    }


def per_field(preds: list[str], rows: list[dict]) -> dict[str, float]:
    acc = {k: 0 for k in ev.TRIAGE_KEYS}
    for p, r in zip(preds, rows):
        obj = ev._parse_json_loose(p)
        if not isinstance(obj, dict):
            continue
        for k in ev.TRIAGE_KEYS:
            got, want = obj.get(k), r["label"][k]
            if got is None:
                continue
            same = (ev.normalize(str(got)) == ev.normalize(want) if k == "product"
                    else str(got).strip().lower() == want)
            acc[k] += int(same)
    return {k: round(v / len(rows), 3) for k, v in acc.items()}


def load_cpu_bf16(tier):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tier.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(tier.model_id, trust_remote_code=True,
                                                 dtype=torch.bfloat16)
    return model, tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-bf16", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated candidate names")
    args = ap.parse_args()

    split = ROOT / "data" / "split"
    if not (split / "val.jsonl").exists():
        raise SystemExit("run NB1 first — data/split/val.jsonl is the dev set")
    train_rows, val_rows = load_jsonl(split / "train.jsonl"), load_jsonl(split / "val.jsonl")

    # Guard: the dev set must be disjoint from the graded set, or this is eval-set tuning.
    graded = {r["input"] for r in load_jsonl(ROOT / "data" / "eval_target.jsonl")}
    assert not graded & {r["input"] for r in val_rows + train_rows}, "dev/train overlaps eval_target"

    cands = build_candidates(train_rows)
    if args.only:
        cands = {k: v for k, v in cands.items() if k in args.only.split(",")}
    if args.print_only:
        for name, text in cands.items():
            print(f"===== {name} ({len(text)} chars)\n{text}\n")
        return 0

    tier = get_tier()
    model, tok = load_cpu_bf16(tier) if args.cpu_bf16 else generate.load_base(tier)
    out = {"model": tier.model_id, "dev_set": "data/split/val.jsonl", "n": len(val_rows),
           "candidates": {}}
    for name, text in cands.items():
        t0 = time.perf_counter()
        preds, lat = generate.generate_batch(model, tok, [r["input"] for r in val_rows],
                                             system=text, label=name)
        tgt = sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, val_rows)) / len(val_rows)
        fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)
        out["candidates"][name] = {
            "target": round(tgt, 4), "format": round(fmt, 4), "latency_ms": round(lat, 1),
            "per_field": per_field(preds, val_rows), "prompt": text,
            "wall_s": round(time.perf_counter() - t0, 1),
            "errors": [{"ticket": r["input"], "label": r["label"], "pred": p}
                       for p, r in zip(preds, val_rows)
                       if ev.triage_field_accuracy(p, r["label"]) < 1.0][:8],
        }
        print(f"{name:10s} target={tgt:.3f} format={fmt:.3f} {lat:.0f}ms "
              f"{out['candidates'][name]['per_field']}", flush=True)
        report.write_json(out, "prompt_dev.json", results_dir=ROOT / "results")

    best = max(out["candidates"], key=lambda k: (out["candidates"][k]["target"],
                                                 -out["candidates"][k]["latency_ms"]))
    out["winner"] = best
    report.write_json(out, "prompt_dev.json", results_dir=ROOT / "results")
    print(f"\nwinner on dev split: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
