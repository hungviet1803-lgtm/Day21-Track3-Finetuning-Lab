#!/usr/bin/env python3
"""Write the dev-split winner from `results/prompt_dev.json` into `config.OPTIMIZED_PROMPT`.

Run between `prompt_dev.py` and NB2. Deterministic: the same prompt_dev.json always
produces the same config.py, so the SHA that NB2 freezes can be reproduced anywhere.
`verify.py` will then WARN that (b) differs from the shipped prompt — that is correct,
and REPORT.md must say it was made stronger, on which split, and by how much.

    python scripts/apply_prompt.py            # apply the winner
    python scripts/apply_prompt.py --check    # only print what would change
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "labkit" / "config.py"
BLOCK = re.compile(r'OPTIMIZED_PROMPT = """.*?"""', re.S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    dev = json.loads((ROOT / "results" / "prompt_dev.json").read_text(encoding="utf-8"))
    winner = dev["winner"]
    text = dev["candidates"][winner]["prompt"]
    if '"""' in text or text.endswith('"'):
        raise SystemExit("prompt contains triple quotes / ends with a quote; cannot embed safely")

    src = CONFIG.read_text(encoding="utf-8")
    if len(BLOCK.findall(src)) != 1:
        raise SystemExit("could not find exactly one OPTIMIZED_PROMPT block in config.py")
    new = BLOCK.sub(lambda _: f'OPTIMIZED_PROMPT = """{text}"""', src)
    scores = {k: v["target"] for k, v in dev["candidates"].items()}
    print(f"dev-split target: {scores}  -> winner: {winner}")
    if new == src:
        print("config.py already holds this prompt — nothing to do")
        return 0
    if args.check:
        print("would rewrite OPTIMIZED_PROMPT in", CONFIG)
        return 0
    CONFIG.write_text(new, encoding="utf-8")
    print("rewrote OPTIMIZED_PROMPT in", CONFIG.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
