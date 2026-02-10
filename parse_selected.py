#!/usr/bin/env python3
# parse_selected_counts.py
import re
import sys
from pathlib import Path

OK_RE = re.compile(r"^\[OK\]\s+(?P<name>\S+)\s+selected=(?P<n>\d+)\b")
SKIP_RE = re.compile(r"^\[SKIP\]\s+bbox not found:\s+(?P<fname>\S+)")

def iter_lines(input_path: str):
    if input_path == "-":
        for line in sys.stdin:
            yield line.rstrip("\n")
    else:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")

def main(argv: list[str]) -> int:
    # デフォルトは同一ディレクトリの log_detection_swiml_v1.txt を読む
    input_path = argv[1] if len(argv) >= 2 else "log_detection_swiml_v1.txt"

    seen = {}  # name -> count
    for line in iter_lines(input_path):
        m = OK_RE.match(line)
        if m:
            name = m.group("name")
            n = int(m.group("n"))
            if name in seen and seen[name] != n:
                print(f"WARNING: duplicate name with different counts: {name} {seen[name]} -> {n}",
                      file=sys.stderr)
            seen[name] = n
            continue

        m = SKIP_RE.match(line)
        if m:
            fname = m.group("fname")
            # .json のstemにしたい場合は下の1行を fname = Path(fname).stem に変更
            name = fname
            if name not in seen:
                seen[name] = 0
            continue

    # 出力（出現順を保ちたいので、seenは挿入順のまま走査）
    for name, n in seen.items():
        print(f"{name}\t{n}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

