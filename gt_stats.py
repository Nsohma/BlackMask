#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from collections import Counter

# -----------------------
# parsing helpers
# -----------------------
FRAME_RE = re.compile(r"^[A-Za-z0-9_]+:$")
IDCOUNT_RE = re.compile(r"^id_count\s*:\s*(\d+)\s*$", re.IGNORECASE)
SUBJ_ID_RE = re.compile(r"^id\s*:\s*(\d+)\s*$", re.IGNORECASE)

def norm(s: str | None) -> str | None:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    if t.lower() == "none":
        return None
    return t

def split_or(val: str) -> list[str]:
    """
    "three_quarter or high" -> ["three_quarter", "high"]
    """
    parts = re.split(r"\s+or\s+", val.strip())
    out = []
    for p in parts:
        p = norm(p)
        if p:
            out.append(p)
    return out

def parse_props(val: str) -> list[str]:
    """
    props : {杖、帽子、椅子、本}
    props : None
    props:{椅子}
    """
    v = val.strip()
    if not v or v.lower() == "none":
        return []
    m = re.search(r"\{(.+)\}", v)
    if not m:
        # "杖、帽子" みたいに波括弧なしが来た場合も拾う
        content = v
    else:
        content = m.group(1)

    items = re.split(r"[、,，]\s*", content.strip())
    out = []
    for it in items:
        it = norm(it)
        if it:
            out.append(it)
    return out

def kv_from_line(line: str) -> tuple[str, str] | None:
    """
    "cropping : bust" / "cropping:bust" / "camera_angle: three_quarter or high"
    """
    if ":" not in line:
        return None
    k, v = line.split(":", 1)
    k = k.strip().lower().replace(" ", "")
    v = v.strip()
    return k, v

def parse_gt(path: Path):
    """
    returns:
      frames: dict[frame_id] = {
        "id_count": int|None,
        "subjects": list[{"cropping":[...], "camera_angle":[...], "pose":[...], "expression":[...]}],
        "props": list[str],
        "interactions": list[str],
      }
    """
    frames = {}
    cur_frame = None
    cur_subj = None

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        # Frame header like "C003_C002:"
        if s.endswith(":") and FRAME_RE.match(s) and not s.lower().startswith("id:"):
            cur_frame = s[:-1].strip()
            frames[cur_frame] = {"id_count": None, "subjects": [], "props": [], "interactions": []}
            cur_subj = None
            continue

        if cur_frame is None:
            continue

        # id_count
        m = IDCOUNT_RE.match(s.replace(" ", ""))
        if m:
            frames[cur_frame]["id_count"] = int(m.group(1))
            continue

        # subject id start
        m = SUBJ_ID_RE.match(s.replace(" ", ""))
        if m:
            cur_subj = {"cropping": [], "camera_angle": [], "pose": [], "expression": []}
            frames[cur_frame]["subjects"].append(cur_subj)
            continue

        kv = kv_from_line(s)
        if kv is None:
            continue
        k, v = kv

        if k == "props":
            frames[cur_frame]["props"].extend(parse_props(v))
            continue

        if k == "interactions":
            # "gaze_contact or confrontation" をそれぞれカウント
            frames[cur_frame]["interactions"].extend(split_or(v))
            continue

        # subject attributes
        if cur_subj is None:
            # idが無い（群衆等）場合はスキップ
            continue

        if k in ("cropping", "camera_angle", "pose", "expression"):
            cur_subj[k].extend(split_or(v) if "or" in v else ([norm(v)] if norm(v) else []))

    return frames

# -----------------------
# stats / printing
# -----------------------
def pct(n, d):
    return (100.0 * n / d) if d else 0.0

def print_counter(title: str, ctr: Counter, denom: int, topn: int = 50, order_hint=None):
    print(f"=== {title} ===")
    if denom == 0:
        print("(no data)\n")
        return

    items = ctr.most_common()
    if order_hint:
        seen = set()
        ordered = []
        for k in order_hint:
            if k in ctr:
                ordered.append((k, ctr[k]))
                seen.add(k)
        for k, v in items:
            if k not in seen:
                ordered.append((k, v))
        items = ordered

    for k, v in items[:topn]:
        ks = str(k)
        print(f"{ks:>16s} : {v:6d}  ({pct(v, denom):6.2f}%)")
    if len(items) > topn:
        print(f"... ({len(items)-topn} more)")
    print("")

def main(gt_path: Path, mode: str, topn: int):
    frames = parse_gt(gt_path)

    scanned_frames = len(frames)
    frames_with_subjects = sum(1 for f in frames.values() if len(f["subjects"]) > 0)

    # id_count: frame-level
    id_count = Counter()
    sum_id_count = 0
    for fid, f in frames.items():
        n = f["id_count"]
        if isinstance(n, int):
            id_count[n] += 1
            sum_id_count += n
        else:
            id_count["(missing)"] += 1

    # subject-level totals (only described subjects)
    total_subjects = sum(len(f["subjects"]) for f in frames.values())

    cropping = Counter()
    camera_angle = Counter()
    pose = Counter()
    expression = Counter()

    props = Counter()
    interactions = Counter()

    def add_attr(counter: Counter, vals: list[str]):
        vals = [norm(x) for x in vals if norm(x)]
        if not vals:
            return
        if mode == "exclusive_first":
            counter[vals[0]] += 1
        else:
            # presence: その人物が含んでいる候補を全部1回ずつ加算
            for v in vals:
                counter[v] += 1

    # collect
    for fid, f in frames.items():
        for p in f["props"]:
            props[p] += 1
        for t in f["interactions"]:
            interactions[t] += 1

        for s in f["subjects"]:
            add_attr(cropping, s.get("cropping", []))
            add_attr(camera_angle, s.get("camera_angle", []))
            add_attr(pose, s.get("pose", []))
            add_attr(expression, s.get("expression", []))

    # ---------------- print ----------------
    print(f"scanned_frames={scanned_frames}")
    print(f"frames_with_subjects={frames_with_subjects}")
    print(f"total_subjects(described)={total_subjects}")
    print(f"sum_id_count(from GT)={sum_id_count}")
    print(f"mode={mode}  (presence: multi-count / exclusive_first: one-per-subject)")
    print("")

    print_counter("id_count per frame", id_count, denom=scanned_frames, topn=topn,
                  order_hint=[0,1,2,3,4,5,6,7,8,9,10,15,20,25])

    # denom は「人物数」で割合を出す（presenceだと合計100%超えうる）
    print_counter("cropping (per subject)", cropping, denom=total_subjects, topn=topn,
                  order_hint=["wide","close_up","bust","half","full"])
    print_counter("camera_angle (per subject)", camera_angle, denom=total_subjects, topn=topn)
    print_counter("pose (per subject)", pose, denom=total_subjects, topn=topn)
    print_counter("expression (per subject)", expression, denom=total_subjects, topn=topn)

    # props/interactions は “出現回数” を分母に（参考割合）
    print_counter("props", props, denom=sum(props.values()) if props else 0, topn=topn)
    print_counter("interactions(type)", interactions, denom=sum(interactions.values()) if interactions else 0, topn=topn)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True, help="correct_qwenvl.txt")
    ap.add_argument("--mode", choices=["presence", "exclusive_first"], default="presence",
                    help="orの扱い：presence=全部数える / exclusive_first=先頭だけ数える")
    ap.add_argument("--topn", type=int, default=50)
    args = ap.parse_args()
    main(args.gt, args.mode, args.topn)
