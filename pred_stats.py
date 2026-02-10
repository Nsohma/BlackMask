#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------
# robust helpers
# ---------------------------
def load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    # top-level が list の場合は dict を拾う
    if isinstance(data, list):
        for el in data:
            if isinstance(el, dict):
                return el
        return None
    if isinstance(data, dict):
        return data
    return None

def get_value_field(x):
    """
    cropping/camera_angle が
      - {"value": "...", "confidence": ...}
      - "..."
    などの揺れを吸収して値だけ返す
    """
    if x is None:
        return None
    if isinstance(x, dict):
        v = x.get("value")
        return str(v).strip() if v is not None else None
    if isinstance(x, str):
        return x.strip()
    return None

def extract_desc_list(x):
    """
    pose/expression が
      - [{"description":"...", "confidence":...}, ...]
      - ["...", "..."]
      - {"description": "..."} など
    を吸収して [(desc, conf_or_None), ...] を返す
    """
    out = []
    if x is None:
        return out
    if isinstance(x, dict):
        d = x.get("description") or x.get("value")
        c = x.get("confidence")
        if d is not None:
            out.append((str(d).strip(), float(c) if c is not None else None))
        return out
    if isinstance(x, str):
        return [(x.strip(), None)]
    if isinstance(x, list):
        for el in x:
            if isinstance(el, dict):
                d = el.get("description") or el.get("value")
                c = el.get("confidence")
                if d is not None:
                    out.append((str(d).strip(), float(c) if c is not None else None))
            elif isinstance(el, str):
                out.append((el.strip(), None))
    return out

def choose_top1(descs):
    """
    [(desc, conf), ...] から最も確からしい1つを選ぶ
    confが無ければ先頭
    """
    if not descs:
        return None
    if all(c is None for _, c in descs):
        return descs[0][0]
    best = max(descs, key=lambda t: (-1.0 if t[1] is None else t[1]))
    return best[0]

def norm_token(s: str | None):
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    tl = t.lower()
    if tl == "none":
        return None
    return t

def pct(n, d):
    return (100.0 * n / d) if d else 0.0

# ---------------------------
# main
# ---------------------------
def main(pred_dir: Path, pattern: str, pose_mode: str, expr_mode: str):
    # 集計器
    files = 0
    frames_with_subjects = 0

    id_count = Counter()
    cropping = Counter()
    camera_angle = Counter()
    pose = Counter()
    expression = Counter()

    props = Counter()
    interactions = Counter()

    # 何件サブジェクト（人）を見たか
    n_subjects_total = 0

    for p in pred_dir.rglob(pattern):
        if p.name.endswith(".meta.json"):
            continue
        data = load_json(p)
        if data is None:
            continue

        files += 1

        subjects = data.get("subjects") or []
        if isinstance(subjects, list) and len(subjects) > 0:
            frames_with_subjects += 1

        # 人数
        M = len(subjects) if isinstance(subjects, list) else 0
        id_count[M] += 1

        # props
        pr = data.get("props")
        if isinstance(pr, list):
            for it in pr:
                if isinstance(it, dict):
                    nm = norm_token(it.get("name"))
                else:
                    nm = norm_token(it)
                if nm:
                    props[nm] += 1

        # interactions（typeのみ）
        inter = data.get("interactions")
        if isinstance(inter, list):
            for it in inter:
                if isinstance(it, dict):
                    t = norm_token(it.get("type"))
                else:
                    t = norm_token(it)
                if t:
                    interactions[t] += 1

        # subjects の属性
        if not isinstance(subjects, list):
            continue
        for s in subjects:
            if not isinstance(s, dict):
                continue
            n_subjects_total += 1

            c = norm_token(get_value_field(s.get("cropping")))
            if c:
                cropping[c] += 1

            a = norm_token(get_value_field(s.get("camera_angle")))
            if a:
                camera_angle[a] += 1

            # pose
            p_list = [(norm_token(d), conf) for (d, conf) in extract_desc_list(s.get("pose"))]
            p_list = [(d, conf) for (d, conf) in p_list if d]
            if pose_mode == "all":
                for d, _ in p_list:
                    pose[d] += 1
            else:  # top1
                d = choose_top1(p_list)
                d = norm_token(d)
                if d:
                    pose[d] += 1

            # expression
            e_list = [(norm_token(d), conf) for (d, conf) in extract_desc_list(s.get("expression"))]
            e_list = [(d, conf) for (d, conf) in e_list if d]
            if expr_mode == "all":
                for d, _ in e_list:
                    expression[d] += 1
            else:
                d = choose_top1(e_list)
                d = norm_token(d)
                if d:
                    expression[d] += 1

    # ---------------------------
    # print
    # ---------------------------
    print(f"scanned_files={files}")
    print(f"frames_with_subjects={frames_with_subjects}")
    print(f"total_subjects={n_subjects_total}")
    print("")

    def print_counter(title, ctr: Counter, denom: int, topn: int = 50, order_hint=None):
        print(f"=== {title} ===")
        if denom == 0:
            print("(no data)\n")
            return
        items = ctr.most_common()
        if order_hint:
            # hintにあるキーを先に、その後残りを頻度順
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

    # 人数分布（フレーム単位）
    print_counter("id_count per frame", id_count, denom=files, order_hint=[0,1,2,3,4,5,6,7,8,9,10])

    # cropping / camera_angle は「人物単位」で分母=total_subjects
    print_counter("cropping (per subject)", cropping, denom=n_subjects_total,
                  order_hint=["wide","close_up","bust","half","full"])
    print_counter("camera_angle (per subject)", camera_angle, denom=n_subjects_total)

    # pose/expression は mode により分母が変わる（top1なら subjects、allなら “出力された候補総数”）
    if pose_mode == "top1":
        pose_denom = n_subjects_total
    else:
        pose_denom = sum(pose.values())
    if expr_mode == "top1":
        expr_denom = n_subjects_total
    else:
        expr_denom = sum(expression.values())

    print_counter(f"pose (mode={pose_mode})", pose, denom=pose_denom)
    print_counter(f"expression (mode={expr_mode})", expression, denom=expr_denom)

    # props / interactions は “出現した要素数” を分母に（割合は参考）
    print_counter("props", props, denom=sum(props.values()) if props else 0)
    print_counter("interactions(type)", interactions, denom=sum(interactions.values()) if interactions else 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, required=True, help="Qwen予測jsonがあるディレクトリ")
    ap.add_argument("--pattern", type=str, default="*.json",
                    help="読み込むファイルパターン（例: '*_all.png.json'）")
    ap.add_argument("--pose_mode", choices=["top1","all"], default="top1",
                    help="poseを1人1件(top1)で数えるか、全候補(all)で数えるか")
    ap.add_argument("--expr_mode", choices=["top1","all"], default="top1",
                    help="expressionを1人1件(top1)で数えるか、全候補(all)で数えるか")
    args = ap.parse_args()
    main(args.pred_dir, args.pattern, args.pose_mode, args.expr_mode)

