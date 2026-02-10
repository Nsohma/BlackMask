#!/usr/bin/env python3
# eval_qwenvl_aligned.py
"""
Evaluate Qwen3-VL tags with GT (correct_qwenvl.txt) using order-preserving optimal alignment (DP).

- GT format:
  <FRAME_ID>:
  id_count: <int>
  id: 1
  cropping: ...
  camera_angle: ...
  pose: ...
  expression: ...
  id: 2
  ...
  (optional)
  interactions: ...
  props:{...}   or  props:None

- Pred format (Qwen JSON):
  <FRAME_ID>_all.png.json  (or <FRAME_ID>.png.json)
  {
    "subjects":[ { "bbox":[...], "cropping":{value}, "camera_angle":{value}, "pose":[...], "expression":[...] }, ... ],
    "interactions":[ { "type": "...", ...}, ... ],
    "props":[ { "name":"...", ...}, ... ]
  }

Evaluation:
  (A) id_count: exact acc, MAE, within-1 acc
  (B) per-subject attributes: optimal monotone alignment via DP
      - alignment score uses only GT-non-None attributes to avoid "None" skew
      - reports per-attribute accuracy (include/exclude None)
      - reports aligned_mean (divide by min(N,M)) and penalized_mean (divide by max(N,M))
  (C) props/interactions: set-based micro P/R/F1 + avg Jaccard + exact match
"""

import json
import re
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

NONE = "__NONE__"

# -------------------------
# Parsing helpers
# -------------------------

def norm_key(s: str) -> str:
    return s.strip().lower()

def parse_or_none(value: str, *, lower: bool) -> Set[str]:
    """
    Parse "a or b" into {"a","b"}.
    Parse "None" into {NONE}.
    """
    v = value.strip()
    if not v:
        return {NONE}
    if v.lower() == "none":
        return {NONE}

    # split by "or"
    parts = [p.strip() for p in re.split(r"\bor\b", v) if p.strip()]
    if lower:
        parts = [p.lower() for p in parts]
    return set(parts) if parts else {NONE}

def split_props_braces(s: str) -> Set[str]:
    """
    props:{本、椅子、机}  -> {"本","椅子","机"}
    Accept both full-width and half-width commas.
    """
    v = s.strip()
    if not v or v.lower() == "none":
        return set()
    # remove "props:" prefix if present
    if ":" in v:
        _, v = v.split(":", 1)
        v = v.strip()
    v = v.strip("{}[]（）() \t")
    if not v:
        return set()
    items = re.split(r"[、,，\s]+", v)
    return {it.strip() for it in items if it.strip()}

@dataclass
class GTSubject:
    cropping: Set[str]
    camera_angle: Set[str]
    pose: Set[str]
    expression: Set[str]

@dataclass
class GTRecord:
    frame_id: str
    id_count: int
    subjects: List[GTSubject]          # empty for crowd-only
    props: Set[str]
    interactions: Set[str]             # types only (set, or-expanded), empty if none/absent
    crowd_only: bool                   # True if only id_count exists

@dataclass
class PredSubject:
    # normalized sets (cropping/camera_angle lower, pose/expression keep as-is but stripped)
    cropping: Set[str]
    camera_angle: Set[str]
    pose: Set[str]
    expression: Set[str]
    cx: float                          # bbox center x for sorting

@dataclass
class PredRecord:
    frame_id: str
    subjects: List[PredSubject]
    props: Set[str]
    interactions: Set[str]

def parse_gt_file(gt_path: Path) -> Dict[str, GTRecord]:
    """
    Parse correct_qwenvl.txt according to the described format.
    """
    text = gt_path.read_text(encoding="utf-8", errors="replace").splitlines()

    records: Dict[str, GTRecord] = {}
    current_id: Optional[str] = None
    id_count: Optional[int] = None
    subjects: List[GTSubject] = []
    props: Set[str] = set()
    interactions: Set[str] = set()

    # temp for current subject block
    in_subj = False
    subj_tmp = {"cropping": {NONE}, "camera_angle": {NONE}, "pose": {NONE}, "expression": {NONE}}

    def flush_record():
        nonlocal current_id, id_count, subjects, props, interactions, in_subj, subj_tmp
        if current_id is None:
            return
        if in_subj:
            subjects.append(GTSubject(
            cropping=subj_tmp["cropping"],
            camera_angle=subj_tmp["camera_angle"],
            pose=subj_tmp["pose"],
            expression=subj_tmp["expression"],
        ))
            in_subj = False
        if id_count is None:
            # if missing, infer from subjects length
            idc = len(subjects)
        else:
            idc = id_count
        crowd_only = (len(subjects) == 0)  # by your rule: crowd => id_count only
        records[current_id] = GTRecord(
            frame_id=current_id,
            id_count=int(idc),
            subjects=list(subjects),
            props=set(props),
            interactions=set(interactions),
            crowd_only=crowd_only
        )
        # reset
        current_id = None
        id_count = None
        subjects = []
        props = set()
        interactions = set()
        in_subj = False
        subj_tmp = {"cropping": {NONE}, "camera_angle": {NONE}, "pose": {NONE}, "expression": {NONE}}

    header_pat = re.compile(r"^(.+):\s*$")
    idcount_pat = re.compile(r"^id_count\s*:\s*(\d+)\s*$", flags=re.I)
    id_pat = re.compile(r"^id\s*:\s*(\d+)\s*$", flags=re.I)
    kv_pat = re.compile(r"^(cropping|camera_angle|pose|expression)\s*:\s*(.*)$", flags=re.I)
    inter_pat = re.compile(r"^interactions\s*:\s*(.*)$", flags=re.I)
    props_pat = re.compile(r"^props\s*:\s*(.*)$", flags=re.I)

    for raw in text:
        s = raw.strip()
        if not s:
            continue

        m = header_pat.match(s)
        if m and not s.lower().startswith(("id_count:", "id:", "cropping:", "camera_angle:", "pose:", "expression:", "interactions:", "props:")):
            # new frame block
            flush_record()
            current_id = m.group(1).strip()
            continue

        if current_id is None:
            continue

        m = idcount_pat.match(s)
        if m:
            id_count = int(m.group(1))
            continue

        m = id_pat.match(s)
        if m:
            # starting a new subject block
            if in_subj:
                subjects.append(GTSubject(
                    cropping=subj_tmp["cropping"],
                    camera_angle=subj_tmp["camera_angle"],
                    pose=subj_tmp["pose"],
                    expression=subj_tmp["expression"],
                ))
            in_subj = True
            subj_tmp = {"cropping": {NONE}, "camera_angle": {NONE}, "pose": {NONE}, "expression": {NONE}}
            continue

        m = kv_pat.match(s)
        if m and in_subj:
            key = m.group(1).lower()
            val = m.group(2).strip()
            lower = (key in ("cropping", "camera_angle"))
            subj_tmp[key] = parse_or_none(val, lower=lower)
            continue

        m = inter_pat.match(s)
        if m:
            val = m.group(1).strip()
            if val and val.lower() != "none":
                interactions |= parse_or_none(val, lower=True)
                interactions.discard(NONE)
            continue

        m = props_pat.match(s)
        if m:
            val = m.group(1).strip()
            if val.lower() == "none":
                props = set()
            else:
                props |= split_props_braces(val)
            continue

    # flush last
    flush_record()
    return records

def frame_id_from_pred_filename(name: str) -> str:
    """
    Example:
      C002_G011_all.png.json -> C002_G011
      C019_A_un001_all.png.json -> C019_A_un001
      (GT frame name + '.png.json' also possible)
    """
    base = Path(name).name
    if base.endswith("_all.png.json"):
        return base[:-len("_all.png.json")]
    if base.endswith(".png.json"):
        return base[:-len(".png.json")]
    if base.endswith(".json"):
        return base[:-len(".json")]
    return base

def bbox_center_x(subj: dict) -> float:
    b = subj.get("bbox")
    if isinstance(b, list) and len(b) == 4:
        x1, _, x2, _ = b
        try:
            return float(x1 + x2) / 2.0
        except Exception:
            return 0.0
    return 0.0

def subj_attr_set_from_pred(subj: dict, key: str, *, lower: bool) -> Set[str]:
    """
    Extract attribute set from Qwen JSON subject.
    - cropping/camera_angle: dict {value, confidence} or string
    - pose/expression: list of {description,...} or string/dict
    Missing/empty/None -> {NONE}
    """
    v = subj.get(key, None)
    if v is None:
        return {NONE}

    if isinstance(v, dict):
        # cropping/camera_angle often {"value": "...", "confidence": ...}
        if "value" in v:
            s = str(v["value"]).strip()
        elif "description" in v:
            s = str(v["description"]).strip()
        else:
            s = str(v).strip()
        if not s or s.lower() == "none":
            return {NONE}
        return {s.lower()} if lower else {s}

    if isinstance(v, list):
        out = set()
        for it in v:
            if isinstance(it, dict):
                s = it.get("description") or it.get("value")
                if s is None:
                    continue
                s = str(s).strip()
            else:
                s = str(it).strip()
            if not s or s.lower() == "none":
                continue
            out.add(s.lower() if lower else s)
        return out if out else {NONE}

    # string or other
    s = str(v).strip()
    if not s or s.lower() == "none":
        return {NONE}
    return {s.lower()} if lower else {s}

def load_pred_dir(pred_dir: Path, quiet: bool = False) -> Dict[str, PredRecord]:
    """
    Load Qwen prediction JSONs in pred_dir (recursively).
    - Only reads *_all.png.json or *.png.json
    - Skips *.meta.json
    - If top-level is list, tries to pick a dict element; otherwise skips
    """
    records: Dict[str, PredRecord] = {}

    for p in pred_dir.rglob("*.json"):
        name = p.name

        # skip meta
        if name.endswith(".meta.json"):
            continue

        # ★ Qwen出力だけ読む（混入防止）
        if not (name.endswith("_all.png.json") or name.endswith(".png.json")):
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            if not quiet:
                print(f"[SKIP] {p} (json parse error: {e})")
            continue

        # ★ トップレベルが list の場合の救済
        # ★ トップレベルが list の場合の救済（拡張版）
        if isinstance(data, list):
            # 1) まず「subjectsを持つdict」が入っていればそれを優先（従来互換）
            picked = None
            for el in data:
                if isinstance(el, dict) and ("subjects" in el or "props" in el or "interactions" in el):
                    picked = el
                    break
            if picked is not None:
                data = picked
            else:
                # 2) 「subject dict の配列」か判定して、そうなら {"subjects": data} に包む
                def is_num(x):
                    try:
                        float(x)
                        return True
                    except Exception:
                        return False

                def is_valid_subject_dict(d: dict) -> bool:
                    if not isinstance(d, dict):
                        return False
                    b = d.get("bbox")
                    if not (isinstance(b, list) and len(b) == 4 and all(is_num(v) for v in b)):
                        return False
                    # 何かしら属性がある（bboxだけのゴミを弾く）
                    if any(k in d for k in ("cropping", "camera_angle", "pose", "expression", "id")):
                        return True
                    return False

                subj_like = [el for el in data if is_valid_subject_dict(el)]
                if len(subj_like) == 0:
                    if not quiet:
                        print(f"[SKIP] {p} (top-level list; not dict-with-subjects and not subject-array)")
                    continue

                # 「有効なsubjectだけ」採用
                data = {"subjects": subj_like}


        # ★ それでもdictでなければスキップ
        if not isinstance(data, dict):
            if not quiet:
                print(f"[SKIP] {p} (top-level type={type(data)})")
            continue

        frame_id = frame_id_from_pred_filename(name)

        # subjects
        subjects_raw = data.get("subjects") or []
        pred_subjects: List[PredSubject] = []
        for subj in subjects_raw:
            if not isinstance(subj, dict):
                continue
            cx = bbox_center_x(subj)
            pred_subjects.append(PredSubject(
                cropping=subj_attr_set_from_pred(subj, "cropping", lower=True),
                camera_angle=subj_attr_set_from_pred(subj, "camera_angle", lower=True),
                pose=subj_attr_set_from_pred(subj, "pose", lower=False),
                expression=subj_attr_set_from_pred(subj, "expression", lower=False),
                cx=cx
            ))
        pred_subjects.sort(key=lambda s: s.cx)

        # props set
        props_set: Set[str] = set()
        for pr in (data.get("props") or []):
            if isinstance(pr, dict):
                name2 = pr.get("name")
            else:
                name2 = pr
            if name2 is None:
                continue
            name2 = str(name2).strip()
            if name2 and name2.lower() != "none":
                props_set.add(name2)

        # interactions types only
        inter_set: Set[str] = set()
        for it in (data.get("interactions") or []):
            t = it.get("type") if isinstance(it, dict) else it
            if t is None:
                continue
            t = str(t).strip().lower()
            if t and t != "none":
                inter_set.add(t)

        records[frame_id] = PredRecord(
            frame_id=frame_id,
            subjects=pred_subjects,
            props=props_set,
            interactions=inter_set
        )

    return records


# -------------------------
# Alignment (DP)
# -------------------------

def match_one(gt_set: Set[str], pred_set: Set[str]) -> bool:
    return len(gt_set & pred_set) > 0

def subject_pair_score_for_alignment(gt: GTSubject, pr: PredSubject) -> float:
    """
    Alignment score: only count GT attributes that are NOT None.
    This avoids aligning based on "None" matches.
    Returns score in [0,1].
    """
    attrs = [
        ("cropping", gt.cropping, pr.cropping),
        ("camera_angle", gt.camera_angle, pr.camera_angle),
        ("pose", gt.pose, pr.pose),
        ("expression", gt.expression, pr.expression),
    ]
    denom = 0
    num = 0
    for _, g, p in attrs:
        if g == {NONE}:
            continue
        denom += 1
        if match_one(g, p):
            num += 1
    if denom == 0:
        return 0.0
    return num / denom

def align_monotone_max(gt_subs: List[GTSubject], pr_subs: List[PredSubject]) -> Tuple[float, List[Tuple[int,int]]]:
    """
    Order-preserving alignment maximizing sum of pair scores.
    - If N>=M: match all pred to subset of GT
    - If M>N: match all GT to subset of pred
    Returns: (best_sum_score, pairs[(gt_i, pr_j)] in matched order)
    """
    N = len(gt_subs)
    M = len(pr_subs)
    if N == 0 or M == 0:
        return 0.0, []

    def dp_align(A, B, score_func) -> Tuple[float, List[Tuple[int,int]]]:
        # A is longer/equal side (lenA >= lenB), align all B to subset of A
        LA, LB = len(A), len(B)
        NEG = -1e18
        dp = [[NEG] * (LB + 1) for _ in range(LA + 1)]
        prev = [[None] * (LB + 1) for _ in range(LA + 1)]
        dp[0][0] = 0.0
        for i in range(1, LA + 1):
            dp[i][0] = 0.0
            prev[i][0] = (i-1, 0, "skip")
        for i in range(1, LA + 1):
            for j in range(1, LB + 1):
                best = dp[i-1][j]
                bp = (i-1, j, "skip")
                v = dp[i-1][j-1] + score_func(A[i-1], B[j-1])
                if v > best:
                    best = v
                    bp = (i-1, j-1, "match")
                dp[i][j] = best
                prev[i][j] = bp

        pairs = []
        i, j = LA, LB
        while i > 0 and j > 0:
            pi, pj, act = prev[i][j]
            if act == "match":
                pairs.append((i-1, j-1))
            i, j = pi, pj
        pairs.reverse()
        return dp[LA][LB], pairs

    if N >= M:
        best, pairs = dp_align(gt_subs, pr_subs, subject_pair_score_for_alignment)
        return best, pairs
    else:
        # swap and invert pairs
        best, pairs_sw = dp_align(pr_subs, gt_subs, lambda pr, gt: subject_pair_score_for_alignment(gt, pr))
        pairs = [(gt_i, pr_j) for (pr_j, gt_i) in pairs_sw]
        return best, pairs

# -------------------------
# Metrics
# -------------------------

def prf(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    return p, r, f1

def set_metrics_over_samples(pairs: List[Tuple[Set[str], Set[str]]]) -> Dict[str, float]:
    """
    pairs: list of (GT_set, Pred_set) per sample
    returns micro P/R/F1, avg jaccard, exact match
    """
    TP=FP=FN=0
    jacc_sum=0.0
    exact_sum=0
    n=0
    for g, p in pairs:
        n += 1
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)
        TP += tp; FP += fp; FN += fn
        denom = len(g | p)
        jacc = (tp / denom) if denom else 1.0
        jacc_sum += jacc
        exact_sum += 1 if g == p else 0
    P,R,F1 = prf(TP,FP,FN)
    return {
        "N": n,
        "TP": TP, "FP": FP, "FN": FN,
        "micro_precision": P,
        "micro_recall": R,
        "micro_f1": F1,
        "avg_jaccard": (jacc_sum / n) if n else 0.0,
        "exact_match": (exact_sum / n) if n else 0.0,
    }

def explain_pair(gt_subj, pr_subj):
    items = [
        ("cropping", gt_subj.cropping, pr_subj.cropping),
        ("camera_angle", gt_subj.camera_angle, pr_subj.camera_angle),
        ("pose", gt_subj.pose, pr_subj.pose),
        ("expression", gt_subj.expression, pr_subj.expression),
    ]
    out = []
    for name, g, p in items:
        if g == {NONE}:
            out.append(f"{name}=SKIP(GT None)")
            continue
        ok = len(g & p) > 0
        out.append(f"{name}={'OK' if ok else 'NG'} GT={sorted(g)} PRED={sorted(p)}")
    return " | ".join(out)


# -------------------------
# Main evaluation
# -------------------------

def evaluate(gt: Dict[str, GTRecord], pred: Dict[str, PredRecord], *, verbose: bool = True) -> Dict:
    all_ids = sorted(set(gt.keys()) | set(pred.keys()))
    missing_pred = [k for k in all_ids if k in gt and k not in pred]
    missing_gt   = [k for k in all_ids if k in pred and k not in gt]

    # (A) count metrics
    n_total = 0
    exact = 0
    within1 = 0
    abs_err_sum = 0.0
    per_count = []
    for k in all_ids:
        if k not in gt or k not in pred:
            continue
        g = gt[k].id_count
        m = len(pred[k].subjects)
        n_total += 1
        abs_err = abs(g - m)
        abs_err_sum += abs_err
        if g == m:
            exact += 1
        if abs_err <= 1:
            within1 += 1
        per_count.append((k, g, m, abs_err))

    count_report = {
        "N": n_total,
        "exact_acc": exact / n_total if n_total else 0.0,
        "within1_acc": within1 / n_total if n_total else 0.0,
        "mae": abs_err_sum / n_total if n_total else 0.0,
    }

    # (B) per-subject attributes with DP alignment
    # aggregate per-attribute accuracies over matched pairs
    attrs = ["cropping", "camera_angle", "pose", "expression"]

    corr_inc = {a: 0 for a in attrs}     # include-None
    tot_inc  = {a: 0 for a in attrs}
    corr_exc = {a: 0 for a in attrs}     # exclude GT None
    tot_exc  = {a: 0 for a in attrs}

    # alignment score summaries
    aligned_sum = 0.0
    penalized_sum = 0.0
    k_eval_attr = 0
    unmatched_sum = 0
    per_sample_attr = []  # for debugging / worst cases

    for k in all_ids:
        if k not in gt or k not in pred:
            continue
        gtr = gt[k]
        prr = pred[k]
        if gtr.crowd_only:
            continue
        N = len(gtr.subjects)
        M = len(prr.subjects)
        if N == 0 and M == 0:
            continue

        best_sum, pairs = align_monotone_max(gtr.subjects, prr.subjects)
        K = len(pairs)  # should be min(N,M) unless empty
        if K == 0:
            # no matched pairs
            per_sample_attr.append((k, 0.0, 0.0, N, M, []))
            unmatched_sum += abs(N - M)
            k_eval_attr += 1
            continue

        # normalized scores
        aligned_mean = best_sum / min(N, M) if min(N, M) else 0.0
        penalized_mean = best_sum / max(N, M) if max(N, M) else 0.0

        aligned_sum += aligned_mean
        penalized_sum += penalized_mean
        unmatched_sum += abs(N - M)
        k_eval_attr += 1

        # per attribute correctness over matched pairs
        for (gi, pj) in pairs:
            gs = gtr.subjects[gi]
            ps = prr.subjects[pj]
            # define getters
            for a in attrs:
                gset = getattr(gs, a)
                pset = getattr(ps, a)
                ok = match_one(gset, pset)

                tot_inc[a] += 1
                if ok:
                    corr_inc[a] += 1

                if gset != {NONE}:
                    tot_exc[a] += 1
                    if ok:
                        corr_exc[a] += 1

        # store for worst-case analysis
        per_sample_attr.append((k, aligned_mean, penalized_mean, N, M, pairs))

    attr_report = {
        "N_samples": k_eval_attr,
        "avg_aligned_mean": aligned_sum / k_eval_attr if k_eval_attr else 0.0,     # /min(N,M)
        "avg_penalized_mean": penalized_sum / k_eval_attr if k_eval_attr else 0.0, # /max(N,M)
        "avg_unmatched_count": unmatched_sum / k_eval_attr if k_eval_attr else 0.0,
        "per_attribute_accuracy_include_none": {
            a: (corr_inc[a] / tot_inc[a] if tot_inc[a] else 0.0) for a in attrs
        },
        "per_attribute_accuracy_exclude_none": {
            a: (corr_exc[a] / tot_exc[a] if tot_exc[a] else 0.0) for a in attrs
        },
        "totals_include_none": tot_inc,
        "totals_exclude_none": tot_exc,
    }

    # (C) props/interactions set metrics (per sample)
    props_pairs = []
    inter_pairs = []
    for k in all_ids:
        if k not in gt or k not in pred:
            continue
        gtr = gt[k]
        prr = pred[k]
        props_pairs.append((gtr.props, prr.props))
        inter_pairs.append((gtr.interactions, prr.interactions))

    props_report = set_metrics_over_samples(props_pairs)
    inter_report = set_metrics_over_samples(inter_pairs)

    # Worst samples by aligned_mean (for inspection)
    worst = sorted(per_sample_attr, key=lambda x: x[1])[:10]
    worst_samples = [
        {"id": k, "aligned_mean": am, "penalized_mean": pm, "gtN": N, "predM": M, "pairs": pairs}
        for (k, am, pm, N, M, pairs) in worst
    ]

    attr_prf_report = attribute_prf_over_aligned_pairs(gt, pred, exclude_gt_none=True)

    report = {
        "counts": count_report,
        "attributes_aligned": attr_report,
        "attributes_prf_micro": attr_prf_report,  
        "props": props_report,
        "interactions_type_only": inter_report,
        "missing_pred_for_gt": missing_pred[:20],
        "missing_gt_for_pred": missing_gt[:20],
        "worst10_by_aligned_mean": worst_samples,
        "notes": {
            "alignment": "order-preserving optimal alignment via DP; alignment score ignores GT-None attributes",
            "props": "evaluated as set; beware naming variants unless normalized",
            "interactions": "type-only evaluation (participants ignored)",
        }
    }

    if verbose:
        print("=== Count (id_count) ===")
        print(f"N={count_report['N']}  exact={count_report['exact_acc']:.3f}  within1={count_report['within1_acc']:.3f}  MAE={count_report['mae']:.3f}")
        print("")
        print("=== Per-subject attributes (DP aligned) ===")
        print(f"N_samples={attr_report['N_samples']}")
        print(f"avg aligned_mean (/min)={attr_report['avg_aligned_mean']:.3f}  avg penalized_mean (/max)={attr_report['avg_penalized_mean']:.3f}  avg |N-M|={attr_report['avg_unmatched_count']:.3f}")
        print("accuracy include None:")
        for a,v in attr_report["per_attribute_accuracy_include_none"].items():
            print(f"  {a:12s}: {v:.3f}  (n={attr_report['totals_include_none'][a]})")
        print("accuracy exclude GT None:")
        for a,v in attr_report["per_attribute_accuracy_exclude_none"].items():
            print(f"  {a:12s}: {v:.3f}  (n={attr_report['totals_exclude_none'][a]})")
        print("")
        print("=== Per-subject attributes (DP aligned, multi-label micro P/R/F1) ===")
        for a, v in report["attributes_prf_micro"].items():
            print(f"  {a:12s}: P={v['micro_P']:.3f} R={v['micro_R']:.3f} F1={v['micro_F1']:.3f} "
                  f"(TP={v['TP']}, FP={v['FP']}, FN={v['FN']}, pairs={v['n_pairs']}, skip={v['skipped_pairs']})")
        print("")



        print("=== Per-subject attributes (hit-based P/R/F1; denom=sets) ===")
        rep_inc = attribute_hit_pr_over_aligned_pairs(gt, pred, exclude_gt_none=False)
        for a, v in rep_inc.items():
            print(f"  {a:12s}: P={v['micro_P']:.3f} R={v['micro_R']:.3f} F1={v['micro_F1']:.3f} "
                  f"(hit={v['hit']}, pred_sets={v['denom_pred_sets']}, gt_sets={v['denom_gt_sets']})")
        print("")
        
        print("=== Per-subject attributes (hit-based P/R/F1; exclude GT None) ===")
        rep_exc = attribute_hit_pr_over_aligned_pairs(gt, pred, exclude_gt_none=True)
        for a, v in rep_exc.items():
            print(f"  {a:12s}: P={v['micro_P']:.3f} R={v['micro_R']:.3f} F1={v['micro_F1']:.3f} "
                  f"(hit={v['hit']}, pred_sets={v['denom_pred_sets']}, gt_sets={v['denom_gt_sets']}, skip={v['skip_gt_none_pairs']})")
        print("")






        print("=== Props (set) ===")
        print(f"N={props_report['N']}  microF1={props_report['micro_f1']:.3f}  P={props_report['micro_precision']:.3f}  R={props_report['micro_recall']:.3f}  avgJ={props_report['avg_jaccard']:.3f}  exact={props_report['exact_match']:.3f}")
        print("")
        print("=== Interactions type-only (set) ===")
        print(f"N={inter_report['N']}  microF1={inter_report['micro_f1']:.3f}  P={inter_report['micro_precision']:.3f}  R={inter_report['micro_recall']:.3f}  avgJ={inter_report['avg_jaccard']:.3f}  exact={inter_report['exact_match']:.3f}")
        print("")
        print("Worst 10 samples by aligned_mean:")
        for w in worst_samples:
            print(f"  {w['id']}: aligned={w['aligned_mean']:.3f} penalized={w['penalized_mean']:.3f} GT={w['gtN']} Pred={w['predM']} pairs={w['pairs']}")
            # 追加：最初のペアを詳細表示
            k = w["id"]
            gtr = gt[k]
            prr = pred[k]
            if w["pairs"]:
                gi, pj = w["pairs"][0]
                print("      ", explain_pair(gtr.subjects[gi], prr.subjects[pj]))
        

    return report


# --- add: multilabel PRF for subject attributes -----------------------------

def prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1

def attribute_prf_over_aligned_pairs(
    gt: Dict[str, GTRecord],
    pred: Dict[str, PredRecord],
    *,
    exclude_gt_none: bool = True
) -> Dict:
    """
    DPで対応付けたペア(gi,pj)だけを対象に、属性ごとに multi-label の micro P/R/F1 を算出する。
    - GTの {__NONE__} はデフォルトで除外（評価対象外）
    - Pred の {__NONE__} は空集合として扱う（=何も予測していない）
    """
    attrs = ["cropping", "camera_angle", "pose", "expression"]

    agg = {a: {"TP": 0, "FP": 0, "FN": 0, "pairs": 0, "skipped_pairs": 0} for a in attrs}

    all_ids = sorted(set(gt.keys()) | set(pred.keys()))
    for k in all_ids:
        if k not in gt or k not in pred:
            continue
        gtr = gt[k]
        prr = pred[k]
        if gtr.crowd_only:
            continue

        N = len(gtr.subjects)
        M = len(prr.subjects)
        if N == 0 or M == 0:
            continue

        _, pairs = align_monotone_max(gtr.subjects, prr.subjects)
        if not pairs:
            continue

        for (gi, pj) in pairs:
            gs = gtr.subjects[gi]
            ps = prr.subjects[pj]
            for a in attrs:
                G = set(getattr(gs, a))
                P = set(getattr(ps, a))

                if exclude_gt_none and G == {NONE}:
                    agg[a]["skipped_pairs"] += 1
                    continue

                # NONEはラベルではなく「未記入」扱いにして落とす
                G = {x for x in G if x != NONE}
                P = {x for x in P if x != NONE}

                tp = len(G & P)
                fp = len(P - G)
                fn = len(G - P)

                agg[a]["TP"] += tp
                agg[a]["FP"] += fp
                agg[a]["FN"] += fn
                agg[a]["pairs"] += 1

    report = {}
    for a in attrs:
        tp, fp, fn = agg[a]["TP"], agg[a]["FP"], agg[a]["FN"]
        p, r, f1 = prf1(tp, fp, fn)
        report[a] = {
            "micro_P": p,
            "micro_R": r,
            "micro_F1": f1,
            "TP": tp, "FP": fp, "FN": fn,
            "n_pairs": agg[a]["pairs"],
            "skipped_pairs": agg[a]["skipped_pairs"],
        }
    return report


def prf_from_hits(hit: int, denom_p: int, denom_r: int):
    P = hit / denom_p if denom_p else 0.0
    R = hit / denom_r if denom_r else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) else 0.0
    return P, R, F1

def attribute_hit_pr_over_aligned_pairs(
    gt: dict,
    pred: dict,
    *,
    exclude_gt_none: bool = False
) -> dict:
    """
    1人=1ラベルセットとして、
      hit = (allowed_set ∩ pred_set != ∅) の人数
    を分子にして
      Precision denom = 予測ラベルセット数（=予測人数）
      Recall denom    = 正解ラベルセット数（=GT人数）
    で割る。

    exclude_gt_none=True のとき：
      - GTが {NONE} の人物は Recall 分母から除外
      - その人物に対応した Pred も Precision 分母から除外（未知なので評価しない）
      - ただし「余分に出した予測人物（unpaired）」は Precision 分母に残してペナルティにする
    """
    attrs = ["cropping", "camera_angle", "pose", "expression"]
    out = {a: {"hit": 0, "denom_pred": 0, "denom_gt": 0, "pairs": 0, "skip_gt_none": 0} for a in attrs}

    all_ids = sorted(set(gt.keys()) | set(pred.keys()))
    for fid in all_ids:
        if fid not in gt or fid not in pred:
            continue
        gtr = gt[fid]
        prr = pred[fid]
        if getattr(gtr, "crowd_only", False):
            continue

        Gsubs = gtr.subjects
        Psubs = prr.subjects
        N = len(Gsubs)
        M = len(Psubs)

        # DP対応
        if N > 0 and M > 0:
            _, pairs = align_monotone_max(Gsubs, Psubs)
        else:
            pairs = []

        paired_pj = {pj for (_, pj) in pairs}
        unpaired_count = M - len(paired_pj)  # 余分に出した予測人物数（M>Nのとき等）

        # 属性ごとに hit と分母を加算
        for a in attrs:
            # Recall分母（GT人数）
            if exclude_gt_none:
                gt_evaluable = [i for i in range(N) if set(getattr(Gsubs[i], a)) != {NONE}]
                denom_gt = len(gt_evaluable)
            else:
                denom_gt = N

            # Precision分母（予測人数）
            if exclude_gt_none:
                # GTがNoneでない人物に対応したpredだけ数える + unpaired は常に数える
                evaluable_paired_pj = set()
                for gi, pj in pairs:
                    Gset = set(getattr(Gsubs[gi], a))
                    if Gset != {NONE}:
                        evaluable_paired_pj.add(pj)
                    else:
                        out[a]["skip_gt_none"] += 1
                denom_pred = len(evaluable_paired_pj) + unpaired_count
            else:
                denom_pred = M

            # hit（対応ペアの中で a∩p≠∅ の数）
            hit = 0
            for gi, pj in pairs:
                Gset = set(getattr(Gsubs[gi], a))
                Pset = set(getattr(Psubs[pj], a))

                if exclude_gt_none and Gset == {NONE}:
                    continue  # 評価対象外

                # include-none のときは NONE も集合に残す（NONE同士ならhit）
                # exclude-none のときは NONE を落としてから評価（空ならhitしない）
                if exclude_gt_none:
                    Gset = {x for x in Gset if x != NONE}
                    Pset = {x for x in Pset if x != NONE}

                if (Gset & Pset):
                    hit += 1

            out[a]["hit"] += hit
            out[a]["denom_pred"] += denom_pred
            out[a]["denom_gt"] += denom_gt
            out[a]["pairs"] += len(pairs)

    # micro P/R/F1 を作る
    report = {}
    for a in attrs:
        hit = out[a]["hit"]
        dp = out[a]["denom_pred"]
        dg = out[a]["denom_gt"]
        P, R, F1 = prf_from_hits(hit, dp, dg)
        report[a] = {
            "micro_P": float(P),
            "micro_R": float(R),
            "micro_F1": float(F1),
            "hit": int(hit),
            "denom_pred_sets": int(dp),
            "denom_gt_sets": int(dg),
            "pairs": int(out[a]["pairs"]),
            "skip_gt_none_pairs": int(out[a]["skip_gt_none"]),
        }
    return report



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True, help="correct_qwenvl.txt")
    ap.add_argument("--pred_dir", type=Path, required=True, help="directory containing *_all.png.json")
    ap.add_argument("--out_json", type=Path, default=None, help="optional: write report json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    gt = parse_gt_file(args.gt)
    pred = load_pred_dir(args.pred_dir)
    report = evaluate(gt, pred, verbose=(not args.quiet))

    if args.out_json:
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
