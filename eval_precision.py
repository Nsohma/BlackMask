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
