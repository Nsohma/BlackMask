import re
import argparse
from pathlib import Path
from collections import defaultdict

def parse_correct(path: Path) -> dict[str, set[str]]:
    """
    correct_label.txt:
    C002_G011:
    face
    eyes
    ...
    """
    gt: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(":"):
            current = s[:-1].strip()
            gt[current] = []
        else:
            if current is None:
                continue
            gt[current].append(s.lower())
    return {k: set(lbl.strip() for lbl in v if lbl.strip()) for k, v in gt.items()}

def parse_log(path: Path) -> dict[str, set[str]]:
    """
    log_overlap.txt:
    [OK] C002_G011  selected=6  -> ...
       01   nose judge=...
       02  mouth judge=...
    """
    pred: dict[str, list[str]] = {}
    current = None

    ok_pat = re.compile(r"^\[OK\]\s+(\S+)\s+selected=")
    line_pat = re.compile(r"^\s*\d+\s+([A-Za-z_]+)\s+judge=")

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ok_pat.match(line)
        if m:
            current = m.group(1)
            pred[current] = []
            continue
        if current is None:
            continue
        m2 = line_pat.match(line)
        if m2:
            pred[current].append(m2.group(1).lower())

    return {k: set(v) for k, v in pred.items()}

def f1_pr(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) else 0.0
    return p, r, f1

def main(correct_txt: Path, log_txt: Path):
    gt = parse_correct(correct_txt)
    pred = parse_log(log_txt)

    keys = sorted(set(gt.keys()) | set(pred.keys()))
    # 欠けているIDがあれば空集合として扱う
    for k in keys:
        gt.setdefault(k, set())
        pred.setdefault(k, set())

    # micro
    TP = FP = FN = 0
    per_sample = {}
    for k in keys:
        g = gt[k]
        p = pred[k]
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)
        TP += tp; FP += fp; FN += fn

        prec, rec, f1 = f1_pr(tp, fp, fn)
        jacc = tp / len(g | p) if (g | p) else 1.0
        exact = (g == p)
        per_sample[k] = (f1, jacc, exact, g, p, tp, fp, fn)

    micro_p, micro_r, micro_f1 = f1_pr(TP, FP, FN)
    exact_acc = sum(1 for k in keys if per_sample[k][2]) / len(keys)
    avg_f1 = sum(per_sample[k][0] for k in keys) / len(keys)
    avg_jacc = sum(per_sample[k][1] for k in keys) / len(keys)

    labels = sorted(set().union(*gt.values(), *pred.values()))

    print(f"N={len(keys)}")
    print(f"Micro  P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}  (TP={TP}, FP={FP}, FN={FN})")
    print(f"Macro(avg-per-sample) F1={avg_f1:.3f}  AvgJaccard={avg_jacc:.3f}  ExactMatch={exact_acc:.3f}")
    print("")

    print("Per-label (P/R/F1, support):")
    for lab in labels:
        tp=fp=fn=0
        for k in keys:
            g = lab in gt[k]
            p = lab in pred[k]
            if g and p: tp += 1
            elif (not g) and p: fp += 1
            elif g and (not p): fn += 1
        p_, r_, f1_ = f1_pr(tp, fp, fn)
        support = sum(1 for k in keys if lab in gt[k])
        print(f"  {lab:>6}  P={p_:.3f} R={r_:.3f} F1={f1_:.3f}  support={support}")

    print("\nWorst 10 samples by F1:")
    worst = sorted(per_sample.items(), key=lambda kv: kv[1][0])[:10]
    for k, (f1, jacc, exact, g, p, tp, fp, fn) in worst:
        print(f"  {k}: F1={f1:.3f} Jacc={jacc:.3f}  GT={sorted(g)}  PRED={sorted(p)}  (tp={tp}, fp={fp}, fn={fn})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("correct_label", type=Path)
    ap.add_argument("log_overlap", type=Path)
    args = ap.parse_args()
    main(args.correct_label, args.log_overlap)
