#prompt "face . eyes . mouth . arm . head . leg . nose . person ."
import json
import math
from pathlib import Path
from collections import defaultdict
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def bbox_float_to_int_xyxy_inclusive(b, W, H):
    x1, y1, x2, y2 = b
    xi1 = int(math.floor(x1))
    yi1 = int(math.floor(y1))
    xi2 = int(math.ceil(x2)) - 1
    yi2 = int(math.ceil(y2)) - 1

    xi1 = max(0, min(W - 1, xi1))
    yi1 = max(0, min(H - 1, yi1))
    xi2 = max(0, min(W - 1, xi2))
    yi2 = max(0, min(H - 1, yi2))

    if xi2 < xi1 or yi2 < yi1:
        return None
    return (xi1, yi1, xi2, yi2)

def build_row_index(segments):
    rows = defaultdict(list)
    for y, xs, xe in segments:
        rows[int(y)].append((int(xs), int(xe)))
    return rows

def count_overlap_pixels_in_bbox(rows, bbox_xyxy_inc):
    x1, y1, x2, y2 = bbox_xyxy_inc
    total = 0
    for y in range(y1, y2 + 1):
        for xs, xe in rows.get(y, []):
            l = max(xs, x1)
            r = min(xe, x2)
            if l <= r:
                total += (r - l + 1)
    return total

def compute_for_pair(lines_json_path: str, bbox_json_path: str,
                     min_pixels: int = 10, min_ratio: float = 0.0005,
                     judge_thresh: float = 0.02, top_k: int | None = None):
    """
    1ペア分の計算結果(dict)を返す（ファイル書き出しは呼び出し側）
    """
    lines = load_json(lines_json_path)
    dets  = load_json(bbox_json_path)

    H = int(lines["height"])
    W = int(lines["width"])
    rows = build_row_index(lines["segments"])

    scored = []
    for idx, d in enumerate(dets.get("detections", [])):
        bbox_inc = bbox_float_to_int_xyxy_inclusive(d["bbox_xyxy"], W, H)

        ink_pixels = 0
        area = 0
        ink_ratio = 0.0
        has_ink = False
        bbox_int = None

        if bbox_inc is not None:
            x1, y1, x2, y2 = bbox_inc
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            ink_pixels = count_overlap_pixels_in_bbox(rows, bbox_inc)
            ink_ratio = (ink_pixels / area) if area > 0 else 0.0
            has_ink = (ink_pixels >= min_pixels) and (ink_ratio >= min_ratio)
            bbox_int = [x1, y1, x2, y2]

        score = float(d.get("score", 0.0))
        judge_score = (ink_ratio * score) if has_ink else 0.0

        scored.append({
            "det_index": idx,
            "label": d.get("label"),
            "score": score,
            "bbox_xyxy": d.get("bbox_xyxy"),
            "bbox_xyxy_int_inclusive": bbox_int,
            "ink_pixels": int(ink_pixels),
            "bbox_area": int(area),
            "ink_ratio": float(ink_ratio),
            "has_ink": bool(has_ink),
            "judge_score": float(judge_score),
        })

    scored.sort(key=lambda x: x["judge_score"], reverse=True)

    selected = [d for d in scored if d["judge_score"] >= judge_thresh]
    if top_k is not None:
        selected = selected[:top_k]

    selected_labels = [d["label"] for d in selected]

    result = {
        "image": dets.get("image"),
        "lines_json": str(Path(lines_json_path).resolve()),
        "bbox_json": str(Path(bbox_json_path).resolve()),
        "criteria": {
            "min_pixels": int(min_pixels),
            "min_ratio": float(min_ratio),
            "judge_thresh": float(judge_thresh)
        },
        "selected_labels": selected_labels,      # 重複あり
        "selected_detections": selected,
        "all_scored_detections": scored
    }
    return result

def pair_name_from_raf(stem: str) -> str:
    """
    "C002_G011_raf" -> "C002_G011"
    """
    if stem.endswith("_raf"):
        return stem[:-4]
    return stem

def batch_run():
    base_dir = Path(__file__).resolve().parent

    lines_dir = base_dir / "05_raf_image_json"   # *_raf.json がある
    bbox_dir  = base_dir / "03_sub_cell"         # *_sub.json がある

    out_dir   = base_dir / "06_overlap_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # パラメータ
    min_pixels   = 10
    min_ratio    = 0.0005
    judge_thresh = 0.02
    top_k        = None

    raf_files = sorted(lines_dir.glob("*.json"))

    ok = 0
    ng = 0

    for raf_path in raf_files:
        stem = raf_path.stem                       # C002_G011_raf
        key = pair_name_from_raf(stem)             # C002_G011
        bbox_name = f"{key}_sub.json"              # C002_G011_sub.json
        bbox_path = bbox_dir / bbox_name

        if not bbox_path.exists():
            print(f"[SKIP] bbox not found: {bbox_name} (for {raf_path.name})")
            ng += 1
            continue

        out_path = out_dir / f"{key}_overlap.json" # 出力名

        try:
            result = compute_for_pair(
                str(raf_path), str(bbox_path),
                min_pixels=min_pixels,
                min_ratio=min_ratio,
                judge_thresh=judge_thresh,
                top_k=top_k
            )
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            # 標準出力（上位だけ見せる）
            top_show = result["selected_detections"][:]
            print(f"[OK] {key}  selected={len(result['selected_detections'])}  -> {out_path.name}")
            for i, d in enumerate(top_show, 1):
                print(f"   {i:02d} {d['label']:>6} judge={d['judge_score']:.6f} score={d['score']:.3f} ratio={d['ink_ratio']:.6f} px={d['ink_pixels']}")
            ok += 1

        except Exception as e:
            print(f"[NG] {key}: {e}")
            ng += 1

    print(f"\nDONE: ok={ok}, ng={ng}, total={ok+ng}")

if __name__ == "__main__":
    batch_run()
