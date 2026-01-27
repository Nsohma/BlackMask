import cv2
import numpy as np
import json
from pathlib import Path

def make_mask_bg_diff_only(img_bgr: np.ndarray, diff_thresh: float = 30.0):
    img16 = img_bgr.astype(np.int16)
    bg = np.median(img16.reshape(-1, 3), axis=0)  # BGR
    diff = np.linalg.norm(img16 - bg, axis=2)
    mask = diff > diff_thresh
    return mask, bg

def mask_to_rle_segments(mask: np.ndarray):
    h, w = mask.shape
    segs = []
    for y in range(h):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(xs) > 1)
        starts = np.r_[xs[0], xs[cuts + 1]]
        ends   = np.r_[xs[cuts], xs[-1]]
        for s, e in zip(starts, ends):
            segs.append([int(y), int(s), int(e)])
    return segs

def extract_positions_methodC(
    in_path: str,
    out_json: str,
    out_mask_png: str | None = None,
    diff_thresh: float = 30.0,
    open_ksize: int = 0
):
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(in_path)

    mask, bg = make_mask_bg_diff_only(img, diff_thresh=diff_thresh)

    if open_ksize and open_ksize > 1:
        m = (mask.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        mask = (m > 0)

    if out_mask_png is not None:
        cv2.imwrite(out_mask_png, (mask.astype(np.uint8) * 255))

    segs = mask_to_rle_segments(mask)
    data = {
        "image": str(Path(in_path).resolve()),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "bg_bgr_median": [int(bg[0]), int(bg[1]), int(bg[2])],
        "thresholds": {"diff_thresh": float(diff_thresh), "open_ksize": int(open_ksize)},
        "format": "rle_segments_y_xstart_xend_inclusive",
        "segments": segs,
        "num_segments": int(len(segs)),
        "num_pixels": int(mask.sum())
    }

    Path(out_json).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    in_dir = base_dir / "05_raf_image"
    out_json_dir = base_dir / "05_raf_image_json"
    out_mask_dir = base_dir / "05_raf_image_mask"   # マスク不要なら使わない

    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    diff_thresh = 30.0
    open_ksize = 0
    save_masks = True  # マスクPNGも出すなら True、不要なら False

    # 05_raf_image 内の全画像を処理
    for img_path in sorted(in_dir.iterdir()):
        if not img_path.is_file() or not is_image_file(img_path):
            continue

        stem = img_path.stem  # 拡張子なし
        out_json = out_json_dir / f"{stem}.json"
        out_mask = (out_mask_dir / f"{stem}_mask.png") if save_masks else None

        try:
            extract_positions_methodC(
                in_path=str(img_path),
                out_json=str(out_json),
                out_mask_png=str(out_mask) if out_mask is not None else None,
                diff_thresh=diff_thresh,
                open_ksize=open_ksize
            )
            print(f"[OK] {img_path.name} -> {out_json.name}")
        except Exception as e:
            print(f"[NG] {img_path.name} : {e}")
