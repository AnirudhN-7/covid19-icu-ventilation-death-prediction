"""Extract radiomics-style features from the earliest original chest X-ray of each patient.

Why not pyradiomics? It often fails to install on recent Python versions, so this module
uses scikit-image / SciPy to compute the same kinds of features (first-order intensity
statistics, GLCM texture, gradient, left-right asymmetry) inside a region of interest.

Usage:
    python -m src.extract_radiomics --metadata path/to/metadata.csv --images path/to/image_root \
        --out data/radiomics.csv [--mask central|full] [--use-enhanced]
`--images` is the folder that contains the patient folders (A000801/..., A000802/...).
Then:  python -m src.prepare_nysbu --details patient_details.csv --radiomics data/radiomics.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats, ndimage
from skimage.feature import graycomatrix, graycoprops
from .config import DATA_DIR, ID_COL

LEVELS = 32  # grey-level quantisation for texture matrices


def make_roi(img: np.ndarray, mode: str) -> np.ndarray:
    """Boolean mask. 'central' = middle box that mostly covers both lungs (no segmentation
    model needed; it also contains tubes/leads, which is a known limitation).
    'full' = every non-black pixel."""
    h, w = img.shape
    if mode == "full":
        return img > 0
    roi = np.zeros_like(img, dtype=bool)
    roi[int(0.10 * h):int(0.80 * h), int(0.15 * w):int(0.85 * w)] = True
    return roi


def _first_order(x: np.ndarray, prefix: str) -> dict:
    p = np.percentile(x, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(x, bins=LEVELS, range=(0, 255))
    pr = hist[hist > 0] / hist.sum()
    return {f"{prefix}mean": x.mean(), f"{prefix}std": x.std(), f"{prefix}skew": stats.skew(x),
            f"{prefix}kurtosis": stats.kurtosis(x), f"{prefix}entropy": -(pr * np.log2(pr)).sum(),
            f"{prefix}p10": p[0], f"{prefix}p25": p[1], f"{prefix}median": p[2], f"{prefix}p75": p[3],
            f"{prefix}p90": p[4], f"{prefix}iqr": p[3] - p[1], f"{prefix}energy": (pr ** 2).sum()}


def _glcm(patch: np.ndarray, prefix: str) -> dict:
    q = np.clip((patch.astype(float) / 256 * LEVELS).astype(np.uint8), 0, LEVELS - 1)
    m = graycomatrix(q, distances=[1, 3, 5], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=LEVELS, symmetric=True, normed=True)
    return {f"{prefix}{prop}": graycoprops(m, prop).mean()
            for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]}


def extract_features(path, mask_mode="central") -> dict:
    img = np.array(Image.open(path).convert("L"))
    roi = make_roi(img, mask_mode)
    ys, xs = np.where(roi)
    box = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    feats = _first_order(img[roi].astype(float), "rad_fo_")
    feats.update(_glcm(box, "rad_glcm_"))
    # gradient / edge density inside ROI (tubes, lines, consolidations all raise it)
    gx, gy = ndimage.sobel(img.astype(float), axis=1), ndimage.sobel(img.astype(float), axis=0)
    grad = np.hypot(gx, gy)[roi]
    feats.update({"rad_grad_mean": grad.mean(), "rad_grad_std": grad.std(),
                  "rad_edge_density": (grad > np.percentile(grad, 90)).mean()})
    # left/right (image halves) intensity asymmetry: unilateral vs bilateral disease
    mid = box.shape[1] // 2
    l, r = box[:, :mid].astype(float), box[:, mid:].astype(float)
    feats.update({"rad_lr_mean_diff": l.mean() - r.mean(), "rad_lr_std_diff": l.std() - r.std()})
    # upper / middle / lower thirds (zone-wise brightness, common in CXR severity scoring)
    for name, part in zip(["upper", "mid", "lower"], np.array_split(box, 3, axis=0)):
        feats[f"rad_zone_{name}_mean"] = part.mean()
        feats[f"rad_zone_{name}_std"] = part.std()
    return feats


def pick_images(metadata_csv, use_enhanced=False) -> pd.DataFrame:
    """One image per patient: earliest dated series, original (non-enhanced) copy, first frame."""
    m = pd.read_csv(metadata_csv)
    m = m[m["enhanced"] == use_enhanced].copy()
    m["series_date"] = pd.to_datetime(m["series"].str.extract(r"^(\d\d-\d\d-\d{4})")[0],
                                      format="%m-%d-%Y", errors="coerce")
    m = m.sort_values(["patient_id", "series_date", "filename"])
    return m.groupby("patient_id", as_index=False).first()[["patient_id", "series_date", "filename"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images", required=True, help="root folder containing the patient folders")
    ap.add_argument("--out", default=str(DATA_DIR / "radiomics.csv"))
    ap.add_argument("--mask", default="central", choices=["central", "full"])
    ap.add_argument("--use-enhanced", action="store_true", help="use the contrast-enhanced copies")
    a = ap.parse_args()

    sel = pick_images(a.metadata, a.use_enhanced)
    root, rows, missing = Path(a.images), [], 0
    for i, r in enumerate(sel.itertuples(index=False), 1):
        path = root / r.filename
        if not path.exists():
            missing += 1
            continue
        try:
            rows.append({ID_COL: r.patient_id, **extract_features(path, a.mask)})
        except Exception as e:
            print(f"  failed {r.patient_id}: {e}")
        if i % 100 == 0:
            print(f"  {i}/{len(sel)} patients processed")
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Saved {a.out}: {len(out)} patients x {out.shape[1] - 1} features "
          f"({missing} images not found under {root})")


if __name__ == "__main__":
    main()
