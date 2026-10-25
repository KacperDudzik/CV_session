"""Optional classical descriptors from Session 4.1 (Canny, Harris, LBP, HOG, SIFT).

These are labeled optional extensions. Failures degrade to zeros so the
pipeline still runs when OpenCV/skimage extras are incomplete.
"""

import numpy as np

OPTIONAL_PREFIXES = ("opt_canny", "opt_harris", "opt_lbp", "opt_hog", "opt_sift")


def optional_descriptor_features(gray_u8):
    """Compact numeric summaries. `gray_u8` is uint8 HxW."""
    values = {}
    values.update(_canny_features(gray_u8))
    values.update(_harris_features(gray_u8))
    values.update(_lbp_features(gray_u8))
    values.update(_hog_features(gray_u8))
    values.update(_sift_features(gray_u8))
    return values


def optional_feature_names():
    toy = np.zeros((64, 64), dtype=np.uint8)
    return sorted(optional_descriptor_features(toy).keys())


def _canny_features(gray):
    try:
        import cv2

        edges = cv2.Canny(gray, 50, 150)
        return {
            "opt_canny_density": float((edges > 0).mean()),
            "opt_canny_mean": float(edges.mean()),
        }
    except Exception:
        return {"opt_canny_density": 0.0, "opt_canny_mean": 0.0}


def _harris_features(gray):
    try:
        import cv2

        response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        thr = 0.01 * float(response.max()) if response.size else 0.0
        return {
            "opt_harris_mean": float(response.mean()),
            "opt_harris_max": float(response.max()) if response.size else 0.0,
            "opt_harris_frac": float((response > thr).mean()) if thr > 0 else 0.0,
        }
    except Exception:
        return {"opt_harris_mean": 0.0, "opt_harris_max": 0.0, "opt_harris_frac": 0.0}


def _lbp_features(gray):
    try:
        from skimage.feature import local_binary_pattern

        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        n_bins = 10
        counts, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        shares = counts / max(lbp.size, 1)
        values = {f"opt_lbp_bin{i}": float(shares[i - 1]) for i in range(1, n_bins + 1)}
        values["opt_lbp_mean"] = float(lbp.mean())
        values["opt_lbp_std"] = float(lbp.std())
        return values
    except Exception:
        values = {f"opt_lbp_bin{i}": 0.0 for i in range(1, 11)}
        values.update({"opt_lbp_mean": 0.0, "opt_lbp_std": 0.0})
        return values


def _hog_features(gray):
    try:
        from skimage.feature import hog

        vec = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        return {
            "opt_hog_mean": float(vec.mean()) if vec.size else 0.0,
            "opt_hog_std": float(vec.std()) if vec.size else 0.0,
            "opt_hog_energy": float(np.square(vec).mean()) if vec.size else 0.0,
        }
    except Exception:
        return {"opt_hog_mean": 0.0, "opt_hog_std": 0.0, "opt_hog_energy": 0.0}


def _sift_features(gray):
    try:
        import cv2

        sift = cv2.SIFT_create()
        keypoints, desc = sift.detectAndCompute(gray, None)
        n = 0 if keypoints is None else len(keypoints)
        if desc is None or len(desc) == 0:
            return {
                "opt_sift_n_keypoints": float(n),
                "opt_sift_desc_mean": 0.0,
                "opt_sift_desc_std": 0.0,
            }
        return {
            "opt_sift_n_keypoints": float(n),
            "opt_sift_desc_mean": float(desc.mean()),
            "opt_sift_desc_std": float(desc.std()),
        }
    except Exception:
        return {
            "opt_sift_n_keypoints": 0.0,
            "opt_sift_desc_mean": 0.0,
            "opt_sift_desc_std": 0.0,
        }
