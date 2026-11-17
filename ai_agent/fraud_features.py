"""
Feature extraction for the ID fraud classifier.

Copied from section 5 of the Telecom 2B notebook (Telecom_2B_1.ipynb) so the agent
computes exactly the same 295 features the model was trained on.
If the team changes extract_features() in the notebook, copy the new version here too.
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = (512, 320)    # (width, height) for resized features, keeps the ~1.6 card aspect ratio
GRID = 8                 # 8x8 grid for local features
FINE = 16                # 16x16 grid so small edits (like text) aren't diluted


# --- helpers ---

def grid(arr, n=GRID):
    # split image into an n x n grid of cells
    h, w = arr.shape[:2]
    return [arr[i*h//n:(i+1)*h//n, j*w//n:(j+1)*w//n] for i in range(n) for j in range(n)]

def anomaly_stats(vals):
    # how much the most unusual cells stand out from the rest of the card.
    # doesn't depend on where the edit is
    v = np.asarray(vals, dtype=np.float32)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    z = np.abs(v - med) / mad
    return [v.max(), v.min(), v.std(), v.max() - v.min(), z.max(), np.sort(z)[-3:].mean()]

def chi2_to_mean(hists):
    # how different each cell's histogram is from the card's average histogram
    mean_hist = hists.mean(axis=0)
    return ((hists - mean_hist) ** 2 / (hists + mean_hist + 1e-6)).sum(axis=1)


# --- noise profile (targets gaussian vs poisson noise) ---

def noise_vs_brightness(gray, residual):
    # gaussian noise is the same size everywhere; poisson noise grows with brightness.
    # measured only in flat areas so edges aren't counted as noise
    smooth = cv2.GaussianBlur(gray, (7, 7), 0).astype(np.float32)
    grad = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    flat = grad < np.percentile(grad, 50)

    def band_std(lo, hi):
        m = flat & (smooth >= lo) & (smooth < hi)
        return residual[m].std() if m.sum() > 500 else 0.0

    dark, mid, bright = band_std(0, 85), band_std(85, 170), band_std(170, 256)
    return [dark, mid, bright, bright / (dark + 1e-6)]


# --- jpeg grid (targets copy-move, splice, text) ---
# jpeg works on 8x8 blocks, leaving slightly stronger jumps every 8 columns/rows.
# pasted patches were saved at low jpeg quality, so they carry their own stronger grid,
# usually shifted relative to the rest of the card.

def _best_offset(profile, start):
    # profile[k] = average jump at image coordinate start + k.
    # score each of the 8 possible grid positions, return best score and its position
    scores = [profile[(o - start) % 8::8].mean() for o in range(8)]
    o = int(np.argmax(scores))
    return scores[o] / (profile.mean() + 1e-6), o

def jpeg_grid_cells(gray, n=FINE):
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))   # jump between column x and x+1
    dy = np.abs(np.diff(g, axis=0))   # jump between row y and y+1
    h, w = g.shape
    strength, offset = [], []
    for i in range(n):
        y0, y1 = i*h//n, min((i+1)*h//n, h - 1)
        for j in range(n):
            x0, x1 = j*w//n, min((j+1)*w//n, w - 1)
            sc, oc = _best_offset(dx[y0:y1, x0:x1].mean(axis=0), x0)
            sr, orow = _best_offset(dy[y0:y1, x0:x1].mean(axis=1), y0)
            strength.append(sc + sr)
            offset.append(oc * 8 + orow)
    offset = np.array(offset)
    # fraction of cells whose grid doesn't line up with the card's main grid
    mismatch = np.mean(offset != np.bincount(offset).argmax())
    return strength, mismatch


# --- copy-move (targets copy-move) ---

def copy_move_score(gray):
    # find keypoints that look almost the same but sit in different spots on the card.
    # a copied region creates a lot of these
    f = 1000 / gray.shape[1]
    g = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=1500)
    kps, des = orb.detectAndCompute(g, None)
    if des is None or len(kps) < 3:
        return [0, 0]

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3)
    count = 0
    for m in matches:
        others = [x for x in m if x.trainIdx != x.queryIdx]  # skip matching a point to itself
        if len(others) < 2:
            continue
        a, b = others[0], others[1]
        if a.distance < 40 and a.distance < 0.7 * b.distance:
            p1, p2 = kps[a.queryIdx].pt, kps[a.trainIdx].pt
            if np.hypot(p1[0] - p2[0], p1[1] - p2[1]) > 40:  # ignore neighbors
                count += 1
    return [count, count / len(kps)]


# --- main extractor ---

def extract_features(path):
    img = cv2.imread(path)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # full resolution maps (resizing would smooth these out)
    residual = gray_full.astype(np.float32) - cv2.medianBlur(gray_full, 3).astype(np.float32)

    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ela = cv2.absdiff(img, cv2.imdecode(enc, cv2.IMREAD_COLOR)).astype(np.float32).mean(axis=2)
    ok, enc75 = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    ela75 = cv2.absdiff(img, cv2.imdecode(enc75, cv2.IMREAD_COLOR)).astype(np.float32).mean(axis=2)

    lap = cv2.Laplacian(gray_full, cv2.CV_32F)                          # sharpness (affected by noise)
    lap_dn = cv2.Laplacian(cv2.medianBlur(gray_full, 5), cv2.CV_32F)    # sharpness with noise removed

    noise_cells = [c.std() for c in grid(residual)]
    ela_cells = [c.mean() for c in grid(ela)]
    sharp_cells = [c.var() for c in grid(lap)]

    noise_fine = [c.std() for c in grid(residual, FINE)]
    ela_fine = [c.mean() for c in grid(ela, FINE)]
    ela75_fine = [c.mean() for c in grid(ela75, FINE)]
    sharp_fine = [c.var() for c in grid(lap, FINE)]
    sharp_dn_fine = [c.var() for c in grid(lap_dn, FINE)]

    block_strength, block_mismatch = jpeg_grid_cells(gray_full)

    # resized features
    small = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # LBP texture: 4x4 histograms, plus how different each 8x8 cell is from the card's average texture
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_vec = np.concatenate([np.histogram(c, bins=10, range=(0, 10), density=True)[0]
                              for c in grid(lbp, 4)])
    lbp_dist = chi2_to_mean(np.array([np.histogram(c, bins=10, range=(0, 10), density=True)[0]
                                      for c in grid(lbp)]))

    # HOG edge directions per 8x8 cell, compared to the card's average
    hog_cells = hog(gray, orientations=9,
                    pixels_per_cell=(IMG_SIZE[1] // GRID, IMG_SIZE[0] // GRID),
                    cells_per_block=(1, 1), feature_vector=False).reshape(-1, 9)
    hog_dist = chi2_to_mean(hog_cells)

    # color in LAB (HSV hue wraps around at red, which breaks averaging)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_cells = np.array([c.reshape(-1, 3).mean(axis=0) for c in grid(lab, 4)])
    color_dev = np.linalg.norm(color_cells - color_cells.mean(axis=0), axis=1)

    parts = {
        "global": [residual.std(), np.median(noise_fine), ela.mean(), ela75.mean(),
                   lap.var(), np.median(sharp_fine), lap_dn.var(), np.median(sharp_dn_fine)],
        "noise_profile": noise_vs_brightness(gray_full, residual),
        "lbp": lbp_vec,
        "color": color_cells.ravel(),
        "anomaly": (anomaly_stats(noise_cells) + anomaly_stats(ela_cells) + anomaly_stats(sharp_cells)
                    + anomaly_stats(lbp_dist) + anomaly_stats(hog_dist) + anomaly_stats(color_dev)),
        "anomaly_fine": (anomaly_stats(noise_fine) + anomaly_stats(ela_fine) + anomaly_stats(ela75_fine)
                         + anomaly_stats(sharp_fine) + anomaly_stats(sharp_dn_fine)
                         + anomaly_stats(block_strength)),
        "jpeg_grid": [block_mismatch],
        "copymove": copy_move_score(gray_full),
    }
    return np.concatenate([np.asarray(v, dtype=np.float32) for v in parts.values()]), parts
