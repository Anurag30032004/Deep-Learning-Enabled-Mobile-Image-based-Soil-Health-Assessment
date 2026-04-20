"""
Self-Calibrating AI — Automatic Image Calibration Layer
=========================================================
Detects lighting conditions in soil images and applies adaptive
normalization (exposure + white balance) before classification.

Algorithms:
  - CLAHE in LAB color space for exposure normalization
  - Gray World algorithm for white balance correction
  - Channel-statistics heuristics for lighting detection
"""

import cv2
import numpy as np
import os

# ── Configurable thresholds ─────────────────────────────────────────────
BRIGHTNESS_HIGH = 200       # overexposed if mean brightness exceeds this
BRIGHTNESS_LOW = 50         # underexposed if mean brightness below this
CONTRAST_LOW = 30           # low-contrast if pixel std-dev below this
COLOR_CAST_DELTA = 25       # channel considered dominant if it exceeds others by this
CLAHE_CLIP_LIMIT = 3.0      # CLAHE clip limit (higher → more equalization)
CLAHE_GRID_SIZE = (8, 8)    # CLAHE tile grid size


# ── Lighting condition detection ────────────────────────────────────────

def detect_lighting_condition(img_bgr):
    """
    Analyze an image and classify its lighting condition.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image as read by cv2.imread.

    Returns
    -------
    str
        One of: 'overexposed', 'underexposed', 'warm_cast', 'cool_cast',
        'low_contrast', 'normal'.
    dict
        Diagnostic metrics: brightness, contrast, channel means.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r_mean = float(np.mean(img_rgb[:, :, 0]))
    g_mean = float(np.mean(img_rgb[:, :, 1]))
    b_mean = float(np.mean(img_rgb[:, :, 2]))
    brightness = float(np.mean(img_rgb))
    contrast = float(np.std(img_rgb))

    diagnostics = {
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'r_mean': round(r_mean, 2),
        'g_mean': round(g_mean, 2),
        'b_mean': round(b_mean, 2),
    }

    # Classify in priority order
    if brightness > BRIGHTNESS_HIGH:
        return 'overexposed', diagnostics
    if brightness < BRIGHTNESS_LOW:
        return 'underexposed', diagnostics
    if r_mean - g_mean > COLOR_CAST_DELTA and r_mean - b_mean > COLOR_CAST_DELTA:
        return 'warm_cast', diagnostics
    if b_mean - r_mean > COLOR_CAST_DELTA and b_mean - g_mean > COLOR_CAST_DELTA:
        return 'cool_cast', diagnostics
    if contrast < CONTRAST_LOW:
        return 'low_contrast', diagnostics

    return 'normal', diagnostics


# ── Exposure normalization (CLAHE in LAB) ───────────────────────────────

def normalize_exposure(img_bgr, condition):
    """
    Apply CLAHE-based adaptive exposure normalization.

    Stronger equalization for over/underexposed images; lighter touch for
    normal images. Works in LAB color space to adjust luminance without
    distorting color.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image.
    condition : str
        Lighting condition label from detect_lighting_condition().

    Returns
    -------
    np.ndarray
        Exposure-normalized BGR image.
    """
    # Adaptive clip limit based on severity
    clip_map = {
        'overexposed': 4.0,
        'underexposed': 4.0,
        'low_contrast': 3.5,
        'warm_cast': CLAHE_CLIP_LIMIT,
        'cool_cast': CLAHE_CLIP_LIMIT,
        'normal': 2.0,
    }
    clip = clip_map.get(condition, CLAHE_CLIP_LIMIT)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_GRID_SIZE)
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── White balance correction (Gray World) ───────────────────────────────

def correct_white_balance(img_bgr, condition):
    """
    Remove color cast using the Gray World assumption: the average of all
    channels should converge to the same neutral gray value.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image (exposure already normalized).
    condition : str
        Lighting condition — color cast conditions get stronger correction.

    Returns
    -------
    np.ndarray
        White-balance-corrected BGR image.
    """
    if condition == 'normal':
        # Minimal correction for well-lit images
        return img_bgr

    img = img_bgr.astype(np.float32)
    b_avg = np.mean(img[:, :, 0])
    g_avg = np.mean(img[:, :, 1])
    r_avg = np.mean(img[:, :, 2])
    overall_avg = (b_avg + g_avg + r_avg) / 3.0

    # Blend factor: stronger for color-cast images
    if condition in ('warm_cast', 'cool_cast'):
        alpha = 0.9   # heavy correction
    else:
        alpha = 0.6   # moderate correction

    # Scale each channel towards the overall average
    if b_avg > 0:
        img[:, :, 0] = np.clip(img[:, :, 0] * (1 - alpha + alpha * overall_avg / b_avg), 0, 255)
    if g_avg > 0:
        img[:, :, 1] = np.clip(img[:, :, 1] * (1 - alpha + alpha * overall_avg / g_avg), 0, 255)
    if r_avg > 0:
        img[:, :, 2] = np.clip(img[:, :, 2] * (1 - alpha + alpha * overall_avg / r_avg), 0, 255)

    return img.astype(np.uint8)


# ── Orchestrator ────────────────────────────────────────────────────────

def auto_calibrate(img_path):
    """
    Full automatic calibration pipeline for a single image.

    Parameters
    ----------
    img_path : str
        Path to the input image file.

    Returns
    -------
    calibrated_img : np.ndarray
        Calibrated BGR image.
    metadata : dict
        Calibration report containing:
          - condition: detected lighting condition
          - diagnostics: channel statistics of the original image
          - adjustments: list of corrections applied
          - brightness_delta: change in mean brightness after calibration
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    original_brightness = float(np.mean(img))

    # Step 1 — detect
    condition, diagnostics = detect_lighting_condition(img)

    # Step 2 — normalize exposure
    calibrated = normalize_exposure(img, condition)

    # Step 3 — correct white balance
    calibrated = correct_white_balance(calibrated, condition)

    # Compute deltas
    calibrated_brightness = float(np.mean(calibrated))

    adjustments = []
    if condition != 'normal':
        adjustments.append(f"Exposure normalization (CLAHE, condition={condition})")
    if condition in ('warm_cast', 'cool_cast'):
        adjustments.append(f"White balance correction (Gray World, cast={condition})")
    elif condition not in ('normal',):
        adjustments.append("Moderate white balance adjustment")
    if not adjustments:
        adjustments.append("Minimal correction (image well-lit)")

    metadata = {
        'condition': condition,
        'diagnostics': diagnostics,
        'adjustments': adjustments,
        'brightness_before': round(original_brightness, 2),
        'brightness_after': round(calibrated_brightness, 2),
        'brightness_delta': round(calibrated_brightness - original_brightness, 2),
    }

    return calibrated, metadata


def save_calibrated_image(calibrated_img, original_path, output_dir=None):
    """
    Save a calibrated image alongside the original, returning the new path.

    Parameters
    ----------
    calibrated_img : np.ndarray
        The calibrated BGR image.
    original_path : str
        Path to the original image (used for naming).
    output_dir : str, optional
        Directory to save into. Defaults to a 'calibrated' subfolder
        next to the original.

    Returns
    -------
    str
        Absolute path to the saved calibrated image.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(original_path), 'calibrated')
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(original_path)
    name, ext = os.path.splitext(base)
    cal_filename = f"{name}_calibrated{ext}"
    cal_path = os.path.join(output_dir, cal_filename)

    cv2.imwrite(cal_path, calibrated_img)
    return cal_path
