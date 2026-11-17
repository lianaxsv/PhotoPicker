"""OpenCV photo-quality metrics and conservative automatic corrections."""
from __future__ import annotations
import cv2
import numpy as np

DEFAULT_THRESHOLDS = {
    'blur': 75.0,
    'dark': 75.0,
    'bright': 205.0,
    'low_contrast': 38.0,
    'clipping_fraction': 0.18,
}

def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError('Image must be a nonempty NumPy array')
    if image.dtype != np.uint8:
        raise ValueError('Expected an 8-bit image (uint8)')
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError('Expected grayscale, BGR, or BGRA image')

def analyze_image(image: np.ndarray, thresholds: dict | None = None) -> dict:
    """Return explainable quality metrics and flags. Heuristic score is not a universal aesthetic judgment."""
    p = DEFAULT_THRESHOLDS | (thresholds or {})
    bgr = _as_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_fraction = float(np.mean(gray <= 12))
    bright_fraction = float(np.mean(gray >= 243))
    flags = []
    if sharpness < p['blur']:
        flags.append('possibly_blurry')
    if brightness < p['dark']:
        flags.append('underexposed')
    if brightness > p['bright'] or bright_fraction > p['clipping_fraction']:
        flags.append('overexposed')
    if contrast < p['low_contrast']:
        flags.append('low_contrast')
    if dark_fraction > p['clipping_fraction']:
        flags.append('shadow_clipping')
    if bright_fraction > p['clipping_fraction']:
        flags.append('highlight_clipping')
    sharp_score = float(np.clip(np.log1p(sharpness) / np.log1p(500), 0, 1))
    exposure_score = float(np.clip(1 - abs(brightness - 128) / 128, 0, 1))
    contrast_score = float(np.clip(contrast / 65, 0, 1))
    clipping_penalty = float(np.clip((dark_fraction + bright_fraction) * 0.5, 0, 0.4))
    quality_score = float(100 * np.clip(
        0.4 * sharp_score + 0.35 * exposure_score + 0.25 * contrast_score - clipping_penalty, 0, 1))
    return {
        'sharpness': round(sharpness, 3), 'brightness': round(brightness, 3),
        'contrast': round(contrast, 3), 'shadow_fraction': round(dark_fraction, 4),
        'highlight_fraction': round(bright_fraction, 4),
        'quality_score': round(quality_score, 2), 'issues': flags,
        'width': int(gray.shape[1]), 'height': int(gray.shape[0]),
    }

def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """gamma<1 lightens; gamma>1 darkens when output=255*(input/255)^gamma."""
    if gamma <= 0:
        raise ValueError('gamma must be > 0')
    lut = np.round(255 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return cv2.LUT(_as_bgr(image), lut)

def adaptive_gamma(image: np.ndarray, target_brightness: float = 125.0) -> tuple[np.ndarray, float]:
    """Choose bounded gamma based on mean grayscale brightness."""
    bgr = _as_bgr(image)
    mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
    normalized = max(mean / 255.0, 0.02)
    target = np.clip(target_brightness / 255.0, 0.05, 0.95)
    gamma = float(np.clip(np.log(target) / np.log(normalized), 0.45, 1.8))
    return gamma_correct(bgr, gamma), gamma

def clahe_luminance(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    bgr = _as_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((eq, a, b)), cv2.COLOR_LAB2BGR)

def unsharp_mask(image: np.ndarray, strength: float = 0.65) -> np.ndarray:
    bgr = _as_bgr(image)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(bgr, 1 + strength, blurred, -strength, 0)

def enhance_image(image: np.ndarray, mode: str = 'auto') -> tuple[np.ndarray, dict]:
    """Conservatively improve exposure, contrast and mild softness; severe blur may be unrecoverable."""
    out = _as_bgr(image).copy()
    before = analyze_image(out)
    applied: list[str] = []
    if mode not in {'auto', 'gamma', 'clahe', 'sharpen', 'none'}:
        raise ValueError(f'Unknown enhancement mode: {mode}')
    if mode in {'auto', 'gamma'} and (mode == 'gamma' or 'underexposed' in before['issues'] or 'overexposed' in before['issues']):
        out, gamma = adaptive_gamma(out)
        applied.append(f'adaptive_gamma({gamma:.2f})')
    if mode in {'auto', 'clahe'} and (mode == 'clahe' or 'low_contrast' in analyze_image(out)['issues']):
        out = clahe_luminance(out)
        applied.append('CLAHE')
    if mode in {'auto', 'sharpen'} and (mode == 'sharpen' or (25 <= before['sharpness'] < DEFAULT_THRESHOLDS['blur'])):
        out = unsharp_mask(out)
        applied.append('unsharp_mask')
    return out, {'before': before, 'after': analyze_image(out), 'applied': applied}

def rank_images(images: dict[str, np.ndarray]) -> list[dict]:
    """Sort labeled photos by quality_score descending."""
    records = [{'filename': name, **analyze_image(img)} for name, img in images.items()]
    return sorted(records, key=lambda r: r['quality_score'], reverse=True)

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Not a valid supported image')
    return img

def encode_jpeg(image: np.ndarray, quality: int = 95) -> bytes:
    ok, encoded = cv2.imencode('.jpg', _as_bgr(image), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('Failed to encode JPEG')
    return encoded.tobytes()

def enhance_studio(image: np.ndarray, intensity: float = 0.75, style: str = 'Balanced') -> tuple[np.ndarray, dict]:
    """Strength-controlled photo enhancement using tone mapping, CLAHE and selective sharpening."""
    if not 0 <= intensity <= 1:
        raise ValueError('intensity must be in [0,1]')
    if style not in {'Natural', 'Balanced', 'Vivid'}:
        raise ValueError('Unknown style')
    source = _as_bgr(image)
    before = analyze_image(source)
    if intensity == 0:
        return source.copy(), {'before': before, 'after': before, 'applied': [], 'mean_absolute_change': 0.0}
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lum, a, b = cv2.split(lab)
    mean = float(lum.mean())
    spread = float(lum.std())
    target = 137 if style == 'Vivid' else 132 if style == 'Balanced' else 125
    gamma = float(np.clip(np.log(target/255) / np.log(max(mean/255, 0.015)), 0.40, 1.75))
    if 95 < mean < 165:
        gamma = 1 + (gamma - 1) * 0.4
    gamma = 1 + (gamma - 1) * intensity
    lut = np.clip(np.round(255 * (np.arange(256)/255.0)**gamma),0,255).astype('uint8')
    corrected = cv2.LUT(lum, lut)
    clip = {'Natural': 1.35, 'Balanced': 2.2, 'Vivid': 3.1}[style]
    contrast = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(corrected)
    contrast_blend = intensity * (0.33 if style == 'Natural' else 0.62 if style == 'Balanced' else 0.86)
    lum_out = cv2.addWeighted(contrast, contrast_blend, corrected, 1-contrast_blend, 0)
    if spread < 55:
        multiplier = 1 + intensity * min((55-spread)/80, 0.23)
        lum_out = np.clip((lum_out.astype('float32') - 127.5)*multiplier+127.5, 0,255).astype('uint8')
    corrected_bgr = cv2.cvtColor(cv2.merge([lum_out,a,b]), cv2.COLOR_LAB2BGR)
    if style == 'Vivid':
        hsv = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1].astype('float32')*(1+0.12*intensity),0,255).astype('uint8')
        corrected_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if before['sharpness'] >= 25 and style != 'Natural':
        sharp = unsharp_mask(corrected_bgr, strength=0.25 * intensity)
        corrected_bgr = cv2.addWeighted(sharp, 0.65, corrected_bgr, 0.35, 0)
    difference = float(np.mean(np.abs(corrected_bgr.astype('float32')-source.astype('float32'))))
    return corrected_bgr, {
        'before': before, 'after': analyze_image(corrected_bgr),
        'applied': [f'adaptive luminance (gamma {gamma:.2f})', 'local contrast (CLAHE)',
                    'selective sharpening' if before['sharpness']>=25 and style!='Natural' else 'color preserved'],
        'mean_absolute_change': round(difference, 2)}
