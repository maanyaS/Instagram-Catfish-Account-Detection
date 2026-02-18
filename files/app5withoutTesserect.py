"""
CatfishGuard v5 - AI Catfishing Detection Tool

Two modes:
  1. IMAGE MODE  - Analyze a single image for AI generation signs
  2. PROFILE MODE - Analyze an Instagram profile screenshot
                   (extracts follower/following counts, bio, post count,
                    and runs AI detection on the profile picture)

Calibrated on 12 AI + 11 Real test images. Key discriminators:
  - Saturation uniformity (gap=0.226)
  - Noise spatial uniformity (gap=0.174)
  - Face-vs-background texture ratio (gap=0.45)
  - Noise-brightness correlation (gap=0.091)
"""
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageFilter, ImageStat
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
import numpy as np
import re
import os
import uuid

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prep_image(image, max_dim=512):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    ratio = min(max_dim / image.width, max_dim / image.height)
    if ratio < 1:
        image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)
    return image

def get_center(arr, frac=0.5):
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    rh, rw = int(h * frac / 2), int(w * frac / 2)
    rh, rw = max(rh, 1), max(rw, 1)
    return arr[cy-rh:cy+rh, cx-rw:cx+rw]


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  IMAGE ANALYSIS LAYERS — calibrated on real test data            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def analyze_saturation_uniformity(image):
    """
    #1 DISCRIMINATOR (gap=0.226)
    AI: sat_cv avg=0.462 [0.237..0.862]
    Real: sat_cv avg=0.688 [0.358..1.051]
    
    AI images have more uniform saturation. Score maps sat_cv
    through a sigmoid centered at 0.55 (midpoint of distributions).
    """
    hsv = np.array(image.convert('HSV')).astype(float)
    center_sat = get_center(hsv[:, :, 1], 0.50)

    if center_sat.size < 100 or center_sat.mean() < 5:
        return 0.35  # Can't determine

    sat_cv = center_sat.std() / (center_sat.mean() + 1e-10)

    # Sigmoid centered at 0.55, steep enough to separate distributions
    # sat_cv < 0.35 → score ~0.85 (very AI-like)
    # sat_cv ≈ 0.55 → score ~0.50 (ambiguous)
    # sat_cv > 0.80 → score ~0.15 (very real-like)
    score = 1.0 / (1.0 + np.exp(8.0 * (sat_cv - 0.55)))

    return float(max(0, min(1, score)))


def analyze_noise_uniformity(image):
    """
    #2 DISCRIMINATOR (gap=0.174)
    AI: noise_cv avg=0.698 [0.271..1.041]
    Real: noise_cv avg=0.872 [0.676..1.180]
    
    AI images have more spatially uniform noise (lower CV).
    Real cameras produce noise that varies with scene content.
    """
    gray = np.array(image.convert('L')).astype(float)
    h, w = gray.shape

    blur = np.array(image.convert('L').filter(
        ImageFilter.GaussianBlur(radius=2)
    )).astype(float)
    noise = gray - blur

    block_size = 32
    nvs = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            nvs.append(np.std(noise[y:y+block_size, x:x+block_size]))

    if len(nvs) < 4:
        return 0.35

    nv = np.array(nvs)
    ncv = nv.std() / (nv.mean() + 1e-10)

    # Sigmoid centered at 0.78
    # ncv < 0.50 → score ~0.85 (uniform noise = AI)
    # ncv ≈ 0.78 → score ~0.50
    # ncv > 1.00 → score ~0.15 (varied noise = real)
    score = 1.0 / (1.0 + np.exp(6.0 * (ncv - 0.78)))

    return float(max(0, min(1, score)))


def analyze_texture_ratio(image):
    """
    #3 DISCRIMINATOR (ratio gap=0.45)
    AI: face/bg texture ratio avg=1.17
    Real: face/bg texture ratio avg=1.62
    
    Real photos typically have sharp faces + blurred backgrounds
    (depth of field), creating a HIGH ratio.
    AI images have more uniform sharpness throughout (LOW ratio).
    
    Screenshot-robust: both regions degrade equally.
    """
    gray = np.array(image.convert('L')).astype(float)
    h, w = gray.shape

    blur = np.array(image.convert('L').filter(
        ImageFilter.GaussianBlur(radius=2)
    )).astype(float)
    texture = np.abs(gray - blur)

    center = get_center(texture, 0.45)
    mask = np.ones_like(texture, dtype=bool)
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.45 / 2), int(w * 0.45 / 2)
    rh, rw = max(rh, 1), max(rw, 1)
    mask[cy-rh:cy+rh, cx-rw:cx+rw] = False
    periphery = texture[mask]

    if len(periphery) < 100 or center.size < 100:
        return 0.35

    center_mean = center.mean()
    periph_mean = periphery.mean()

    if periph_mean < 0.5:
        return 0.35  # Can't determine

    ratio = center_mean / (periph_mean + 1e-10)

    # AI: ratio ≈ 1.17 (uniform texture), Real: ratio ≈ 1.62 (face sharper)
    # LOWER ratio = more AI-like
    # Sigmoid centered at 1.4
    # ratio < 1.1 → score ~0.80
    # ratio ≈ 1.4 → score ~0.50
    # ratio > 1.7 → score ~0.20
    score = 1.0 / (1.0 + np.exp(4.0 * (ratio - 1.40)))

    return float(max(0, min(1, score)))


def analyze_noise_correlation(image):
    """
    #4 DISCRIMINATOR (gap=0.091)
    AI: corr avg=-0.230
    Real: corr avg=-0.139
    
    Weaker signal but adds value when combined.
    More negative correlation = slightly more AI-like.
    """
    gray = np.array(image.convert('L')).astype(float)
    h, w = gray.shape

    blur = np.array(image.convert('L').filter(
        ImageFilter.GaussianBlur(radius=2)
    )).astype(float)
    noise = gray - blur

    block_size = 32
    bvs, nvs = [], []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            bvs.append(blur[y:y+block_size, x:x+block_size].mean())
            nvs.append(np.std(noise[y:y+block_size, x:x+block_size]))

    if len(bvs) < 10:
        return 0.35

    bv, nv = np.array(bvs), np.array(nvs)
    bc, nc = bv - bv.mean(), nv - nv.mean()
    bstd, nstd = bc.std(), nc.std()

    if bstd < 1e-6 or nstd < 1e-6:
        return 0.5

    corr = np.mean(bc * nc) / (bstd * nstd)

    # Sigmoid centered at -0.18
    # corr < -0.4 → score ~0.70
    # corr ≈ -0.18 → score ~0.50
    # corr > 0.1 → score ~0.30
    score = 1.0 / (1.0 + np.exp(5.0 * (corr + 0.18)))

    return float(max(0, min(1, score)))


def analyze_frequency_decay(image):
    """
    Weak discriminator (gap=0.058) but included for robustness.
    AI: slope avg=-1.593, Real: slope avg=-1.536
    """
    gray = np.array(image.convert('L')).astype(float)
    h, w = gray.shape

    fs = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fs)
    cy, cx = h // 2, w // 2
    mr = min(h, w) // 2
    yc, xc = np.ogrid[:h, :w]
    dist = np.sqrt((yc - cy)**2 + (xc - cx)**2)

    nb = 32
    rp = []
    for i in range(nb):
        ri, ro = (i / nb) * mr, ((i + 1) / nb) * mr
        m = (dist >= ri) & (dist < ro)
        rp.append(np.mean(mag[m]) if m.sum() > 0 else 0)

    rp = np.array(rp)
    rp = rp / (rp[0] + 1e-10)

    valid = rp[2:] > 0
    if valid.sum() < 5:
        return 0.35

    lf = np.log(np.arange(2, nb)[valid] + 1)
    lp = np.log(rp[2:][valid] + 1e-10)
    n = len(lf)
    slope = (n * np.sum(lf * lp) - np.sum(lf) * np.sum(lp)) / \
            (n * np.sum(lf**2) - np.sum(lf)**2 + 1e-10)

    # Sigmoid centered at -1.56
    # slope < -1.7 → score ~0.60
    # slope ≈ -1.56 → score ~0.50
    # slope > -1.4 → score ~0.40
    score = 1.0 / (1.0 + np.exp(5.0 * (slope + 1.56)))

    return float(max(0, min(1, score)))


def run_image_analysis(image):
    """Run all image analysis layers and compute final score."""
    image = prep_image(image)

    layers = {
        'saturation_uniformity': analyze_saturation_uniformity(image),
        'noise_uniformity': analyze_noise_uniformity(image),
        'texture_ratio': analyze_texture_ratio(image),
        'noise_correlation': analyze_noise_correlation(image),
        'frequency_decay': analyze_frequency_decay(image),
    }

    # Weights proportional to discriminative power
    weights = {
        'saturation_uniformity': 0.30,   # Best (gap=0.226)
        'noise_uniformity': 0.25,        # Second (gap=0.174)
        'texture_ratio': 0.20,           # Third (ratio gap=0.45)
        'noise_correlation': 0.15,       # Fourth (gap=0.091)
        'frequency_decay': 0.10,         # Weakest (gap=0.058)
    }

    raw = sum(layers[k] * weights[k] for k in weights) * 100

    # Mild boost when multiple layers agree
    values = list(layers.values())
    above_55 = sum(1 for v in values if v > 0.55)
    above_65 = sum(1 for v in values if v > 0.65)

    if above_55 >= 4:
        boost = 1.20
    elif above_55 >= 3:
        boost = 1.12
    else:
        boost = 1.0

    if above_65 >= 3:
        boost *= 1.10

    score = min(99, raw * boost)
    return round(score, 1), {k: round(v, 3) for k, v in layers.items()}


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PROFILE SCREENSHOT ANALYSIS                                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

def extract_profile_text(image):
    if not HAS_TESSERACT:
        return ""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception:
        return ""


def parse_number(s):
    s = s.strip().replace(',', '').replace(' ', '')
    try:
        if 'k' in s.lower():
            return int(float(s.lower().replace('k', '')) * 1000)
        elif 'm' in s.lower():
            return int(float(s.lower().replace('m', '')) * 1000000)
        elif 'b' in s.lower():
            return int(float(s.lower().replace('b', '')) * 1000000000)
        else:
            return int(float(s))
    except (ValueError, TypeError):
        return None


def parse_instagram_stats(text):
    stats = {'posts': None, 'followers': None, 'following': None}
    text_clean = text.lower().replace('\n', ' ').replace('  ', ' ')

    # Pattern 1: "X posts  Y followers  Z following"
    m = re.search(r'([\d,.]+[kmb]?)\s*posts?\s+([\d,.]+[kmb]?)\s*followers?\s+([\d,.]+[kmb]?)\s*following', text_clean)
    if m:
        stats['posts'] = parse_number(m.group(1))
        stats['followers'] = parse_number(m.group(2))
        stats['following'] = parse_number(m.group(3))
        return stats

    # Pattern 2: Individual matches
    for label, key in [('post', 'posts'), ('follower', 'followers'), ('following', 'following')]:
        m = re.search(r'([\d,.]+[kmb]?)\s*' + label, text_clean)
        if m:
            stats[key] = parse_number(m.group(1))

    return stats


def analyze_profile_metadata(stats, bio_text):
    signals = {}
    warnings = []

    # --- Post count ---
    posts = stats.get('posts')
    if posts is not None:
        if posts == 0:
            signals['post_count'] = 1.0
            warnings.append({'type': 'high', 'icon': '🔴',
                'title': 'Zero Posts',
                'detail': 'This account has no posts. Catfish accounts are often freshly created with no content.'})
        elif posts <= 3:
            signals['post_count'] = 0.8
            warnings.append({'type': 'high', 'icon': '🔴',
                'title': f'Very Few Posts ({posts})',
                'detail': 'Accounts with very few posts are a major red flag. Real users accumulate posts over time.'})
        elif posts <= 10:
            signals['post_count'] = 0.5
            warnings.append({'type': 'medium', 'icon': '🟡',
                'title': f'Low Post Count ({posts})',
                'detail': 'Relatively low post count could indicate a new or recently created account.'})
        else:
            signals['post_count'] = 0.1
    else:
        signals['post_count'] = 0.3

    # --- Follower/Following ratio ---
    followers = stats.get('followers')
    following = stats.get('following')

    if followers is not None and following is not None and following > 0:
        ratio = followers / following

        if following > 1000 and followers < 100:
            signals['follow_ratio'] = 0.9
            warnings.append({'type': 'high', 'icon': '🔴',
                'title': f'Suspicious Follow Ratio ({followers}/{following})',
                'detail': 'Follows huge numbers but almost nobody follows back — classic catfish/spam pattern.'})
        elif following > 500 and ratio < 0.1:
            signals['follow_ratio'] = 0.8
            warnings.append({'type': 'high', 'icon': '🔴',
                'title': 'Very Low Follower Ratio',
                'detail': f'Following {following} but only {followers} followers. Genuine users have more balanced ratios.'})
        elif ratio < 0.2 and following > 200:
            signals['follow_ratio'] = 0.6
            warnings.append({'type': 'medium', 'icon': '🟡',
                'title': 'Unbalanced Follow Ratio',
                'detail': 'Follower-to-following ratio is low, which can indicate a fake account.'})
        elif followers > 10000 and posts is not None and posts < 5:
            signals['follow_ratio'] = 0.7
            warnings.append({'type': 'high', 'icon': '🔴',
                'title': 'Suspicious Followers vs Posts',
                'detail': f'{followers} followers but only {posts} posts — suggests purchased followers.'})
        else:
            signals['follow_ratio'] = 0.15
    else:
        signals['follow_ratio'] = 0.3

    # --- Posts-to-followers consistency ---
    if posts is not None and followers is not None and posts > 0:
        pf_ratio = followers / posts
        if pf_ratio > 2000 and posts < 10:
            signals['engagement_consistency'] = 0.8
            warnings.append({'type': 'high', 'icon': '🔴',
                'title': 'Impossible Growth Pattern',
                'detail': f'{followers} followers from {posts} posts suggests fake followers.'})
        else:
            signals['engagement_consistency'] = 0.15
    else:
        signals['engagement_consistency'] = 0.3

    # --- Bio analysis ---
    bio_lower = bio_text.lower()
    suspicious_patterns = [
        (r'dm\s*(me|for)', 'DM solicitation'),
        (r'(crypto|forex|bitcoin|trading|invest)', 'Crypto/trading'),
        (r'(link\s*in\s*bio|linktree|linktr)', 'Link promotion'),
        (r'(single|looking|lonely|love)', 'Romantic bait'),
        (r'(cashapp|venmo|paypal|send)', 'Payment solicitation'),
        (r'(18\+|nsfw|adult|onlyfans)', 'Adult content'),
    ]

    bio_flags = sum(1 for p, _ in suspicious_patterns if re.search(p, bio_lower))

    if bio_flags >= 3:
        signals['bio_analysis'] = 0.9
        warnings.append({'type': 'high', 'icon': '🔴',
            'title': 'Highly Suspicious Bio',
            'detail': 'Multiple patterns commonly found in catfish/scam accounts.'})
    elif bio_flags >= 2:
        signals['bio_analysis'] = 0.6
        warnings.append({'type': 'medium', 'icon': '🟡',
            'title': 'Suspicious Bio Patterns',
            'detail': 'Bio contains language commonly associated with fake accounts.'})
    elif bio_flags >= 1:
        signals['bio_analysis'] = 0.35
    elif len(bio_text.strip()) < 10:
        signals['bio_analysis'] = 0.4
        warnings.append({'type': 'medium', 'icon': '🟡',
            'title': 'Minimal or Empty Bio',
            'detail': 'Very short or empty bios are common on quickly-created fake accounts.'})
    else:
        signals['bio_analysis'] = 0.1

    return signals, warnings


def run_profile_analysis(image):
    raw_text = extract_profile_text(image)
    stats = parse_instagram_stats(raw_text)
    bio = raw_text

    meta_signals, meta_warnings = analyze_profile_metadata(stats, bio)

    # Image analysis on full screenshot
    image_score, image_layers = run_image_analysis(image)

    # Analyze sub-regions
    img_array = np.array(prep_image(image, 600))
    h, w = img_array.shape[:2]

    if h > 200:
        bottom_region = Image.fromarray(img_array[h//2:, :])
        bottom_score, _ = run_image_analysis(bottom_region)
    else:
        bottom_score = image_score

    if h > 200 and w > 200:
        top_region = Image.fromarray(img_array[:h//3, :w//2])
        top_score, _ = run_image_analysis(top_region)
    else:
        top_score = image_score

    # Combine scores
    meta_weights = {
        'post_count': 0.30,
        'follow_ratio': 0.30,
        'engagement_consistency': 0.20,
        'bio_analysis': 0.20,
    }
    meta_score = sum(meta_signals.get(k, 0.3) * wt for k, wt in meta_weights.items()) * 100

    avg_image_score = image_score * 0.5 + top_score * 0.3 + bottom_score * 0.2

    stats_found = sum(1 for v in stats.values() if v is not None)
    if stats_found >= 2:
        final = meta_score * 0.55 + avg_image_score * 0.45
    elif stats_found == 1:
        final = meta_score * 0.35 + avg_image_score * 0.65
    else:
        final = avg_image_score
        meta_warnings.append({'type': 'medium', 'icon': '🟡',
            'title': 'Could Not Read Profile Stats',
            'detail': 'OCR could not extract follower/following counts. Analysis based on image characteristics only.'})

    final = round(min(99, max(0, final)), 1)

    if avg_image_score > 55:
        meta_warnings.append({'type': 'high', 'icon': '🔴',
            'title': 'AI-Generated Images Detected',
            'detail': 'Strong AI generation characteristics found in profile picture or posts.'})
    elif avg_image_score > 40:
        meta_warnings.append({'type': 'medium', 'icon': '🟡',
            'title': 'Possible AI-Generated Images',
            'detail': 'Some AI generation characteristics detected in images.'})

    return {
        'overall_score': final,
        'risk': get_risk_level(final),
        'metadata': {
            'stats': stats,
            'bio_preview': bio[:200] if bio else '',
            'stats_found': stats_found,
        },
        'meta_signals': {k: round(v, 3) for k, v in meta_signals.items()},
        'meta_score': round(meta_score, 1),
        'image_score': round(avg_image_score, 1),
        'image_layers': image_layers,
        'warnings': meta_warnings,
    }


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  WARNINGS & RISK                                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝

def generate_image_warnings(layers, score):
    warnings = []

    checks = [
        ('saturation_uniformity', 0.60, 0.45,
         'Unnaturally Uniform Color Saturation',
         'The face region has suspiciously uniform color saturation. Real skin has natural color variation from blood flow, freckles, and uneven tones that AI images lack.',
         'Somewhat Uniform Color',
         'The face shows less color variation than typical photographs.'),

        ('noise_uniformity', 0.60, 0.45,
         'Artificially Uniform Noise',
         'Image noise is too spatially uniform. Real cameras produce varied noise levels across different brightness regions and scene elements.',
         'Somewhat Uniform Noise',
         'Noise patterns are more uniform than typical camera photos.'),

        ('texture_ratio', 0.60, 0.45,
         'Face/Background Sharpness Mismatch',
         'The face and background have unusually similar texture levels. Real photos typically show sharper faces relative to backgrounds due to camera depth-of-field.',
         'Reduced Depth-of-Field Effect',
         'The difference between face and background sharpness is lower than expected.'),

        ('noise_correlation', 0.60, 0.45,
         'Non-Physical Noise Pattern',
         'The noise pattern does not follow the brightness-dependent behavior of real camera sensors.',
         'Unusual Noise Characteristics',
         'Some noise characteristics are unusual for real camera output.'),

        ('frequency_decay', 0.60, 0.45,
         'AI Frequency Signature',
         'The image frequency spectrum shows steeper-than-normal rolloff, a characteristic of AI image generation.',
         'Unusual Frequency Profile',
         'Some frequency characteristics are atypical for real photographs.'),
    ]

    for key, ht, mt, htitle, hdetail, mtitle, mdetail in checks:
        val = layers.get(key, 0)
        if val > ht:
            warnings.append({'type': 'high', 'icon': '🔴', 'title': htitle, 'detail': hdetail})
        elif val > mt:
            warnings.append({'type': 'medium', 'icon': '🟡', 'title': mtitle, 'detail': mdetail})

    if not warnings:
        warnings.append({
            'type': 'low', 'icon': '🟢',
            'title': 'No Major AI Indicators',
            'detail': 'No strong AI generation signals detected. The image appears natural, but always verify through other means.'
        })

    return warnings


def get_risk_level(score):
    if score >= 70:
        return {
            'level': 'HIGH', 'color': '#ef4444',
            'message': 'Strong indicators of catfishing or AI generation. Exercise extreme caution.'
        }
    elif score >= 50:
        return {
            'level': 'MEDIUM', 'color': '#f59e0b',
            'message': 'Several suspicious characteristics detected. Proceed with caution.'
        }
    elif score >= 30:
        return {
            'level': 'LOW', 'color': '#3b82f6',
            'message': 'Minor indicators found. Likely genuine, but stay alert.'
        }
    else:
        return {
            'level': 'MINIMAL', 'color': '#10b981',
            'message': 'No significant red flags detected. Profile appears genuine.'
        }


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FLASK ROUTES                                                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

@app.route('/')
def index():
    return render_template('index5.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        score, layers = run_image_analysis(image)
        warnings = generate_image_warnings(layers, score)
        risk = get_risk_level(score)
        os.remove(filepath)

        return jsonify({
            'success': True,
            'mode': 'image',
            'overall_score': score,
            'risk': risk,
            'analysis': layers,
            'warnings': warnings,
            'tips': [
                'Reverse image search the picture on Google Images or TinEye.',
                'Look for the "waxy shine" — AI skin looks coated in vaseline.',
                'Check ears, fingers, and hairlines for distortions.',
                'Zoom into backgrounds for warped lines or melting objects.',
                'Look for any text — AI text is almost always garbled.',
                'Be cautious if they avoid video calls or live photos.',
            ]
        })
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/analyze-profile', methods=['POST'])
def analyze_profile():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        result = run_profile_analysis(image)
        os.remove(filepath)

        return jsonify({
            'success': True,
            'mode': 'profile',
            **result,
            'tips': [
                'Reverse image search the profile picture.',
                'Check when the account was created — new accounts are suspicious.',
                'Look at post variety — real people post diverse content.',
                'Check if comments are genuine or generic bot responses.',
                'Be wary of accounts that push to move to DMs or WhatsApp.',
                'Ask to video chat — catfish always avoid live interaction.',
                'Check mutual followers — do real people you know follow this account?',
            ]
        })
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
"""