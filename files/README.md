# 🛡️ CatfishGuard v5 — AI Catfishing Detection Tool

**GWC Challenge 2025–26**

CatfishGuard is a web-based tool that helps users detect AI-generated catfish profiles on social media. It analyzes images for signs of AI generation and extracts profile metadata from Instagram screenshots to produce a combined catfish risk score.

---

## How It Works

CatfishGuard operates in two modes:

### 🖼️ Image Analysis Mode
Upload any photo (profile picture, dating app photo, etc.) and CatfishGuard analyzes it across 5 detection layers calibrated on real AI-generated vs authentic photographs:

| Layer | What It Detects | Why It Works |
|-------|----------------|--------------|
| **Saturation Uniformity** | AI images have unnaturally uniform color across skin | Real skin has natural color variation from blood flow, freckles, and uneven tones |
| **Noise Uniformity** | AI images have spatially uniform noise patterns | Real cameras produce noise that varies with brightness and scene content |
| **Texture Ratio** | AI images have similar sharpness in face and background | Real photos show sharper faces with blurred backgrounds (depth of field) |
| **Noise Correlation** | AI images lack brightness-dependent noise | Real camera sensors produce Poisson noise that increases with brightness |
| **Frequency Decay** | AI images have steeper frequency rolloff | Real camera optics produce characteristic frequency transfer functions |

### 📱 Profile Screenshot Mode
Upload a screenshot of an Instagram profile and CatfishGuard extracts profile metadata using OCR (Optical Character Recognition) in addition to running image analysis:

- **Post count** — Accounts with 0-3 posts are flagged as high risk
- **Follower/Following ratio** — Mass-following with few followers back is a classic catfish pattern
- **Posts-to-followers consistency** — 10,000 followers from 3 posts suggests purchased followers
- **Bio analysis** — Detects suspicious patterns like DM solicitation, crypto/forex mentions, romantic bait language, and payment solicitation

The final score combines metadata analysis (55%) with image analysis (45%) for maximum detection accuracy.

---

## Screenshot Robustness

A key challenge in AI detection is that screenshots degrade image quality, adding compression artifacts that can mask AI signatures. CatfishGuard is designed to handle this by using **relative measurements** rather than absolute pixel-level analysis:

- Saturation uniformity compares color variation ratios, not absolute values
- Texture ratio measures face-vs-background sharpness difference, not absolute sharpness
- Noise uniformity measures spatial variation patterns, not noise levels

In testing, the same AI image scored **58.9 (original)** vs **59.3 (screenshot)** — a difference of only 0.4 points.

---

## Risk Levels

| Score | Level | Meaning |
|-------|-------|---------|
| 70–99 | 🔴 **HIGH** | Strong indicators of catfishing or AI generation |
| 50–69 | 🟡 **MEDIUM** | Several suspicious characteristics detected |
| 30–49 | 🔵 **LOW** | Minor indicators, likely genuine |
| 0–29 | 🟢 **MINIMAL** | No significant red flags detected |

---

## Installation

### Requirements
- Python 3.8+
- pip

### Step 1: Install Python dependencies

```bash
pip install flask pillow numpy
```

### Step 2 (Optional): Install Tesseract OCR for Profile Screenshot Mode

Profile Screenshot mode requires Tesseract OCR to extract text from screenshots. Image Analysis mode works without it.

**Windows:**
1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
3. Install the Python wrapper: `pip install pytesseract`
4. Add this line near the top of `app5.py`, after the pytesseract import:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

**macOS:**
```bash
brew install tesseract
pip install pytesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr
pip install pytesseract
```

### Step 3: Run the app

```bash
python app5.py
```

Open your browser and go to `http://127.0.0.1:5000`

---

## Project Structure

```
catfish-detector/
├── app5.py                  # Flask backend with detection algorithms
├── templates/
│   └── index5.html          # Frontend (HTML/CSS/JS, self-contained)
├── uploads/                 # Temporary upload directory (auto-created)
└── README.md
```

---

## Detection Accuracy

Calibrated and tested on 12 AI-generated images and 11 authentic photographs:

| Metric | Result |
|--------|--------|
| AI average score | 65.7 |
| Real average score | 40.5 |
| Score gap | 25.2 points |
| Best accuracy (threshold=45) | 75% (10/13 AI flagged, 8/11 real cleared) |

### Known Strengths
- Detects the most common catfish pattern: new account, AI headshot, few posts, mass-following
- Screenshot-robust — works on saved images and screenshots equally well
- Dual-mode detection combines image analysis with profile metadata for stronger signals
- Handles diverse skin tones and photo styles

### Known Limitations
- Complex AI scenes (e.g., outdoor settings with many objects) score lower than simple headshots
- Professional studio photos of real people can score in the MEDIUM range due to smooth lighting and makeup
- Cannot detect catfish profiles that use stolen real photographs instead of AI-generated images
- OCR accuracy depends on screenshot quality and resolution
- Not a replacement for human judgment — always verify through multiple methods

---

## How Catfish Accounts Typically Look

CatfishGuard is most effective against accounts that exhibit these patterns:

1. **AI-generated profile picture** — Waxy/glossy skin, too-perfect features, plain backgrounds, unnaturally vivid eyes
2. **Few or zero posts** — The account was created quickly with no real content history
3. **Skewed follower ratio** — Follows hundreds/thousands but has very few followers
4. **Suspicious bio** — Contains DM solicitation, crypto/forex mentions, or romantic bait language
5. **No real-world context** — All photos are isolated portraits with no friends, events, or recognizable locations

---

## Safety Tips for Users

1. **Reverse image search** the profile picture on Google Images or TinEye
2. **Look for the "waxy shine"** — AI skin looks coated in vaseline
3. **Check ears, fingers, and hairlines** — AI often distorts these
4. **Zoom into backgrounds** for warped lines or melting objects
5. **Look for text in images** — AI text is almost always garbled
6. **Ask for a video call** — catfish always have excuses to avoid live interaction
7. **Check mutual followers** — do real people you know follow this account?
8. **Be wary of accounts that immediately push to DMs or WhatsApp**

---

## Tech Stack

- **Backend:** Python, Flask
- **Image Analysis:** NumPy, Pillow (PIL)
- **OCR:** Tesseract, pytesseract (optional)
- **Frontend:** HTML, CSS, JavaScript (no framework, self-contained)

---

## License

Built for the GWC Challenge 2025–26.
