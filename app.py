import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import base64

st.set_page_config(
    page_title="OncoScan · Clinical Decision Support",
    page_icon="⬡",
    layout="wide"
)

# ==============================================================
# LOAD MODEL ASSETS
# ==============================================================
model        = joblib.load("model.pkl")
scaler       = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")  # exact features model was trained on

# ==============================================================
# FULL METADATA FOR ALL 30 POSSIBLE FEATURES
# (min, max, default, label, tooltip)
# Only entries that appear in feature_cols will be used
# ==============================================================
FEATURE_META = {
    # ── Mean ───────────────────────────────────────────────────
    "radius_mean":             (6.981,  28.11,   14.13, "Radius",            "Mean distance from center to perimeter"),
    "texture_mean":            (9.71,   39.28,   19.29, "Texture",           "Standard deviation of gray-scale values"),
    "perimeter_mean":          (43.79,  188.5,   91.97, "Perimeter",         "Mean perimeter of the cell nucleus"),
    "area_mean":               (143.5,  2501.0,  654.9, "Area",              "Mean area of the cell nucleus"),
    "smoothness_mean":         (0.053,  0.163,   0.096, "Smoothness",        "Local variation in radius lengths"),
    "compactness_mean":        (0.019,  0.345,   0.104, "Compactness",       "Perimeter squared / area minus 1.0"),
    "concavity_mean":          (0.0,    0.427,   0.089, "Concavity",         "Severity of concave contour portions"),
    "concave points_mean":     (0.0,    0.201,   0.049, "Concave Points",    "Number of concave portions of the contour"),
    "symmetry_mean":           (0.106,  0.304,   0.181, "Symmetry",          "Symmetry of the cell nucleus"),
    "fractal_dimension_mean":  (0.05,   0.097,   0.063, "Fractal Dimension", "Coastline approximation minus 1"),
    # ── SE ─────────────────────────────────────────────────────
    "radius_se":               (0.112,  2.873,   0.405, "Radius",            "Standard error of radius"),
    "texture_se":              (0.36,   4.885,   1.217, "Texture",           "Standard error of texture"),
    "perimeter_se":            (0.757,  21.98,   2.866, "Perimeter",         "Standard error of perimeter"),
    "area_se":                 (6.802,  542.2,   40.34, "Area",              "Standard error of area"),
    "smoothness_se":           (0.002,  0.031,   0.007, "Smoothness",        "Standard error of smoothness"),
    "compactness_se":          (0.002,  0.135,   0.025, "Compactness",       "Standard error of compactness"),
    "concavity_se":            (0.0,    0.396,   0.032, "Concavity",         "Standard error of concavity"),
    "concave points_se":       (0.0,    0.053,   0.012, "Concave Points",    "Standard error of concave points"),
    "symmetry_se":             (0.008,  0.079,   0.021, "Symmetry",          "Standard error of symmetry"),
    "fractal_dimension_se":    (0.001,  0.03,    0.004, "Fractal Dimension", "Standard error of fractal dimension"),
    # ── Worst ──────────────────────────────────────────────────
    "radius_worst":            (7.93,   36.04,   16.27, "Radius",            "Worst radius value"),
    "texture_worst":           (12.02,  49.54,   25.68, "Texture",           "Worst texture value"),
    "perimeter_worst":         (50.41,  251.2,   107.3, "Perimeter",         "Worst perimeter value"),
    "area_worst":              (185.2,  4254.0,  880.6, "Area",              "Worst area value"),
    "smoothness_worst":        (0.071,  0.223,   0.132, "Smoothness",        "Worst smoothness value"),
    "compactness_worst":       (0.027,  1.058,   0.254, "Compactness",       "Worst compactness value"),
    "concavity_worst":         (0.0,    1.252,   0.272, "Concavity",         "Worst concavity value"),
    "concave points_worst":    (0.0,    0.291,   0.115, "Concave Points",    "Worst concave points value"),
    "symmetry_worst":          (0.156,  0.664,   0.290, "Symmetry",          "Worst symmetry value"),
    "fractal_dimension_worst": (0.055,  0.208,   0.084, "Fractal Dimension", "Worst fractal dimension value"),
}

# Group only what actually survived the correlation filter
MEAN_FEATS  = [f for f in feature_cols if f.endswith("_mean")]
SE_FEATS    = [f for f in feature_cols if f.endswith("_se") or f == "concave points_se"]
WORST_FEATS = [f for f in feature_cols if f.endswith("_worst")]

# ==============================================================
# STYLES
# ==============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #050c1a;
    --surface:  #0a1628;
    --s2:       #0d1c35;
    --teal:     #00d4c8;
    --red:      #e8365d;
    --green:    #00e5a0;
    --amber:    #f5a623;
    --text:     #ddeaf7;
    --muted:    #3d5a7a;
    --border:   rgba(0,212,200,0.09);
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Exo 2', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: var(--sans); background: var(--bg); color: var(--text); }
.stApp { background: var(--bg); }

.stApp::before {
    content: ''; position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(0,212,200,0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,200,0.028) 1px, transparent 1px);
    background-size: 42px 42px;
    pointer-events: none; z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAV ── */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 52px; height: 58px;
    background: rgba(5,12,26,0.97); backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 200;
}
.nav-logo { font-size: 17px; font-weight: 900; letter-spacing: -0.5px; color: var(--text); }
.nav-logo em { font-style: normal; color: var(--teal); }
.nav-r { display: flex; align-items: center; gap: 20px; }
.nav-label { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); }
.nav-dot {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(0,229,160,0.07); border: 1px solid rgba(0,229,160,0.2);
    color: var(--green); font-family: var(--mono); font-size: 9px;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 6px 14px; border-radius: 100px;
}
.nav-dot::before {
    content: ''; width: 5px; height: 5px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 7px var(--green);
    animation: blink 2.5s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.15;} }

/* ── HERO LEFT ── */
.hero-l {
    padding: 76px 52px; display: flex; flex-direction: column;
    justify-content: center; position: relative;
    border-right: 1px solid var(--border); min-height: 500px;
}
.hero-l::before {
    content: ''; position: absolute; top: -80px; left: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(ellipse, rgba(0,212,200,0.06) 0%, transparent 65%);
    pointer-events: none; animation: blob 12s ease-in-out infinite alternate;
}
@keyframes blob { from{transform:translate(0,0);} to{transform:translate(20px,-15px);} }

.tag {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 3px;
    text-transform: uppercase; color: var(--teal); margin-bottom: 22px;
}
.tag::before {
    content: ''; width: 26px; height: 1px;
    background: var(--teal); box-shadow: 0 0 8px rgba(0,212,200,0.6);
}
.h1 {
    font-size: clamp(2.4rem,3.8vw,3.8rem); font-weight: 900;
    line-height: 1.05; letter-spacing: -2px; color: #fff; margin-bottom: 22px;
}
.h1 .t { color: var(--teal); text-shadow: 0 0 40px rgba(0,212,200,0.35); }
.desc {
    font-family: var(--mono); font-size: 12px; font-weight: 300;
    line-height: 1.9; color: var(--muted); max-width: 420px; margin-bottom: 36px;
}
.disclaimer {
    display: inline-flex; align-items: center; gap: 10px;
    background: rgba(245,166,35,0.05); border: 1px solid rgba(245,166,35,0.18);
    color: var(--amber); font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.5px; padding: 10px 18px; border-radius: 6px;
}

/* ── CONTENT ── */
.content { max-width: 1200px; margin: 0 auto; padding: 52px 52px 0; }
.sr { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
.sn { font-family:var(--mono); font-size:10px; color:var(--teal); opacity:0.6; letter-spacing:1px; }
.se-lbl { font-family:var(--mono); font-size:10px; letter-spacing:3px; text-transform:uppercase; color:var(--muted); }
.sl { flex:1; height:1px; background:linear-gradient(90deg, var(--border), transparent); }
.st { font-size:1.25rem; font-weight:800; letter-spacing:-0.4px; color:var(--text); margin-bottom:24px; }

/* ── EXPANDERS ── */
.stExpander {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
}
.stExpander summary {
    font-family: var(--mono) !important; font-size: 11px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: var(--teal) !important; padding: 14px 20px !important;
}

/* ── INPUTS ── */
.stNumberInput label {
    font-family: var(--mono) !important; font-size: 9px !important;
    font-weight: 500 !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; color: var(--muted) !important;
}
.stNumberInput > div > div {
    background: var(--bg) !important;
    border: 1px solid rgba(0,212,200,0.11) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: var(--mono) !important; font-size: 14px !important;
    transition: all 0.15s;
}
.stNumberInput > div > div:focus-within {
    border-color: rgba(0,212,200,0.42) !important;
    box-shadow: 0 0 0 3px rgba(0,212,200,0.07) !important;
    background: rgba(0,212,200,0.025) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: transparent !important; color: var(--teal) !important;
    font-family: var(--sans) !important; font-weight: 800 !important;
    font-size: 13px !important; letter-spacing: 1px !important;
    text-transform: uppercase !important; padding: 14px 42px !important;
    border: 1px solid rgba(0,212,200,0.4) !important;
    border-radius: 8px !important; width: auto !important;
    transition: all 0.18s !important;
    box-shadow: 0 0 24px rgba(0,212,200,0.06) !important;
}
.stButton > button:hover {
    background: rgba(0,212,200,0.07) !important; color: #fff !important;
    box-shadow: 0 0 40px rgba(0,212,200,0.18) !important;
    transform: translateY(-1px) !important;
}

/* ── RESULT ── */
.rtop {
    display: flex; align-items: center; justify-content: space-between;
    padding: 13px 28px; background: var(--s2);
    border-radius: 12px 12px 0 0;
    border: 1px solid var(--border); border-bottom: none;
}
.rtop-lbl { font-family: var(--mono); font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--muted); }
.rbadge { font-family: var(--mono); font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; padding: 5px 14px; border-radius: 100px; }
.rbadge.mal { background: rgba(232,54,93,0.1);  border: 1px solid rgba(232,54,93,0.28);  color: var(--red); }
.rbadge.ben { background: rgba(0,212,200,0.08); border: 1px solid rgba(0,212,200,0.22); color: var(--teal); }

.rl {
    padding: 40px 32px 40px 36px; background: var(--surface);
    border: 1px solid var(--border); border-top: none;
    border-right: none; border-radius: 0 0 0 12px; height: 100%;
}
.rv { font-size: clamp(1.6rem,3vw,2.4rem); font-weight: 900; letter-spacing: -0.8px; line-height: 1.1; margin-bottom: 12px; }
.rv.mal { color: var(--red);  text-shadow: 0 0 40px rgba(232,54,93,0.28); }
.rv.ben { color: var(--teal); text-shadow: 0 0 40px rgba(0,212,200,0.28); }
.rdesc { font-family: var(--mono); font-size: 11px; font-weight: 300; line-height: 1.85; color: var(--muted); margin-bottom: 32px; }
.dhr { height: 1px; background: var(--border); margin: 24px 0; }
.pl { font-family: var(--mono); font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.pn { font-size: clamp(3rem,5.5vw,4.8rem); font-weight: 900; letter-spacing: -3px; line-height: 1; }
.pn.mal { color: var(--red); }
.pn.ben { color: var(--teal); }
.pu { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 1px; margin-bottom: 28px; }
.nbox {
    background: rgba(0,212,200,0.03); border-left: 2px solid rgba(0,212,200,0.3);
    padding: 13px 16px; border-radius: 0 6px 6px 0;
    font-family: var(--mono); font-size: 11px; line-height: 1.75; color: var(--muted);
}
.wbox {
    background: rgba(245,166,35,0.04); border: 1px solid rgba(245,166,35,0.15);
    border-left: 2px solid var(--amber); padding: 11px 16px;
    border-radius: 0 6px 6px 0; font-family: var(--mono); font-size: 11px;
    color: var(--amber); margin-top: 12px; line-height: 1.6;
}
.rr {
    padding: 40px 36px 40px 32px; background: var(--surface);
    border: 1px solid var(--border); border-top: none;
    border-left: 1px solid rgba(0,212,200,0.07);
    border-radius: 0 0 12px 0; height: 100%;
    display: flex; flex-direction: column; justify-content: center;
}
.chips { display: flex; gap: 12px; margin-top: 16px; }
.chip {
    flex: 1; background: rgba(255,255,255,0.02);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px; text-align: center;
}
.clbl { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 7px; }
.cval { font-size: 1.25rem; font-weight: 800; }
.cval.t { color: var(--teal); }
.cval.g { color: var(--green); }

/* ── FOOTER ── */
.foot {
    max-width: 1200px; margin: 0 auto; padding: 28px 52px 48px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px; border-top: 1px solid var(--border);
}
.flogo { font-weight: 900; font-size: 14px; letter-spacing: -0.3px; color: var(--text); }
.flogo em { font-style: normal; color: var(--teal); }
.fnote { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); }

hr { display: none !important; }

/* ── RESPONSIVE — tablet (max 900px) ── */
@media (max-width: 900px) {
    .nav { padding: 0 20px; }
    .nav-label { display: none; }

    .hero-l {
        padding: 48px 24px;
        border-right: none;
        border-bottom: 1px solid var(--border);
        min-height: unset;
    }

    .content { padding: 36px 20px 0; }

    .rl {
        padding: 28px 20px;
        border-right: 1px solid var(--border) !important;
        border-radius: 0 0 12px 12px !important;
    }

    .rr {
        padding: 28px 20px;
        border-left: none !important;
        border-top: 1px solid var(--border) !important;
        border-radius: 0 0 12px 12px !important;
    }

    .foot { padding: 24px 20px 40px; flex-direction: column; text-align: center; }
}

/* ── RESPONSIVE — mobile (max 640px) ── */
@media (max-width: 640px) {
    .nav { padding: 0 16px; height: 52px; }
    .nav-logo { font-size: 15px; }
    .nav-dot { font-size: 8px; padding: 5px 10px; }

    .hero-l { padding: 40px 16px 36px; }
    .h1 { font-size: 2rem; letter-spacing: -1px; }
    .desc { font-size: 11px; }
    .disclaimer { font-size: 9px; padding: 9px 14px; }

    .content { padding: 28px 16px 0; }
    .st { font-size: 1.1rem; }

    /* Stack Streamlit columns on mobile */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        min-width: 100% !important;
        flex: none !important;
    }

    .rtop { padding: 11px 16px; }
    .rl {
        padding: 24px 16px;
        border-right: 1px solid var(--border) !important;
        border-radius: 0 !important;
    }
    .rr {
        padding: 24px 16px;
        border-left: none !important;
        border-top: 1px solid var(--border) !important;
        border-radius: 0 0 12px 12px !important;
    }
    .rv { font-size: 1.4rem; }
    .pn { font-size: 3rem; letter-spacing: -2px; }
    .chips { flex-direction: column; }

    .stButton > button {
        width: 100% !important;
        padding: 14px 20px !important;
    }

    .foot { padding: 20px 16px 36px; gap: 8px; }
    .fnote { text-align: center; }
}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# NAV
# ==============================================================
st.markdown("""
<div class="nav">
  <div class="nav-logo">Onco<em>Scan</em></div>
  <div class="nav-r">
    <span class="nav-label">Breast Cancer Risk Assessment</span>
    <span class="nav-dot">System Ready</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================
# HERO — two Streamlit columns (SVG via base64 to avoid escaping)
# ==============================================================
hero_l, hero_r = st.columns(2, gap="small")

with hero_l:
    st.markdown("""
    <div class="hero-l">
      <div class="tag">Clinical Decision Support</div>
      <h1 class="h1">Breast Cancer<br><span class="t">Risk Assessment</span></h1>
      <p class="desc">
        Enter tumor morphology measurements from fine needle aspirate
        imaging to receive an AI-assisted malignancy probability score
        based on nuclear morphology analysis.
      </p>
      <div class="disclaimer">
        &#9888; &ensp; For research and educational use only &mdash;
        not a substitute for clinical diagnosis
      </div>
    </div>
    """, unsafe_allow_html=True)

with hero_r:
    svg = """<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg" width="400" height="400">
  <defs>
    <radialGradient id="cg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00d4c8" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#00d4c8" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="ng" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#e8365d" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#e8365d" stop-opacity="0.04"/>
    </radialGradient>
    <filter id="g1">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="g2">
      <feGaussianBlur stdDeviation="8" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="400" height="400" fill="#07111f"/>
  <line x1="200" y1="0"   x2="200" y2="400" stroke="rgba(0,212,200,0.07)" stroke-width="0.5"/>
  <line x1="0"   y1="200" x2="400" y2="200" stroke="rgba(0,212,200,0.07)" stroke-width="0.5"/>
  <line x1="100" y1="0"   x2="100" y2="400" stroke="rgba(0,212,200,0.04)" stroke-width="0.3"/>
  <line x1="300" y1="0"   x2="300" y2="400" stroke="rgba(0,212,200,0.04)" stroke-width="0.3"/>
  <line x1="0"   y1="100" x2="400" y2="100" stroke="rgba(0,212,200,0.04)" stroke-width="0.3"/>
  <line x1="0"   y1="300" x2="400" y2="300" stroke="rgba(0,212,200,0.04)" stroke-width="0.3"/>
  <path d="M 20 20 L 20 48 M 20 20 L 48 20"     stroke="#00d4c8" stroke-width="1.8" fill="none" opacity="0.5"/>
  <path d="M 380 20 L 380 48 M 380 20 L 352 20"  stroke="#00d4c8" stroke-width="1.8" fill="none" opacity="0.5"/>
  <path d="M 20 380 L 20 352 M 20 380 L 48 380"  stroke="#00d4c8" stroke-width="1.8" fill="none" opacity="0.5"/>
  <path d="M 380 380 L 380 352 M 380 380 L 352 380" stroke="#00d4c8" stroke-width="1.8" fill="none" opacity="0.5"/>
  <circle cx="200" cy="200" r="170" fill="none" stroke="rgba(0,212,200,0.06)" stroke-width="1"/>
  <circle cx="200" cy="200" r="155" fill="none" stroke="rgba(0,212,200,0.08)" stroke-width="0.5" stroke-dasharray="5 15">
    <animateTransform attributeName="transform" type="rotate" from="0 200 200" to="360 200 200" dur="24s" repeatCount="indefinite"/>
  </circle>
  <circle cx="200" cy="200" r="132" fill="none" stroke="rgba(0,212,200,0.06)" stroke-width="0.5" stroke-dasharray="3 12">
    <animateTransform attributeName="transform" type="rotate" from="360 200 200" to="0 200 200" dur="18s" repeatCount="indefinite"/>
  </circle>
  <ellipse cx="200" cy="200" rx="112" ry="106" fill="url(#cg)" stroke="rgba(0,212,200,0.32)" stroke-width="1.5" filter="url(#g1)">
    <animate attributeName="rx" values="112;117;112" dur="5s" repeatCount="indefinite"/>
    <animate attributeName="ry" values="106;100;106" dur="5s" repeatCount="indefinite"/>
  </ellipse>
  <ellipse cx="200" cy="200" rx="112" ry="106" fill="none" stroke="rgba(0,212,200,0.09)" stroke-width="12" stroke-dasharray="3 10">
    <animate attributeName="rx" values="112;117;112" dur="5s" repeatCount="indefinite"/>
    <animate attributeName="ry" values="106;100;106" dur="5s" repeatCount="indefinite"/>
  </ellipse>
  <ellipse cx="205" cy="198" rx="48" ry="44" fill="url(#ng)" stroke="rgba(232,54,93,0.55)" stroke-width="1.5" filter="url(#g2)">
    <animate attributeName="rx" values="48;53;48" dur="3.5s" repeatCount="indefinite"/>
    <animate attributeName="ry" values="44;48;44" dur="3.5s" repeatCount="indefinite"/>
  </ellipse>
  <circle cx="210" cy="195" r="15" fill="rgba(232,54,93,0.2)" stroke="rgba(232,54,93,0.65)" stroke-width="1.2" filter="url(#g1)"/>
  <circle cx="210" cy="195" r="6"  fill="rgba(232,54,93,0.65)" filter="url(#g1)">
    <animate attributeName="r" values="6;8;6" dur="2.2s" repeatCount="indefinite"/>
  </circle>
  <ellipse cx="148" cy="178" rx="12" ry="6" fill="rgba(0,212,200,0.1)" stroke="rgba(0,212,200,0.35)" stroke-width="0.9" transform="rotate(-25 148 178)"/>
  <ellipse cx="255" cy="222" rx="14" ry="6" fill="rgba(0,212,200,0.1)" stroke="rgba(0,212,200,0.3)"  stroke-width="0.9" transform="rotate(20 255 222)"/>
  <ellipse cx="165" cy="232" rx="10" ry="5" fill="rgba(0,212,200,0.08)" stroke="rgba(0,212,200,0.25)" stroke-width="0.9" transform="rotate(-10 165 232)"/>
  <ellipse cx="240" cy="166" rx="11" ry="5" fill="rgba(0,212,200,0.08)" stroke="rgba(0,212,200,0.25)" stroke-width="0.9" transform="rotate(30 240 166)"/>
  <ellipse cx="178" cy="156" rx="9"  ry="4" fill="rgba(0,212,200,0.06)" stroke="rgba(0,212,200,0.2)"  stroke-width="0.9" transform="rotate(-40 178 156)"/>
  <line x1="200" y1="200" x2="312" y2="200" stroke="rgba(232,54,93,0.5)"  stroke-width="1"/>
  <circle cx="312" cy="200" r="3.5" fill="rgba(232,54,93,0.8)"/>
  <text x="238" y="193" font-family="JetBrains Mono,monospace" font-size="9" fill="rgba(232,54,93,0.65)"  letter-spacing="1">RADIUS</text>
  <line x1="200" y1="200" x2="200" y2="80"  stroke="rgba(0,212,200,0.3)"  stroke-width="1"/>
  <circle cx="200" cy="80"  r="3.5" fill="rgba(0,212,200,0.7)"/>
  <text x="208" y="118" font-family="JetBrains Mono,monospace" font-size="9" fill="rgba(0,212,200,0.55)" letter-spacing="1">PERIM</text>
  <circle cx="200" cy="200" r="3" fill="none" stroke="rgba(0,212,200,0.4)" stroke-width="1.2"/>
  <line x1="200" y1="200" x2="200" y2="32" stroke="rgba(0,212,200,0.6)" stroke-width="1.5" stroke-linecap="round" filter="url(#g1)">
    <animateTransform attributeName="transform" type="rotate" from="0 200 200" to="360 200 200" dur="5s" repeatCount="indefinite"/>
  </line>
  <text x="24"  y="22"  font-family="JetBrains Mono,monospace" font-size="9" fill="rgba(0,212,200,0.35)" letter-spacing="1">FNA</text>
  <text x="330" y="390" font-family="JetBrains Mono,monospace" font-size="9" fill="rgba(0,212,200,0.3)"  letter-spacing="1">SCAN</text>
  <text x="24"  y="390" font-family="JetBrains Mono,monospace" font-size="9" fill="rgba(0,212,200,0.25)" letter-spacing="1">x1000</text>
  <line x1="0" y1="0" x2="400" y2="0" stroke="rgba(0,212,200,0.55)" stroke-width="1.5">
    <animate attributeName="y1" values="0;400;0" dur="4s" repeatCount="indefinite"/>
    <animate attributeName="y2" values="0;400;0" dur="4s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.55;0.1;0.55" dur="4s" repeatCount="indefinite"/>
  </line>
</svg>"""
    svg_b64 = base64.b64encode(svg.encode()).decode()
    st.markdown(
        f'<div class="hero-svg-panel" style="background:linear-gradient(140deg,#060e1e,#09152a);'
        f'display:flex;align-items:center;justify-content:center;'
        f'min-height:500px;border-left:1px solid rgba(0,212,200,0.09);">'
        f'<img src="data:image/svg+xml;base64,{svg_b64}" width="400" height="400"/>'
        f'</div>'
        f'<style>'
        f'@media(max-width:900px){{.hero-svg-panel{{display:none!important;}}}}'
        f'</style>',
        unsafe_allow_html=True
    )

st.markdown('<div style="height:1px;background:rgba(0,212,200,0.09);"></div>',
            unsafe_allow_html=True)

# ==============================================================
# INPUTS
# ==============================================================
st.markdown('<div class="content">', unsafe_allow_html=True)

st.markdown("""
<div class="sr">
  <span class="sn">01</span>
  <span class="se-lbl">Input Parameters</span>
  <span class="sl"></span>
</div>
<div class="st">Tumor Morphology Parameters</div>
""", unsafe_allow_html=True)

values = {}

def render_group(feats, cols=2):
    """Render number inputs for a list of features in a grid."""
    columns = st.columns(cols, gap="large")
    for i, feat in enumerate(feats):
        mn, mx, default, label, tooltip = FEATURE_META[feat]
        columns[i % cols].number_input(
            label,
            min_value=float(mn),
            max_value=float(mx),
            value=float(default),
            format="%.4f",
            help=tooltip,
            key=feat,
            on_change=None
        )
        values[feat] = st.session_state[feat]

# Only render groups that have features
if MEAN_FEATS:
    with st.expander(f"📊  Mean Features  —  {len(MEAN_FEATS)} parameters", expanded=True):
        render_group(MEAN_FEATS)

if SE_FEATS:
    with st.expander(f"📉  Standard Error (SE) Features  —  {len(SE_FEATS)} parameters", expanded=False):
        render_group(SE_FEATS)

if WORST_FEATS:
    with st.expander(f"📈  Worst Features  —  {len(WORST_FEATS)} parameters", expanded=False):
        render_group(WORST_FEATS)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

st.markdown("""
<div class="sr">
  <span class="sn">02</span>
  <span class="se-lbl">Analysis</span>
  <span class="sl"></span>
</div>
<div class="st">Generate Assessment</div>
""", unsafe_allow_html=True)

# ==============================================================
# PREDICTION
# ==============================================================
if st.button("Run Analysis →"):

    # Collect values in the exact order model expects
    input_df = pd.DataFrame(
        [[st.session_state[f] for f in feature_cols]],
        columns=feature_cols
    )
    X_scaled    = scaler.transform(input_df)
    probability = model.predict_proba(X_scaled)[0][1]
    prediction  = model.predict(X_scaled)[0]
    risk        = probability * 100
    benign_pct  = 100 - risk
    conf        = "HIGH" if abs(risk - 50) > 20 else "MODERATE"

    if prediction == 1:
        cls, verdict = "mal", "Malignant Pattern Detected"
        desc         = "The assessed morphology is consistent with malignancy. Immediate specialist review is strongly advised."
        score_color  = "#e8365d"
        badge_label  = "High Risk"
    else:
        cls, verdict = "ben", "Benign Pattern Indicated"
        desc         = "The assessed morphology does not indicate malignancy. Routine clinical follow-up is recommended."
        score_color  = "#00d4c8"
        badge_label  = "Low Risk"

    # Result card top bar
    st.markdown(f"""
    <div class="rtop">
      <span class="rtop-lbl">Diagnostic Result</span>
      <span class="rbadge {cls}">{badge_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Two-column result body
    lc, rc = st.columns([1, 1], gap="medium")

    with lc:
        warn_html = ""
        if 45 <= risk <= 55:
            warn_html = """<div class="wbox">
                &#9888; Borderline result &mdash; indeterminate zone.
                Further clinical evaluation required.</div>"""
        st.markdown(f"""
        <div class="rl">
          <div class="rv {cls}">{verdict}</div>
          <p class="rdesc">{desc}</p>
          <div class="dhr"></div>
          <div class="pl">Malignancy Probability</div>
          <div class="pn {cls}">{risk:.1f}</div>
          <div class="pu">percent</div>
          {warn_html}
          <div class="nbox">
            This result is probabilistic and must be interpreted by a qualified
            clinician alongside full patient history and additional diagnostic data.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with rc:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%", "valueformat": ".1f",
                    "font": {"size": 42, "color": "#ddeaf7", "family": "Exo 2"}},
            title={"text": "RISK INDEX",
                   "font": {"size": 10, "color": "#3d5a7a", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "nticks": 6,
                         "tickcolor": "#0a1628",
                         "tickfont": {"color": "#1a3050", "size": 9,
                                      "family": "JetBrains Mono"}},
                "bar": {"color": score_color, "thickness": 0.2},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0,  35], "color": "rgba(0,212,200,0.06)"},
                    {"range": [35, 65], "color": "rgba(245,166,35,0.06)"},
                    {"range": [65,100], "color": "rgba(232,54,93,0.06)"},
                ],
                "threshold": {"line": {"color": score_color, "width": 2},
                              "thickness": 0.72, "value": risk}
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=270, margin=dict(l=24, r=24, t=56, b=8),
            font={"color": "#ddeaf7", "family": "Exo 2"}
        )
        st.markdown('<div class="rr">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div class="chips">
          <div class="chip">
            <div class="clbl">Benign Prob.</div>
            <div class="cval g">{benign_pct:.1f}%</div>
          </div>
          <div class="chip">
            <div class="clbl">Confidence</div>
            <div class="cval t">{conf}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================
# FOOTER
# ==============================================================
st.markdown("""
<div style="height:52px"></div>
<div class="foot">
  <span class="flogo">Onco<em>Scan</em></span>
  <span class="fnote">
    Research Prototype &nbsp;&middot;&nbsp;
    Educational Use Only &nbsp;&middot;&nbsp;
    Not for Clinical Deployment
  </span>
</div>
""", unsafe_allow_html=True)
