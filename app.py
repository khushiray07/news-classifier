import streamlit as st
import joblib
import json
import re
import time
import os

# Auto-train if model doesn't exist (for Streamlit Cloud)
if not os.path.exists('model/news_classifier.pkl'):
    import subprocess
    with st.spinner("🔄 First run: Training model... (2-3 mins)"):
        subprocess.run(['python', 'train.py'])

st.set_page_config(
    page_title="NewsLens — AI Classifier",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif;
    background: radial-gradient(120% 160% at 20% 20%, #132051 0%, #0c1024 35%, #080a18 65%);
    color: #f2f4ff;
}

div.block-container {
    max-width: 960px;
    padding-top: 1.25rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

.stApp::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background:
        radial-gradient(ellipse at 20% 20%, #7c3aed22 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, #06b6d422 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, #f59e0b11 0%, transparent 60%);
    animation: bgPulse 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes bgPulse {
    0%   { transform: translate(0,0) rotate(0deg); }
    100% { transform: translate(2%,2%) rotate(3deg); }
}
@keyframes fadeSlideDown {
    from { opacity:0; transform:translateY(-24px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity:0; transform:translateY(24px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes glowPulse {
    0%,100% { text-shadow:0 0 20px #7c3aed88,0 0 40px #7c3aed44; }
    50%      { text-shadow:0 0 30px #06b6d488,0 0 60px #06b6d444; }
}
@keyframes shimmer {
    0%   { background-position:-200% center; }
    100% { background-position: 200% center; }
}
@keyframes borderSpin {
    0%   { background-position:0% 50%; }
    50%  { background-position:100% 50%; }
    100% { background-position:0% 50%; }
}
@keyframes resultPop {
    0%  { opacity:0; transform:scale(0.88) translateY(20px); }
    70% { transform:scale(1.02) translateY(-4px); }
    100%{ opacity:1; transform:scale(1) translateY(0); }
}
@keyframes barGrow { from { width:0% !important; } }
@keyframes floatBadge {
    0%,100% { transform:translateY(0px); }
    50%      { transform:translateY(-6px); }
}
@keyframes countUp {
    from { opacity:0; transform:scale(0.5); }
    to   { opacity:1; transform:scale(1); }
}

.hero { text-align:center; padding:3.5rem 1rem 2rem; animation:fadeSlideDown 0.7s ease both; }
.hero-badge {
    display:inline-block; font-size:0.65rem; letter-spacing:0.25em;
    text-transform:uppercase; color:#c9d1ff; background:#202b5a;
    border:1px solid #304184; padding:0.35rem 1rem; border-radius:99px;
    margin-bottom:1.4rem; animation:floatBadge 3s ease-in-out infinite;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:clamp(3rem,8vw,5rem); font-weight:800;
    line-height:1;
    background:linear-gradient(135deg,#e2e0f0 0%,#a78bfa 40%,#06b6d4 70%,#f59e0b 100%);
    background-size:300% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:shimmer 4s linear infinite, glowPulse 4s ease-in-out infinite;
    margin-bottom:0.8rem;
}
.hero-sub { font-size:1rem; color:#9ca7d8; font-weight:400; letter-spacing:0.04em; animation:fadeSlideUp 0.9s 0.2s ease both; }

.stats-row { display:flex; gap:1rem; margin:2rem 0; animation:fadeSlideUp 0.8s 0.3s ease both; }
.stat-card {
    flex:1; background:linear-gradient(135deg,#10172f,#0f1b3f);
    border:1px solid #24315d; border-radius:14px; padding:1.1rem 0.5rem;
    text-align:center; position:relative; overflow:hidden; color:#e9edff;
    transition:transform 0.2s,border-color 0.2s;
}
.stat-card:hover { transform:translateY(-3px); border-color:#7c3aed66; }
.stat-card::before {
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,#7c3aed,#06b6d4,#f59e0b);
    background-size:200% auto; animation:borderSpin 3s linear infinite;
}
.stat-num {
    font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
    background:linear-gradient(135deg,#a78bfa,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.stat-lbl { font-size:0.6rem; letter-spacing:0.15em; text-transform:uppercase; color:#b6c0ec; margin-top:0.25rem; }

.input-wrap {
    background:linear-gradient(135deg,#101733,#0f1738);
    border-radius:18px; padding:1.8rem; border:1px solid #263363;
    margin:1.5rem 0; animation:fadeSlideUp 0.8s 0.4s ease both;
    box-shadow:0 20px 60px rgba(0,0,0,0.35);
}
.input-lbl { font-size:0.68rem; letter-spacing:0.2em; text-transform:uppercase; color:#c9d1ff; margin-bottom:0.8rem; font-weight:700; }

.stTextArea textarea {
    background:#0e1530 !important; border:1px solid #304184 !important;
    border-radius:12px !important; color:#f5f6ff !important;
    font-family:'Space Grotesk',sans-serif !important; font-size:1.05rem !important; padding:1.1rem !important;
    line-height:1.5 !important;
}
.stTextArea textarea:focus { border-color:#7c9bff !important; box-shadow:0 0 0 3px #30418455 !important; }
.stTextArea textarea::placeholder { color:#6f7bb2 !important; }

.stButton > button {
    background:linear-gradient(135deg,#5b8bff 0%, #4f46e5 45%, #06b6d4 100%) !important;
    background-size:180% auto !important; color:#fff !important;
    border:none !important; border-radius:12px !important;
    font-family:'Space Grotesk',sans-serif !important; font-size:0.95rem !important;
    font-weight:700 !important; letter-spacing:0.14em !important;
    text-transform:uppercase !important; padding:0.95rem !important;
    width:100% !important; cursor:pointer !important; box-shadow:0 12px 30px rgba(79,70,229,0.35);
    transition:all 0.3s ease !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 14px 38px rgba(91,139,255,0.45) !important;
}

.result-wrap { animation:resultPop 0.6s cubic-bezier(0.34,1.56,0.64,1) both; margin:1.5rem 0; }
.result-card { border-radius:20px; padding:2.2rem; position:relative; overflow:hidden; margin-bottom:1rem; box-shadow:0 20px 60px rgba(0,0,0,0.35); }
.result-card::before {
    content:''; position:absolute; inset:0; opacity:0.06;
    background-size:20px 20px;
    background-image:radial-gradient(circle,currentColor 1px,transparent 1px);
}
.r-world    { background:linear-gradient(135deg,#0a1628,#0d1f3c); border:1px solid #1e4080; }
.r-sports   { background:linear-gradient(135deg,#0a2010,#0d2a14); border:1px solid #1a5c28; }
.r-business { background:linear-gradient(135deg,#201008,#2a160a); border:1px solid #6b3a0a; }
.r-scitech  { background:linear-gradient(135deg,#150a28,#1e0f3a); border:1px solid #4a1a8c; }

.result-glow {
    position:absolute; top:-60px; right:-60px; width:200px; height:200px;
    border-radius:50%; filter:blur(60px); opacity:0.35;
    animation:glowPulse 3s ease-in-out infinite;
}
.r-world .result-glow    { background:#2176ff; }
.r-sports .result-glow   { background:#10b981; }
.r-business .result-glow { background:#f59e0b; }
.r-scitech .result-glow  { background:#8b5cf6; }

.result-top { display:flex; align-items:flex-start; gap:1.2rem; position:relative; z-index:1; }
.result-icon { font-size:3.5rem; line-height:1; animation:floatBadge 3s ease-in-out infinite; }
.result-eyebrow { font-size:0.65rem; letter-spacing:0.22em; text-transform:uppercase; opacity:0.5; margin-bottom:0.3rem; }
.result-name { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; line-height:1; margin-bottom:0.5rem; }
.r-world .result-name    { color:#60a5fa; }
.r-sports .result-name   { color:#34d399; }
.r-business .result-name { color:#fbbf24; }
.r-scitech .result-name  { color:#c084fc; }

.conf-pill {
    display:inline-flex; align-items:center; gap:0.4rem;
    padding:0.35rem 0.9rem; border-radius:99px; font-size:0.82rem; font-weight:700;
    animation:countUp 0.5s 0.3s ease both;
}
.r-world .conf-pill    { background:#2176ff22; color:#60a5fa; border:1px solid #2176ff44; }
.r-sports .conf-pill   { background:#10b98122; color:#34d399; border:1px solid #10b98144; }
.r-business .conf-pill { background:#f59e0b22; color:#fbbf24; border:1px solid #f59e0b44; }
.r-scitech .conf-pill  { background:#8b5cf622; color:#c084fc; border:1px solid #8b5cf644; }

.prob-card { background:linear-gradient(180deg,#0e132b,#0d1228); border:1px solid #263363; border-radius:16px; padding:1.5rem 1.8rem; animation:fadeSlideUp 0.5s 0.4s ease both; }
.prob-title { font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:#8fa0e0; margin-bottom:1.2rem; font-weight:700; }
.p-row { display:flex; align-items:center; gap:1rem; margin-bottom:1rem; }
.p-row:last-child { margin-bottom:0; }
.p-label { width:78px; font-size:0.8rem; color:#d3dbff; flex-shrink:0; }
.p-bar-bg { flex:1; height:10px; background:#141b36; border-radius:99px; overflow:hidden; }
.p-bar { height:100%; border-radius:99px; animation:barGrow 1s cubic-bezier(0.34,1.56,0.64,1) both; }
.bar-world    { background:linear-gradient(90deg,#1d4ed8,#60a5fa); box-shadow:0 0 8px #2176ff66; }
.bar-sports   { background:linear-gradient(90deg,#059669,#34d399); box-shadow:0 0 8px #10b98166; }
.bar-business { background:linear-gradient(90deg,#d97706,#fbbf24); box-shadow:0 0 8px #f59e0b66; }
.bar-scitech  { background:linear-gradient(90deg,#7c3aed,#c084fc); box-shadow:0 0 8px #8b5cf666; }
.p-pct { width:40px; text-align:right; font-size:0.8rem; font-weight:700; color:#b4bce5; flex-shrink:0; }

.empty-state { text-align:center; padding:4rem 1rem; animation:fadeSlideUp 0.6s ease both; }
.empty-icon { font-size:4rem; margin-bottom:1rem; animation:floatBadge 3s ease-in-out infinite; display:block; }
.empty-text { font-size:0.75rem; letter-spacing:0.18em; text-transform:uppercase; color:#3c4878; }

section[data-testid="stSidebar"] { background:#0b1022 !important; border-right:1px solid #14142a !important; }
section[data-testid="stSidebar"] .stButton > button {
    background:#121a33 !important; color:#c3cbff !important;
    border:1px solid #263363 !important; font-size:0.78rem !important;
    text-transform:none !important; letter-spacing:0.01em !important;
    text-align:left !important; padding:0.6rem 0.85rem !important;
    border-radius:8px !important; animation:none !important; background-size:unset !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color:#7c9bff !important; color:#e7eaff !important;
    background:#162247 !important; transform:translateX(3px) !important; box-shadow:none !important;
}
.stSpinner > div { border-top-color:#7c3aed !important; }

@media (max-width: 640px) {
    .hero { padding:2.4rem 0.6rem 1.2rem; }
    .hero-title { font-size:clamp(2.4rem,10vw,3.4rem); }
    .stats-row { flex-direction:column; }
    .stat-card { padding:0.9rem; }
    .result-name { font-size:2.2rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists('model/news_classifier.pkl'):
        import subprocess
        subprocess.run(['python', 'train.py'], check=True)
    pipeline = joblib.load('model/news_classifier.pkl')
    with open('model/label_map.json', 'r') as f:
        raw = json.load(f)
    return pipeline, {int(k): v for k, v in raw.items()}
pipeline, label_map = load_model()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

META = {
    "World":    {"emoji": "🌍", "css": "world",    "bar": "bar-world"},
    "Sports":   {"emoji": "⚽", "css": "sports",   "bar": "bar-sports"},
    "Business": {"emoji": "💼", "css": "business", "bar": "bar-business"},
    "Sci/Tech": {"emoji": "🔬", "css": "scitech",  "bar": "bar-scitech"},
}

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                    background:linear-gradient(135deg,#a78bfa,#06b6d4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin-bottom:0.3rem;'>NewsLens</div>
        <div style='font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;
                    color:#9ca7d8;margin-bottom:1.5rem;'>AI · NLP · Classifier</div>
        <div style='font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;
                    color:#c3cbff;margin-bottom:0.8rem;font-weight:700;'>Try Headlines</div>
    </div>
    """, unsafe_allow_html=True)

    for ex in [
        "Tesla stock surges after record quarterly earnings",
        "Manchester City wins Premier League title",
        "NASA Artemis mission returns Moon samples",
        "UN Security Council meets over Gaza ceasefire",
        "Google releases Gemini 2.0 with multimodal AI",
        "Fed cuts interest rates by 25 basis points",
        "India wins Cricket World Cup in dramatic final",
        "SpaceX successfully lands Starship booster",
    ]:
        if st.button(ex, key=ex):
            st.session_state['input_text'] = ex

    st.markdown("<hr style='border:none;border-top:1px solid #14142a;margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.65rem;letter-spacing:0.18em;text-transform:uppercase;color:#c3cbff;margin-bottom:0.8rem;font-weight:700;'>Model Details</div>
    <div style='font-size:0.75rem;color:#adb9ec;line-height:2.1;'>
        <span style='color:#9ca7d8'>Dataset</span>&nbsp;&nbsp;&nbsp; AG News<br>
        <span style='color:#9ca7d8'>Samples</span>&nbsp;&nbsp;&nbsp; 120,000<br>
        <span style='color:#9ca7d8'>Model</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Logistic Reg.<br>
        <span style='color:#9ca7d8'>Vectors</span>&nbsp;&nbsp;&nbsp; TF-IDF 50K<br>
        <span style='color:#9ca7d8'>Accuracy</span>&nbsp;&nbsp; ~90%
    </div>
    """, unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Powered by NLP · TF-IDF · ML</div>
    <h1 class="hero-title">NewsLens</h1>
    <p class="hero-sub">Drop any headline. Get the category. Instantly.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-card"><div class="stat-num">120K</div><div class="stat-lbl">Articles</div></div>
    <div class="stat-card"><div class="stat-num">~90%</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-card"><div class="stat-num">4</div><div class="stat-lbl">Categories</div></div>
    <div class="stat-card"><div class="stat-num">&lt;1s</div><div class="stat-lbl">Speed</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
st.markdown('<div class="input-lbl">⚡ Enter Your Headline</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="", value=st.session_state.get('input_text', ''),
    placeholder="e.g. Apple unveils new MacBook with M4 chip and AI features...",
    height=120, label_visibility="collapsed"
)
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    classify_btn = st.button("⚡ CLASSIFY NOW", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Result
if classify_btn:
    if not user_input.strip():
        st.warning("⚠️  Please enter a news headline first!")
    else:
        with st.spinner("Analyzing with AI..."):
            time.sleep(0.4)
            cleaned  = clean_text(user_input)
            pred     = pipeline.predict([cleaned])[0]
            probs    = pipeline.predict_proba([cleaned])[0]
            conf     = probs.max() * 100
            category = label_map[int(pred)]
            meta     = META[category]

        st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card r-{meta['css']}">
            <div class="result-glow"></div>
            <div class="result-top">
                <div class="result-icon">{meta['emoji']}</div>
                <div>
                    <div class="result-eyebrow">Classified As</div>
                    <div class="result-name">{category}</div>
                    <div class="conf-pill">✦ {conf:.1f}% confident</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        bars = '<div class="prob-card"><div class="prob-title">Confidence Breakdown</div>'
        for i, prob in enumerate(probs):
            cat = label_map[int(i)]
            m   = META[cat]
            bars += f"""
            <div class="p-row">
                <div class="p-label">{m['emoji']} {cat}</div>
                <div class="p-bar-bg"><div class="p-bar {m['bar']}" style="width:{prob*100:.1f}%"></div></div>
                <div class="p-pct">{prob*100:.1f}%</div>
            </div>"""
        bars += '</div>'
        st.markdown(bars, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <span class="empty-icon">🔭</span>
        <div class="empty-text">Waiting for your headline</div>
    </div>
    """, unsafe_allow_html=True)
