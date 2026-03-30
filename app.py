"""
app.py — Multi-Modal ML Data Annotation Platform (ALL-IN-ONE)
=============================================================
Contains all modules in a single file:
  • Utils          (DB helpers, AI suggestions)
  • Image Module   (bounding box annotation)
  • Text Module    (sentiment annotation)
  • Audio Module   (emotion annotation)
  • Dashboard      (analytics & SLA tracking)
  • Home Page

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

# 0. IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import glob
import re
import time
import random
from datetime import datetime, timedelta
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE CONFIG 
st.set_page_config(
    page_title="ML Annotation Platform",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. GLOBAL CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-card:      #1c2128;
    --border:       #30363d;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --blue:         #4B8BFF;
    --green:        #3fb950;
    --red:          #f85149;
    --orange:       #d29922;
    --purple:       #bc8cff;
    --pink:         #E91E63;
    --yellow:       #FFD700;
}

.stApp { background-color: var(--bg-primary) !important; font-family:'Inter',sans-serif; }

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

h1,h2,h3,h4,h5,h6 { color:var(--text-primary)!important; font-family:'Inter',sans-serif!important; }
p,span,label,div   { color:var(--text-primary)!important; }

.stTextInput input,.stTextArea textarea {
    background:var(--bg-card)!important; border:1px solid var(--border)!important;
    color:var(--text-primary)!important; border-radius:8px!important;
}
.stButton>button[kind="primary"] {
    background:linear-gradient(135deg,#4B8BFF,#9B59B6)!important;
    color:#fff!important; border:none!important; border-radius:8px!important;
    font-weight:600!important; transition:all .2s ease!important;
}
.stButton>button[kind="primary"]:hover {
    opacity:.9!important; transform:translateY(-1px)!important;
    box-shadow:0 4px 12px rgba(75,139,255,.3)!important;
}
.stButton>button[kind="secondary"] {
    background:var(--bg-card)!important; color:var(--text-primary)!important;
    border:1px solid var(--border)!important; border-radius:8px!important;
}
.stButton>button[kind="secondary"]:hover { border-color:var(--blue)!important; }

[data-testid="stMetric"] {
    background:var(--bg-card)!important; border:1px solid var(--border)!important;
    border-radius:10px!important; padding:.8rem!important;
}
[data-testid="stMetricValue"] { color:var(--text-primary)!important; font-weight:700!important; }

.stDataFrame { background:var(--bg-card)!important; border:1px solid var(--border)!important; border-radius:10px!important; }
.stProgress>div>div { background:linear-gradient(90deg,#4B8BFF,#9B59B6)!important; border-radius:4px!important; }
.stAlert { border-radius:8px!important; }
.streamlit-expanderHeader { background:var(--bg-card)!important; border:1px solid var(--border)!important; border-radius:8px!important; }
hr { border-color:var(--border)!important; margin:1rem 0!important; }
.stRadio label { color:var(--text-muted)!important; }
canvas { border-radius:8px!important; }
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg-primary)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#484f58}
</style>
""", unsafe_allow_html=True)

# 3. CONSTANTS & CONFIG
DB_PATH = "annotations/annotations.db"

LABEL_COLORS = {
    "person": "#FF4B4B", "car": "#4B8BFF", "dog": "#4BFF8B",
    "cat": "#FFD700", "bird": "#FF8C00", "bicycle": "#9B59B6",
    "chair": "#1ABC9C", "bottle": "#E67E22", "other": "#95A5A6",
}

SENTIMENT_CONFIG = {
    "positive": {"icon": "😊", "color": "#4CAF50", "bg": "#1a2e1a"},
    "negative": {"icon": "😞", "color": "#F44336", "bg": "#2e1a1a"},
    "neutral":  {"icon": "😐", "color": "#FF9800", "bg": "#2e251a"},
}

EMOTIONS = {
    "happy":     {"icon": "😄", "color": "#FFD700"},
    "sad":       {"icon": "😢", "color": "#4169E1"},
    "angry":     {"icon": "😠", "color": "#DC143C"},
    "fearful":   {"icon": "😨", "color": "#8B008B"},
    "neutral":   {"icon": "😐", "color": "#808080"},
    "calm":      {"icon": "😌", "color": "#20B2AA"},
    "disgusted": {"icon": "🤢", "color": "#556B2F"},
    "surprised": {"icon": "😲", "color": "#FF8C00"},
}

# 4. UTILS — DATABASE & AI SUGGESTIONS
def _ensure_dirs():
    for d in ["data/images", "data/text", "data/audio", "annotations", "exports"]:
        os.makedirs(d, exist_ok=True)

def _get_conn():
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    return conn

def _ensure_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_file TEXT NOT NULL,
            x_min REAL, y_min REAL, x_max REAL, y_max REAL,
            label TEXT,
            annotator TEXT DEFAULT 'anonymous',
            confidence REAL DEFAULT 1.0,
            time_taken REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS text_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_id INTEGER, review TEXT,
            ground_truth_label TEXT, annotator_label TEXT,
            is_correct INTEGER,
            annotator TEXT DEFAULT 'anonymous',
            confidence REAL DEFAULT 1.0,
            time_taken REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audio_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file TEXT NOT NULL,
            ground_truth_emotion TEXT, annotator_emotion TEXT,
            transcription TEXT, is_correct INTEGER,
            annotator TEXT DEFAULT 'anonymous',
            confidence REAL DEFAULT 1.0,
            time_taken REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    conn.commit()

def _load_table(table):
    try:
        conn = _get_conn()
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def save_image_annotation(image_file, x_min, y_min, x_max, y_max,
                           label, annotator, confidence, time_taken):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO image_annotations (image_file,x_min,y_min,x_max,y_max,label,annotator,confidence,time_taken) VALUES(?,?,?,?,?,?,?,?,?)",
        (image_file, x_min, y_min, x_max, y_max, label, annotator, confidence, time_taken))
    conn.commit(); conn.close()

def save_text_annotation(text_id, review, ground_truth, annotator_label,
                          is_correct, annotator, confidence, time_taken):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO text_annotations (text_id,review,ground_truth_label,annotator_label,is_correct,annotator,confidence,time_taken) VALUES(?,?,?,?,?,?,?,?)",
        (text_id, review[:500], ground_truth, annotator_label, is_correct, annotator, confidence, time_taken))
    conn.commit(); conn.close()

def save_audio_annotation(audio_file, ground_truth_emotion, annotator_emotion,
                           transcription, is_correct, annotator, confidence, time_taken):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO audio_annotations (audio_file,ground_truth_emotion,annotator_emotion,transcription,is_correct,annotator,confidence,time_taken) VALUES(?,?,?,?,?,?,?,?)",
        (audio_file, ground_truth_emotion, annotator_emotion, transcription, is_correct, annotator, confidence, time_taken))
    conn.commit(); conn.close()

def get_ai_sentiment_suggestion(text: str):
    """Returns (label, confidence). Tries transformers, falls back to rule-based."""
    text_clean = re.sub(r"<[^>]+>", "", text).strip().lower()
    try:
        from transformers import pipeline
        import warnings; warnings.filterwarnings("ignore")
        _m = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english",
                      truncation=True, max_length=512)
        r = _m(text[:512])[0]
        label = "positive" if "pos" in r["label"].lower() else "negative"
        return label, r["score"]
    except Exception:
        pass
    pos_w = {"great","excellent","amazing","fantastic","wonderful","brilliant","love",
              "best","superb","outstanding","enjoyed","recommend","perfect","incredible"}
    neg_w = {"terrible","awful","horrible","worst","bad","boring","waste","dreadful",
              "poor","disappointing","rubbish","pathetic","dull","stupid","garbage"}
    words = set(text_clean.split())
    p = len(words & pos_w); n = len(words & neg_w)
    if p > n:   return "positive", min(0.6 + p * 0.05, 0.95)
    elif n > p: return "negative", min(0.6 + n * 0.05, 0.95)
    else:       return "neutral",  0.55

def get_ai_image_suggestion(image_path: str) -> str:
    """Returns a suggested label for an image."""
    labels = list(LABEL_COLORS.keys())
    try:
        from PIL import Image as PILImage
        from transformers import CLIPProcessor, CLIPModel
        import torch
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        img   = PILImage.open(image_path).convert("RGB")
        inputs = proc(text=[f"a photo of a {l}" for l in labels],
                      images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
        idx = out.logits_per_image.softmax(dim=1)[0].argmax().item()
        return labels[idx]
    except Exception:
        pass
    fname = os.path.basename(image_path).lower()
    for lbl in labels[:-1]:
        if lbl in fname: return lbl
    return random.choice(labels)

# 5. IMAGE ANNOTATION MODULE
def _generate_sample_images():
    os.makedirs("data/images", exist_ok=True)
    names = ["car_sample","person_sample","dog_sample","bird_sample","cat_sample"]
    for i, name in enumerate(names):
        arr = np.full((400, 500, 3), 200, dtype=np.uint8)
        # Simple gradient
        for y in range(400):
            for x in range(500):
                arr[y, x] = [(150 + i*20 + y//4) % 256,
                             (180 + i*15 + x//4) % 256,
                             (120 + i*25)        % 256]
        PILImg = Image.fromarray(arr)
        PILImg.save(f"data/images/{name}.jpg")

def _draw_bbox_on_image(img: Image.Image, x_min, y_min, x_max, y_max,
                        label: str, color_hex: str = "#FF4B4B") -> Image.Image:
    """Draw a bounding box + label on a PIL image and return it."""
    import io
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return img

    out  = img.copy()
    draw = ImageDraw.Draw(out)

    # Convert hex → RGB
    h  = color_hex.lstrip("#")
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # Draw thick rectangle
    lw = max(3, img.width // 200)
    for offset in range(lw):
        draw.rectangle(
            [x_min - offset, y_min - offset, x_max + offset, y_max + offset],
            outline=rgb
        )

    # Label background + text
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, img.width // 50))
    except Exception:
        font = ImageFont.load_default()

    text    = f" {label} "
    bbox_t  = draw.textbbox((x_min, y_min - 24), text, font=font)
    draw.rectangle(bbox_t, fill=rgb)
    draw.text((x_min, y_min - 24), text, fill="white", font=font)

    return out


def show_image_module():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                padding:1.2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;
                border-left:4px solid #4B8BFF'>
        <h2 style='color:#fff;margin:0'>🖼️ Image Annotation</h2>
        <p style='color:#8892b0;margin:.2rem 0 0'>Draw bounding boxes and label objects</p>
    </div>
    """, unsafe_allow_html=True)

    # Check canvas availability
    CANVAS_OK = False
    try:
        from streamlit_drawable_canvas import st_canvas
        import streamlit.elements.image as _st_img_mod
        if hasattr(_st_img_mod, "image_to_url"):
            CANVAS_OK = True
        else:
            CANVAS_OK = False
    except Exception:
        CANVAS_OK = False

    col1, col2 = st.columns([2, 1])
    with col1:
        annotator = st.text_input("Annotator Name", value="Annotator_1", key="img_annotator")
    with col2:
        st.metric("Session Annotations", st.session_state.get("img_count", 0))

    if not CANVAS_OK:
        st.info(
            "ℹ️ **Interactive canvas unavailable** — your Streamlit version is not compatible with "
            "`streamlit-drawable-canvas`.\n\n"
            "**Quick fix (choose one):**\n"
            "```\n"
            "pip install streamlit==1.32.0 streamlit-drawable-canvas\n"
            "```\n"
            "The manual slider annotation below works perfectly without the canvas."
        )

    st.divider()

    # Find images
    image_files = list(dict.fromkeys(
        glob.glob("data/images/**/*.jpg",  recursive=True) +
        glob.glob("data/images/**/*.jpeg", recursive=True) +
        glob.glob("data/images/**/*.png",  recursive=True) +
        glob.glob("data/images/*.jpg") +
        glob.glob("data/images/*.png")
    ))

    if not image_files:
        st.warning("No images found in `data/images/`.")
        if st.button("Generate 5 Sample Images"):
            _generate_sample_images()
            st.rerun()
        return

    total = len(image_files)
    if "img_idx" not in st.session_state:
        st.session_state.img_idx = 0
    idx = st.session_state.img_idx % total

    # Navigation
    n1, n2, n3, n4 = st.columns([1, 1, 3, 1])
    with n1:
        if st.button("⬅️ Prev") and idx > 0:
            st.session_state.img_idx -= 1; st.rerun()
    with n2:
        if st.button("Next ➡️") and idx < total - 1:
            st.session_state.img_idx += 1; st.rerun()
    with n3:
        st.progress((idx + 1) / total, text=f"Image {idx+1} of {total}")
    with n4:
        if st.button("🔀 Random"):
            st.session_state.img_idx = np.random.randint(0, total); st.rerun()

    img_path  = image_files[idx]
    img       = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    start_time = time.time()

    col_l, col_t = st.columns([2, 1])
    with col_l:
        label = st.selectbox("Object Label", list(LABEL_COLORS.keys()))
    with col_t:
        st.markdown(f"""
        <div style='background:{LABEL_COLORS.get(label,"#999")};border-radius:6px;
                    padding:.5rem;text-align:center;font-weight:700;color:#fff;margin-top:.5rem'>
            {label.upper()}
        </div>""", unsafe_allow_html=True)

    if st.button("AI Label Suggestion"):
        with st.spinner("Analyzing image..."):
            suggestion = get_ai_image_suggestion(img_path)
        st.info(f" AI suggests: **{suggestion}**")

    # PATH A - streamlit-drawable-canvas
    if CANVAS_OK:
        from streamlit_drawable_canvas import st_canvas
        canvas_w, canvas_h = 640, 480
        img_resized = img.resize((canvas_w, canvas_h))

        st.markdown(f"**{os.path.basename(img_path)}** — Draw a rectangle around the object")

        canvas_result = st_canvas(
            fill_color="rgba(75,139,255,0.15)",
            stroke_width=3,
            stroke_color=LABEL_COLORS.get(label, "#FF4B4B"),
            background_image=img_resized,
            update_streamlit=True,
            height=canvas_h, width=canvas_w,
            drawing_mode="rect",
            key=f"canvas_{idx}",
        )

        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            objects    = canvas_result.json_data["objects"]
            time_taken = round(time.time() - start_time, 2)
            sx, sy     = orig_w / canvas_w, orig_h / canvas_h

            with st.expander(f"{len(objects)} bounding box(es) detected", expanded=True):
                for obj in objects:
                    if obj.get("type") == "rect":
                        x_min = int(obj.get("left", 0) * sx)
                        y_min = int(obj.get("top",  0) * sy)
                        x_max = int((obj.get("left", 0) + obj.get("width",  0)) * sx)
                        y_max = int((obj.get("top",  0) + obj.get("height", 0)) * sy)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("X Min", x_min); c2.metric("Y Min", y_min)
                        c3.metric("X Max", x_max); c4.metric("Y Max", y_max)

            conf = st.slider("Confidence", 0.0, 1.0, 0.9, 0.05, key=f"conf_img_{idx}")
            if st.button("💾 Save Annotation", type="primary", use_container_width=True):
                for obj in objects:
                    if obj.get("type") == "rect":
                        save_image_annotation(
                            image_file=os.path.basename(img_path),
                            x_min=int(obj.get("left", 0) * sx),
                            y_min=int(obj.get("top",  0) * sy),
                            x_max=int((obj.get("left", 0) + obj.get("width",  0)) * sx),
                            y_max=int((obj.get("top",  0) + obj.get("height", 0)) * sy),
                            label=label, annotator=annotator,
                            confidence=conf, time_taken=time_taken,
                        )
                st.session_state.img_count = st.session_state.get("img_count", 0) + 1
                st.success(f"✅ Saved {len(objects)} box(es) as **{label}**")
                if idx < total - 1:
                    st.session_state.img_idx += 1; st.rerun()
        else:
            st.info("👆 Draw a rectangle on the image above, then save")

    # PATH B — Slider-based annotation
    else:
        st.markdown(f"""
        <div style='background:#1c2128;border:1px solid #30363d;border-radius:10px;
                    padding:1rem;margin:.5rem 0'>
            <span style='color:#8b949e;font-size:.85rem'>
                🖼️ <b style='color:#e6edf3'>{os.path.basename(img_path)}</b>
                &nbsp;|&nbsp; {orig_w} × {orig_h} px
                &nbsp;|&nbsp; Use sliders below to set bounding box coordinates
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Display image at fixed width
        display_w = 700
        display_h = int(orig_h * display_w / orig_w)
        img_display = img.resize((display_w, display_h))

        # Initialise bbox state per image
        bbox_key = f"bbox_{idx}"
        if bbox_key not in st.session_state:
            # Smart default: centre 50% of image
            st.session_state[bbox_key] = {
                "x_min": orig_w  // 4,
                "y_min": orig_h  // 4,
                "x_max": orig_w  * 3 // 4,
                "y_max": orig_h  * 3 // 4,
            }

        # Sliders
        st.markdown("#### Set Bounding Box Coordinates")
        sl1, sl2 = st.columns(2)
        with sl1:
            x_min = st.slider("X Min (left edge)",   0, orig_w - 2,
                               st.session_state[bbox_key]["x_min"], key=f"xmin_{idx}")
            y_min = st.slider("Y Min (top edge)",    0, orig_h - 2,
                               st.session_state[bbox_key]["y_min"], key=f"ymin_{idx}")
        with sl2:
            x_max = st.slider("X Max (right edge)",  x_min + 1, orig_w,
                               max(st.session_state[bbox_key]["x_max"], x_min + 1), key=f"xmax_{idx}")
            y_max = st.slider("Y Max (bottom edge)", y_min + 1, orig_h,
                               max(st.session_state[bbox_key]["y_max"], y_min + 1), key=f"ymax_{idx}")

        # Update state
        st.session_state[bbox_key] = {"x_min": x_min, "y_min": y_min,
                                      "x_max": x_max, "y_max": y_max}

        # Live preview with bbox drawn
        # Scale coords to display size
        sx = display_w / orig_w
        sy = display_h / orig_h
        preview = _draw_bbox_on_image(
            img_display,
            int(x_min * sx), int(y_min * sy),
            int(x_max * sx), int(y_max * sy),
            label, LABEL_COLORS.get(label, "#FF4B4B")
        )
        st.image(preview, caption=f"Preview: [{x_min}, {y_min}] → [{x_max}, {y_max}]  |  label: {label}",
                 use_container_width=True)

        # Coordinate readout
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("X Min", x_min)
        m2.metric("Y Min", y_min)
        m3.metric("X Max", x_max)
        m4.metric("Y Max", y_max)
        m5.metric("Box Area", f"{(x_max-x_min)*(y_max-y_min):,} px²")

        # Quick-adjust buttons 
        st.markdown("**Quick Adjust:**")
        qa1, qa2, qa3, qa4, qa5 = st.columns(5)
        step = max(5, min(orig_w, orig_h) // 40)
        with qa1:
            if st.button("⬅ Shrink W"):
                st.session_state[bbox_key]["x_max"] = max(x_min+1, x_max - step)
                st.rerun()
        with qa2:
            if st.button("➡ Grow W"):
                st.session_state[bbox_key]["x_max"] = min(orig_w, x_max + step)
                st.rerun()
        with qa3:
            if st.button("⬆ Shrink H"):
                st.session_state[bbox_key]["y_max"] = max(y_min+1, y_max - step)
                st.rerun()
        with qa4:
            if st.button("⬇ Grow H"):
                st.session_state[bbox_key]["y_max"] = min(orig_h, y_max + step)
                st.rerun()
        with qa5:
            if st.button("Reset"):
                st.session_state[bbox_key] = {"x_min": orig_w//4, "y_min": orig_h//4,
                                               "x_max": orig_w*3//4, "y_max": orig_h*3//4}
                st.rerun()

        st.markdown("")
        conf = st.slider("Confidence Level", 0.0, 1.0, 0.9, 0.05, key=f"conf_img_{idx}")

        sv1, sv2 = st.columns([3, 1])
        with sv1:
            if st.button("💾 Save Annotation", type="primary", use_container_width=True):
                save_image_annotation(
                    image_file=os.path.basename(img_path),
                    x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                    label=label, annotator=annotator,
                    confidence=conf,
                    time_taken=round(time.time() - start_time, 2),
                )
                st.session_state.img_count = st.session_state.get("img_count", 0) + 1
                st.success(f"✅ Saved — **{label}** [{x_min},{y_min}] → [{x_max},{y_max}]")
                if idx < total - 1:
                    st.session_state.img_idx += 1; st.rerun()
        with sv2:
            if st.button("⏭️ Skip"):
                if idx < total - 1:
                    st.session_state.img_idx += 1; st.rerun()

# 6. TEXT ANNOTATION MODULE
def _create_sample_text_data():
    os.makedirs("data/text", exist_ok=True)
    data = [
        ("Absolutely brilliant film. The acting was superb and the story gripping.", "positive"),
        ("Dreadful waste of time. I walked out after 30 minutes.", "negative"),
        ("Decent enough movie. Some good scenes but nothing extraordinary.", "neutral"),
        ("One of the best films I have seen in years. Highly recommend!", "positive"),
        ("Poorly written, terrible acting, and a nonsensical plot.", "negative"),
        ("It was okay. Had its moments but also quite slow at times.", "neutral"),
        ("A masterpiece of storytelling and visual art.", "positive"),
        ("Complete garbage. I cannot believe this got made.", "negative"),
        ("An average movie with some charm but also many flaws.", "neutral"),
        ("Stunning visuals and emotionally resonant performances.", "positive"),
        ("The plot made no sense and the characters were unlikeable.", "negative"),
        ("Entertaining enough for a rainy afternoon but nothing more.", "neutral"),
        ("Genuinely moved me to tears. A truly special film.", "positive"),
        ("I regret watching this. Two hours I will never get back.", "negative"),
        ("Some inspired moments but ultimately disappointing overall.", "neutral"),
    ] * 4
    df = pd.DataFrame(data, columns=["review", "ground_truth"])
    df.to_csv("data/text/text_for_annotation.csv", index=False)
    return df

def show_text_module():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a2e1a,#1a1a2e);
                padding:1.2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;
                border-left:4px solid #4CAF50'>
        <h2 style='color:#fff;margin:0'>📝 Text Annotation</h2>
        <p style='color:#8892b0;margin:.2rem 0 0'>Classify sentiment of movie reviews</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        annotator = st.text_input("Annotator Name", value="Annotator_1", key="txt_annotator")
    with c2:
        st.metric("Done", st.session_state.get("txt_count", 0))
    with c3:
        correct    = st.session_state.get("txt_correct", 0)
        done_total = max(st.session_state.get("txt_count", 1), 1)
        st.metric("Accuracy", f"{correct/done_total*100:.0f}%")

    st.divider()

    # Load data
    df = None
    for p in ["data/text/text_for_annotation.csv",
              "data/text/imdb_sample.csv",
              "data/text/IMDB Dataset.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p); break

    if df is None:
        df = _create_sample_text_data()

    df.columns = [c.lower().strip() for c in df.columns]
    r_col = next((c for c in df.columns if any(k in c for k in ("review","text","sentence"))), df.columns[0])
    l_col = next((c for c in df.columns if any(k in c for k in ("sentiment","label","ground","truth"))), None)
    df = df.rename(columns={r_col: "review"})
    df["ground_truth"] = df[l_col].astype(str).str.lower().str.strip() if l_col else "unknown"
    df["review"] = df["review"].astype(str).str.strip()
    df["review"] = df["review"].str.replace(r"<[^>]+>", "", regex=True)

    if "annotated_ids" not in st.session_state:
        st.session_state.annotated_ids = set()

    remaining = df[~df.index.isin(st.session_state.annotated_ids)]

    if remaining.empty:
        st.balloons()
        st.success("🎉 All reviews annotated! Check the Dashboard.")
        if st.button("🔄 Reset"):
            for k in ("annotated_ids","txt_count","txt_correct","txt_idx","_sel"):
                st.session_state.pop(k, None)
            st.rerun()
        return

    if "txt_idx" not in st.session_state or st.session_state.txt_idx not in remaining.index:
        st.session_state.txt_idx = remaining.index[0]

    row          = df.loc[st.session_state.txt_idx]
    review_text  = row["review"]
    ground_truth = str(row.get("ground_truth", "unknown")).lower().strip()

    done = len(st.session_state.annotated_ids)
    st.progress(done / len(df), text=f"Progress: {done}/{len(df)} reviews annotated")

    truncated = review_text[:500] + ("..." if len(review_text) > 500 else "")
    st.markdown(f"""
    <div style='background:#0d1117;border:1px solid #30363d;border-radius:10px;
                padding:1.5rem;margin:1rem 0;font-size:1.05rem;line-height:1.7;
                color:#c9d1d9;font-family:Georgia,serif;min-height:120px'>
        "{truncated}"
    </div>
    """, unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    i1.caption(f"📏 Words: {len(review_text.split())}")
    i2.caption(f"📌 Review #{st.session_state.txt_idx + 1}")
    i3.caption(f"📋 Remaining: {len(remaining)}")

    if st.button("🤖 AI Suggestion"):
        with st.spinner("Analyzing..."):
            ai_label, ai_conf = get_ai_sentiment_suggestion(review_text)
        cfg = SENTIMENT_CONFIG.get(ai_label, SENTIMENT_CONFIG["neutral"])
        st.markdown(f"""
        <div style='background:{cfg["bg"]};border:1px solid {cfg["color"]};
                    border-radius:8px;padding:.8rem;margin:.5rem 0'>
            <span style='color:{cfg["color"]};font-weight:700'>
                {cfg["icon"]} AI suggests: {ai_label.upper()}
            </span>
            <span style='color:#8892b0'> — confidence: {ai_conf:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Your Annotation:")
    start_time = time.time()
    sc1, sc2, sc3 = st.columns(3)
    selected = None
    with sc1:
        if st.button("😊 Positive", use_container_width=True):
            selected = "positive"
    with sc2:
        if st.button("😞 Negative", use_container_width=True):
            selected = "negative"
    with sc3:
        if st.button("😐 Neutral",  use_container_width=True):
            selected = "neutral"

    if selected:
        st.session_state["_sel"] = selected

    if st.session_state.get("_sel"):
        cur_sel = st.session_state["_sel"]
        cfg     = SENTIMENT_CONFIG.get(cur_sel, SENTIMENT_CONFIG["neutral"])
        st.markdown(f"""
        <div style='background:{cfg["bg"]};border:1px solid {cfg["color"]};
                    border-radius:8px;padding:.8rem;text-align:center;margin:.5rem 0'>
            <span style='color:{cfg["color"]};font-size:1.1rem;font-weight:700'>
                {cfg["icon"]} Selected: {cur_sel.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

        is_correct = (cur_sel == ground_truth) if ground_truth != "unknown" else None
        if is_correct is True:  st.success(f"✅ Matches ground truth: **{ground_truth}**")
        elif is_correct is False: st.warning(f"⚠️ Ground truth: **{ground_truth}**")

        conf = st.slider("Confidence", 0.0, 1.0, 0.85, 0.05, key="txt_conf")

        sv1, sv2 = st.columns([3, 1])
        with sv1:
            if st.button("💾 Save & Next", type="primary", use_container_width=True):
                save_text_annotation(
                    text_id=int(st.session_state.txt_idx), review=review_text[:500],
                    ground_truth=ground_truth, annotator_label=cur_sel,
                    is_correct=int(is_correct) if is_correct is not None else -1,
                    annotator=annotator, confidence=conf,
                    time_taken=round(time.time() - start_time, 1),
                )
                st.session_state.annotated_ids.add(st.session_state.txt_idx)
                st.session_state.txt_count   = st.session_state.get("txt_count", 0) + 1
                if is_correct: st.session_state.txt_correct = st.session_state.get("txt_correct", 0) + 1
                rem2 = df[~df.index.isin(st.session_state.annotated_ids)]
                if not rem2.empty: st.session_state.txt_idx = rem2.index[0]
                st.session_state.pop("_sel", None)
                st.rerun()
        with sv2:
            if st.button("⏭️ Skip"):
                st.session_state.annotated_ids.add(st.session_state.txt_idx)
                rem2 = df[~df.index.isin(st.session_state.annotated_ids)]
                if not rem2.empty: st.session_state.txt_idx = rem2.index[0]
                st.session_state.pop("_sel", None)
                st.rerun()
    else:
        st.info("👆 Click Positive, Negative, or Neutral to annotate")

# 7. AUDIO ANNOTATION MODULE
def _create_sample_audio_meta():
    os.makedirs("data/audio", exist_ok=True)
    emotions = list(EMOTIONS.keys())
    data = [{
        "filename":     f"audio_{i+1:03d}.wav",
        "emotion":      random.choice(emotions),
        "duration_sec": round(random.uniform(2.5, 5.0), 2),
        "actor_id":     random.randint(1, 24),
    } for i in range(30)]
    df = pd.DataFrame(data)
    df.to_csv("data/audio/audio_metadata.csv", index=False)
    return df

def _audio_annotation_form(filename, ground_truth, annotator, mode="file"):
    transcription = st.text_area(
        "Transcription (optional)",
        placeholder="Type what you hear...",
        height=80, key=f"trans_{filename}",
    )

    st.markdown("**Select Emotion:**")
    emo_items = list(EMOTIONS.items())
    cols = st.columns(4)
    sel_emo = st.session_state.get(f"sel_emo_{filename}")

    for i, (emo, cfg) in enumerate(emo_items):
        with cols[i % 4]:
            btn_type = "primary" if sel_emo == emo else "secondary"
            if st.button(f"{cfg['icon']} {emo.capitalize()}",
                         key=f"btn_{emo}_{filename}",
                         use_container_width=True, type=btn_type):
                st.session_state[f"sel_emo_{filename}"] = emo
                st.rerun()

    sel_emo = st.session_state.get(f"sel_emo_{filename}")

    if sel_emo:
        cfg = EMOTIONS[sel_emo]
        st.markdown(f"""
        <div style='background:#0d1117;border:1px solid {cfg["color"]};
                    border-radius:8px;padding:.8rem;text-align:center;margin:.5rem 0'>
            <span style='color:{cfg["color"]};font-size:1.2rem;font-weight:700'>
                {cfg["icon"]} Selected: {sel_emo.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

        is_correct = sel_emo == ground_truth.lower() if ground_truth != "unknown" else None
        if is_correct is True:  st.success(f"✅ Matches ground truth: **{ground_truth}**")
        elif is_correct is False: st.warning(f"⚠️ Ground truth: **{ground_truth}**")

        conf = st.slider("📊 Confidence", 0.0, 1.0, 0.85, 0.05, key=f"conf_aud_{filename}")

        sv1, sv2 = st.columns([3, 1])
        with sv1:
            if st.button("💾 Save & Next", type="primary", use_container_width=True, key=f"save_{filename}"):
                save_audio_annotation(
                    audio_file=filename, ground_truth_emotion=ground_truth,
                    annotator_emotion=sel_emo, transcription=transcription,
                    is_correct=int(is_correct) if is_correct is not None else -1,
                    annotator=st.session_state.get("aud_annotator", "Annotator_1"),
                    confidence=conf, time_taken=0.0,
                )
                st.session_state.aud_count   = st.session_state.get("aud_count", 0) + 1
                if is_correct: st.session_state.aud_correct = st.session_state.get("aud_correct", 0) + 1
                nav_key = "aud_idx" if mode == "file" else "aud_meta_idx"
                st.session_state[nav_key] = st.session_state.get(nav_key, 0) + 1
                st.session_state.pop(f"sel_emo_{filename}", None)
                st.rerun()
        with sv2:
            if st.button("⏭️ Skip", key=f"skip_{filename}"):
                nav_key = "aud_idx" if mode == "file" else "aud_meta_idx"
                st.session_state[nav_key] = st.session_state.get(nav_key, 0) + 1
                st.session_state.pop(f"sel_emo_{filename}", None)
                st.rerun()
    else:
        st.info("👆 Select an emotion above")

def show_audio_module():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#2e1a1a);
                padding:1.2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;
                border-left:4px solid #E91E63'>
        <h2 style='color:#fff;margin:0'>🎧 Audio Annotation</h2>
        <p style='color:#8892b0;margin:.2rem 0 0'>Label emotions in speech audio clips</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        annotator = st.text_input("Annotator Name", value="Annotator_1", key="aud_annotator")
    with c2:
        st.metric("Done", st.session_state.get("aud_count", 0))
    with c3:
        correct    = st.session_state.get("aud_correct", 0)
        done_total = max(st.session_state.get("aud_count", 1), 1)
        st.metric("Accuracy", f"{correct/done_total*100:.0f}%")

    st.divider()

    audio_files = (glob.glob("data/audio/**/*.wav", recursive=True) +
                   glob.glob("data/audio/**/*.mp3", recursive=True) +
                   glob.glob("data/audio/*.wav") +
                   glob.glob("data/audio/*.mp3"))
    audio_files = list(dict.fromkeys(audio_files))

    meta_path = "data/audio/audio_metadata.csv"
    df_meta   = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    if audio_files:
        # ── File mode ──
        total = len(audio_files)
        if "aud_idx" not in st.session_state: st.session_state.aud_idx = 0
        idx = st.session_state.aud_idx % total

        n1, n2, n3, n4 = st.columns([1, 1, 3, 1])
        with n1:
            if st.button("⬅️ Prev") and idx > 0:
                st.session_state.aud_idx -= 1; st.rerun()
        with n2:
            if st.button("Next ➡️") and idx < total - 1:
                st.session_state.aud_idx += 1; st.rerun()
        with n3:
            st.progress((idx + 1) / total, text=f"Audio {idx+1} of {total}")
        with n4:
            if st.button("🔀 Random"):
                st.session_state.aud_idx = np.random.randint(0, total); st.rerun()

        audio_path   = audio_files[idx]
        filename     = os.path.basename(audio_path)
        ground_truth = "unknown"
        if df_meta is not None and "filename" in df_meta.columns:
            m = df_meta[df_meta["filename"] == filename]
            if not m.empty: ground_truth = str(m.iloc[0].get("emotion", "unknown"))

        st.markdown(f"**🎵 {filename}**")
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        # Waveform
        try:
            import librosa, librosa.display
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            y, sr = librosa.load(audio_path, duration=5.0)
            fig, ax = plt.subplots(figsize=(8, 2))
            fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
            ts = np.linspace(0, len(y)/sr, len(y))
            ax.plot(ts, y, color="#E91E63", linewidth=0.5, alpha=0.8)
            ax.fill_between(ts, y, alpha=0.2, color="#E91E63")
            ax.tick_params(colors="#8892b0")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        except Exception:
            st.caption("ℹ️ Install librosa for waveform visualization")

        _audio_annotation_form(filename, ground_truth, annotator, mode="file")

    else:
        # ── Metadata / Demo mode ──
        st.info("📋 **Demo Mode** — No audio files found. Showing metadata-only annotation.")
        if df_meta is None: df_meta = _create_sample_audio_meta()
        if "aud_meta_idx" not in st.session_state: st.session_state.aud_meta_idx = 0
        idx = st.session_state.aud_meta_idx % len(df_meta)
        row = df_meta.iloc[idx]
        st.progress((idx + 1) / len(df_meta), text=f"Audio {idx+1} of {len(df_meta)}")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("File",          row.get("filename",     f"audio_{idx+1:03d}.wav"))
        mc2.metric("Ground Truth",  row.get("emotion",      "unknown"))
        mc3.metric("Duration",      f"{row.get('duration_sec', 3.0):.1f}s")
        _audio_annotation_form(
            str(row.get("filename", f"audio_{idx+1}.wav")),
            str(row.get("emotion", "unknown")),
            annotator, mode="meta",
        )

# 8. DASHBOARD MODULE
def _generate_demo_dashboard_data():
    rng = np.random.default_rng(42)
    labels    = ["person","car","dog","cat","bird"]
    sentiments = ["positive","negative","neutral"]
    emotions  = list(EMOTIONS.keys())
    annotators = ["Annotator_A","Annotator_B","Annotator_C"]

    img_data = [{
        "id": i+1, "image_file": f"img_{i+1:04d}.jpg",
        "x_min": int(rng.integers(0,200)), "y_min": int(rng.integers(0,150)),
        "x_max": int(rng.integers(200,400)), "y_max": int(rng.integers(150,350)),
        "label": random.choice(labels),
        "annotator": random.choice(annotators),
        "confidence": round(float(rng.uniform(0.7,1.0)),2),
        "time_taken": round(float(rng.uniform(5,30)),1),
        "created_at": str(datetime.now() - timedelta(hours=int(rng.integers(0,48)))),
    } for i in range(40)]

    txt_data = [{
        "id": i+1, "text_id": i, "review": f"Sample review {i+1}.",
        "ground_truth_label": rng.choice(sentiments, p=[0.5,0.4,0.1]),
        "annotator_label":    rng.choice(sentiments, p=[0.5,0.4,0.1]),
        "annotator": random.choice(annotators),
        "confidence": round(float(rng.uniform(0.7,1.0)),2),
        "time_taken": round(float(rng.uniform(3,20)),1),
        "created_at": str(datetime.now() - timedelta(hours=int(rng.integers(0,48)))),
    } for i in range(80)]
    for r in txt_data:
        r["is_correct"] = int(r["ground_truth_label"] == r["annotator_label"])

    aud_data = [{
        "id": i+1, "audio_file": f"audio_{i+1:03d}.wav",
        "ground_truth_emotion": random.choice(emotions),
        "annotator_emotion":    random.choice(emotions),
        "transcription": "",
        "annotator": random.choice(annotators),
        "confidence": round(float(rng.uniform(0.6,1.0)),2),
        "time_taken": round(float(rng.uniform(5,25)),1),
        "created_at": str(datetime.now() - timedelta(hours=int(rng.integers(0,48)))),
    } for i in range(60)]
    for r in aud_data:
        r["is_correct"] = int(r["ground_truth_emotion"] == r["annotator_emotion"])

    return pd.DataFrame(img_data), pd.DataFrame(txt_data), pd.DataFrame(aud_data)

def _plotly_dark(fig, height=300):
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#c9d1d9", height=height,
        xaxis=dict(gridcolor="#1e2730"), yaxis=dict(gridcolor="#1e2730"),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig

def show_dashboard():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                padding:1.2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;
                border-left:4px solid #FFD700'>
        <h2 style='color:#fff;margin:0'>📊 Analytics Dashboard</h2>
        <p style='color:#8892b0;margin:.2rem 0 0'>Quality, productivity, and performance tracking</p>
    </div>
    """, unsafe_allow_html=True)

    df_img = _load_table("image_annotations")
    df_txt = _load_table("text_annotations")
    df_aud = _load_table("audio_annotations")

    using_demo = df_img.empty and df_txt.empty and df_aud.empty
    if using_demo:
        st.info("No real annotations yet — showing **demo data** to preview the dashboard.")
        df_img, df_txt, df_aud = _generate_demo_dashboard_data()

    # ── KPIs ──
    st.markdown("### Overall Statistics")
    t_img = len(df_img); t_txt = len(df_txt); t_aud = len(df_aud)
    t_all = t_img + t_txt + t_aud

    txt_acc = (df_txt["is_correct"].clip(lower=0).mean()*100) if not df_txt.empty and "is_correct" in df_txt.columns else 0
    aud_acc = (df_aud["is_correct"].clip(lower=0).mean()*100) if not df_aud.empty and "is_correct" in df_aud.columns else 0
    overall_acc = (txt_acc + aud_acc) / 2 if (txt_acc + aud_acc) > 0 else 0
    acc_color = "#4CAF50" if overall_acc >= 85 else "#FF9800" if overall_acc >= 70 else "#F44336"

    kpi = lambda value, label, color, border: f"""
    <div style='background:#0d1117;border:1px solid {border};border-top:3px solid {color};
                border-radius:10px;padding:1rem;text-align:center'>
        <div style='font-size:1.8rem;font-weight:800;color:{color}'>{value}</div>
        <div style='font-size:.75rem;color:#8b949e;margin-top:.2rem'>{label}</div>
    </div>"""

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(kpi(t_all,   "Total Annotations","#fff",    "#30363d"), unsafe_allow_html=True)
    k2.markdown(kpi(t_img,   "🖼️ Image",         "#4B8BFF", "#4B8BFF"), unsafe_allow_html=True)
    k3.markdown(kpi(t_txt,   "📝 Text",           "#4CAF50", "#4CAF50"), unsafe_allow_html=True)
    k4.markdown(kpi(t_aud,   "🎧 Audio",          "#E91E63", "#E91E63"), unsafe_allow_html=True)
    k5.markdown(kpi(f"{overall_acc:.1f}%","Overall Accuracy",acc_color,acc_color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ──
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("#### Annotations by Module")
        fig = go.Figure(go.Bar(
            x=["🖼️ Image","📝 Text","🎧 Audio"], y=[t_img,t_txt,t_aud],
            marker_color=["#4B8BFF","#4CAF50","#E91E63"],
            text=[t_img,t_txt,t_aud], textposition="outside",
        ))
        st.plotly_chart(_plotly_dark(fig), use_container_width=True)

    with r1c2:
        st.markdown("#### Accuracy by Module")
        fig2 = go.Figure()
        for mod, acc in [("📝 Text", txt_acc), ("🎧 Audio", aud_acc)]:
            c = "#4CAF50" if acc >= 85 else "#FF9800" if acc >= 70 else "#F44336"
            fig2.add_trace(go.Bar(name=mod, x=[mod], y=[acc],
                                  marker_color=c, text=[f"{acc:.1f}%"], textposition="outside"))
        fig2.add_hline(y=85, line_dash="dash", line_color="#FFD700",
                       annotation_text="Target 85%", annotation_position="top right")
        fig2.update_layout(showlegend=False, yaxis=dict(range=[0,115]))
        st.plotly_chart(_plotly_dark(fig2), use_container_width=True)

    # ── Row 2 ──
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("#### Sentiment Distribution")
        if not df_txt.empty and "annotator_label" in df_txt.columns:
            lc = df_txt["annotator_label"].value_counts()
            fig3 = px.pie(values=lc.values, names=lc.index, hole=0.4,
                          color_discrete_map={"positive":"#4CAF50","negative":"#F44336","neutral":"#FF9800"})
            st.plotly_chart(_plotly_dark(fig3), use_container_width=True)
        else:
            st.info("No text annotations yet")

    with r2c2:
        st.markdown("#### Emotion Distribution")
        if not df_aud.empty and "annotator_emotion" in df_aud.columns:
            ec = df_aud["annotator_emotion"].value_counts()
            fig4 = px.bar(x=ec.index, y=ec.values, color=ec.index,
                          text=ec.values, color_discrete_sequence=px.colors.qualitative.Set3)
            fig4.update_layout(showlegend=False)
            st.plotly_chart(_plotly_dark(fig4), use_container_width=True)
        else:
            st.info("No audio annotations yet")

    # ── Annotator Performance ──
    st.markdown("#### Annotator Performance")
    agg = {}
    for df, has_correct in [(df_txt, True), (df_aud, True)]:
        if df.empty or "annotator" not in df.columns: continue
        for ann, grp in df.groupby("annotator"):
            if ann not in agg: agg[ann] = {"count":0,"correct":0,"time":0}
            agg[ann]["count"] += len(grp)
            if has_correct and "is_correct" in grp.columns:
                agg[ann]["correct"] += int(grp["is_correct"].clip(lower=0).sum())
            if "time_taken" in grp.columns:
                agg[ann]["time"] += float(grp["time_taken"].sum())

    if agg:
        rows = []
        for name, s in agg.items():
            acc   = s["correct"] / max(s["count"],1) * 100
            avg_t = s["time"]    / max(s["count"],1)
            rows.append({"Annotator":name,"Annotations":s["count"],
                         "Accuracy (%)":round(acc,1),"Avg Time (s)":round(avg_t,1)})
        df_perf = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)

        def hl(row):
            c = "#1a2e1a" if row["Accuracy (%)"]>=85 else "#2e251a" if row["Accuracy (%)"]>=70 else "#2e1a1a"
            return [f"background-color:{c}"]*len(row)

        st.dataframe(
            df_perf.style.apply(hl, axis=1).format({"Accuracy (%)":"{:.1f}%","Avg Time (s)":"{:.1f}s"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("No annotator data available")

    # ── SLA Tracker ──
    st.markdown("#### ⏱️ SLA & Daily Target Tracker")
    daily_target = st.slider(" Daily Target", 10, 200, 50, 10)
    today_count  = t_all % max(daily_target, 1) if t_all > 0 else int(daily_target * 0.65)
    pct          = min(today_count / daily_target, 1.0)
    gauge_color  = "#4CAF50" if pct >= 1.0 else "#FF9800" if pct >= 0.7 else "#F44336"

    gc1, gc2 = st.columns(2)
    with gc1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=today_count,
            delta={"reference": daily_target, "valueformat":".0f"},
            title={"text":"Today's Progress","font":{"color":"#c9d1d9"}},
            gauge={
                "axis":{"range":[0,daily_target],"tickcolor":"#8892b0"},
                "bar":{"color":gauge_color},
                "bgcolor":"#0d1117","bordercolor":"#30363d",
                "steps":[{"range":[0,daily_target*.7],"color":"#1e2730"},
                         {"range":[daily_target*.7,daily_target],"color":"#1a2a1a"}],
                "threshold":{"line":{"color":"#FFD700","width":3},"thickness":.75,"value":daily_target},
            },
            number={"font":{"color":"#c9d1d9"}},
        ))
        fig_g.update_layout(paper_bgcolor="#0d1117",font_color="#c9d1d9",
                            height=250,margin=dict(t=30,b=10,l=30,r=30))
        st.plotly_chart(fig_g, use_container_width=True)

    with gc2:
        remaining = max(daily_target - today_count, 0)
        status    = "✅ ON TRACK" if pct >= .7 else "⚠️ BEHIND"
        st.markdown(f"""
        <div style='background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:1.2rem;'>
            <h4 style='color:#fff;margin-top:0'>SLA Summary</h4>
            <table style='width:100%;color:#c9d1d9;'>
                <tr><td>📌 Daily Target</td>
                    <td style='text-align:right;font-weight:700'>{daily_target}</td></tr>
                <tr><td>✅ Completed</td>
                    <td style='text-align:right;color:#4CAF50;font-weight:700'>{today_count}</td></tr>
                <tr><td>⏳ Remaining</td>
                    <td style='text-align:right;color:#FF9800;font-weight:700'>{remaining}</td></tr>
                <tr><td>📊 Completion</td>
                    <td style='text-align:right;color:{gauge_color};font-weight:700'>{pct*100:.0f}%</td></tr>
                <tr><td>🚦 Status</td>
                    <td style='text-align:right;font-weight:700;color:{gauge_color}'>{status}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Export ──
    st.markdown("#### 📥 Export Annotations")
    ec1, ec2, ec3 = st.columns(3)
    for col, df, name in [(ec1,df_img,"image"),(ec2,df_txt,"text"),(ec3,df_aud,"audio")]:
        with col:
            if not df.empty:
                st.download_button(
                    f"⬇️ {name.capitalize()} CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name}_annotations.csv",
                    mime="text/csv", use_container_width=True,
                )
            else:
                st.button(f"⬇️ {name.capitalize()} CSV", disabled=True, use_container_width=True)

# 9. HOME PAGE
def show_home():
    st.markdown("""
    <div style='text-align:center;padding:3rem 0 2rem'>
        <div style='font-size:4rem;margin-bottom:.5rem'>🏷️</div>
        <h1 style='font-size:2.5rem;font-weight:800;
                   background:linear-gradient(135deg,#4B8BFF,#E91E63);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0'>
            ML Data Annotation Platform
        </h1>
        <p style='color:#8b949e;font-size:1.1rem;margin-top:.8rem'>
            A production-grade multi-modal annotation system for AI training data
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "🖼️","Image Annotation","Draw bounding boxes and label objects using an interactive canvas","#4B8BFF"),
        (c2, "📝","Text Annotation", "Classify sentiment of movie reviews with AI-assisted suggestions","#4CAF50"),
        (c3, "🎧","Audio Annotation","Label emotions in speech audio clips from RAVDESS dataset",        "#E91E63"),
        (c4, "📊","Dashboard",       "Track quality, accuracy, SLA compliance, and annotator performance","#FFD700"),
    ]
    for col, icon, title, desc, color in cards:
        with col:
            st.markdown(f"""
            <div style='background:#1c2128;border:1px solid #30363d;border-top:3px solid {color};
                        border-radius:12px;padding:1.5rem;min-height:200px'>
                <div style='font-size:2rem;margin-bottom:.5rem'>{icon}</div>
                <div style='font-weight:700;color:#e6edf3;font-size:1rem;margin-bottom:.5rem'>{title}</div>
                <div style='color:#8b949e;font-size:.85rem;line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📦 Datasets")
    d1, d2, d3 = st.columns(3)
    for col, (name, desc, lbl) in zip([d1,d2,d3], [
        (" COCO 2017 / Pascal VOC","Object detection with bounding boxes","Person, Car, Dog, Cat, Bird..."),
        (" IMDB Movie Reviews",     "Sentiment classification (50K reviews)","Positive / Negative / Neutral"),
        (" RAVDESS Emotional Speech","Speech emotion recognition (24 actors)","Happy, Sad, Angry, Fearful..."),
    ]):
        with col:
            st.markdown(f"""
            <div style='background:#1c2128;border:1px solid #30363d;border-radius:10px;padding:1rem'>
                <div style='font-weight:700;color:#e6edf3;margin-bottom:.4rem'>{name}</div>
                <div style='color:#8b949e;font-size:.82rem;margin-bottom:.3rem'>{desc}</div>
                <div style='font-size:.78rem;color:#484f58'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Use the sidebar to navigate. Run the Jupyter notebook first to download Kaggle datasets.")

# 10. SIDEBAR NAVIGATION
_ensure_dirs()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem'>
        <div style='font-size:2.5rem'>🏷️</div>
        <div style='font-size:1.1rem;font-weight:800;color:#e6edf3;letter-spacing:-.5px'>
            Annotation Platform
        </div>
        <div style='font-size:.7rem;color:#8b949e;margin-top:.2rem'>ML Data Labeling Tool</div>
    </div>
    <hr style='border-color:#30363d'>
    """, unsafe_allow_html=True)

    module = st.selectbox(
        "Select Module",
        ["🏠 Home", "🖼️ Image Annotation", "📝 Text Annotation",
         "🎧 Audio Annotation", "📊 Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    img_c = st.session_state.get("img_count", 0)
    txt_c = st.session_state.get("txt_count", 0)
    aud_c = st.session_state.get("aud_count", 0)
    total = img_c + txt_c + aud_c

    st.markdown(f"""
    <div style='background:#1c2128;border:1px solid #30363d;border-radius:10px;padding:1rem'>
        <div style='font-size:.7rem;color:#8b949e;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:.8rem'>Session Stats</div>
        <div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>
            <span style='color:#8b949e;font-size:.85rem'> Images</span>
            <span style='color:#4B8BFF;font-weight:700'>{img_c}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>
            <span style='color:#8b949e;font-size:.85rem'> Text</span>
            <span style='color:#4CAF50;font-weight:700'>{txt_c}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:.6rem'>
            <span style='color:#8b949e;font-size:.85rem'> Audio</span>
            <span style='color:#E91E63;font-weight:700'>{aud_c}</span>
        </div>
        <hr style='border-color:#30363d;margin:.5rem 0'>
        <div style='display:flex;justify-content:space-between'>
            <span style='color:#e6edf3;font-size:.85rem;font-weight:600'>Total</span>
            <span style='color:#FFD700;font-weight:800;font-size:1rem'>{total}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem;color:#484f58;text-align:center'>
        Built for Amazon ML Annotation Role<br>
        Datasets: COCO · IMDB · RAVDESS
    </div>
    """, unsafe_allow_html=True)

# 11. ROUTING

if   module == "🏠 Home":             show_home()
elif module == "🖼️ Image Annotation": show_image_module()
elif module == "📝 Text Annotation":  show_text_module()
elif module == "🎧 Audio Annotation": show_audio_module()
elif module == "📊 Dashboard":        show_dashboard()
