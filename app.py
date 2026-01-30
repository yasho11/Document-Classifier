import streamlit as st
import pickle
import os

from classifier import preprocess

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Aesthetic UI (CSS)
# -----------------------------
st.markdown("""
<style>
    .app-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        backdrop-filter: blur(10px);
    }
    .pill {
        display:inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.06);
        margin-right: 8px;
        font-size: 13px;
    }
    .title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 15px;
        opacity: 0.85;
        margin-bottom: 16px;
    }
    .small-muted {
        font-size: 12px;
        opacity: 0.7;
    }
    .stTextArea textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        background: rgba(255,255,255,0.05) !important;
    }
    div.stButton > button {
        border-radius: 14px;
        padding: 10px 16px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.16);
        background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
        transition: 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        border: 1px solid rgba(255,255,255,0.25);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title & Description
# -----------------------------
st.markdown('<div class="title">📰 News Category Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Classify news into <b>Business</b>, <b>Entertainment</b>, or <b>Health</b> using a saved Naïve Bayes model.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<span class="pill">⚡ Fast (loads from .pkl)</span>
<span class="pill">🧠 TF-IDF + Naïve Bayes</span>
<span class="pill">📌 3 category</span>
""",
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_pickle_model(model_path="news_classifier.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_pickle_model("news_classifier.pkl")

# -----------------------------
# Main Card
# -----------------------------
st.markdown('<div class="app-card">', unsafe_allow_html=True)

user_text = st.text_area(
    "Enter a news article / paragraph:",
    height=220,
    placeholder="Paste a paragraph of news here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    classify = st.button("✨ Classify")
with col2:
    st.caption("Tip: longer text usually gives better confidence.")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if classify:
    if user_text.strip():
        processed_input = preprocess(user_text)
        prediction = model.predict([processed_input])[0]
        probabilities = model.predict_proba([processed_input])[0]

        st.write("")
        st.success(f"**Predicted Category:** {prediction.upper()}")

        # Confidence UI
        st.markdown("### Confidence Scores")
        labels_probs = sorted(zip(model.classes_, probabilities), key=lambda x: -x[1])

        # top result metrics
        top_label, top_prob = labels_probs[0]
        st.metric("Top Confidence", f"{top_prob:.2%}", top_label.capitalize())

        for label, prob in labels_probs:
            st.write(f"**{label.capitalize()}** — {prob:.4f}")
            st.progress(float(prob))

        st.caption("Confidence is based on model probability estimates.")

    else:
        st.warning("Please enter some text before classifying.")
