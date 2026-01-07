import streamlit as st
from classifier import preprocess, train_and_return_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("📰 News Category Classifier")
st.markdown(
    "Classify news articles into **Business**, **Entertainment**, or **Health** using a Naïve Bayes model."
)

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return train_and_return_model()

model = load_model()

# -----------------------------
# User Input
# -----------------------------
user_text = st.text_area(
    "Enter a news article or paragraph:",
    height=200,
    placeholder="Type or paste news content here..."
)



# -----------------------------
# Predict Button
# -----------------------------
if st.button("Classify News"):
    if user_text.strip():
        processed_input = preprocess(user_text)
        prediction = model.predict([processed_input])[0]
        probabilities = model.predict_proba([processed_input])[0]

        st.success(f"**Predicted Category:** {prediction.upper()}")

        st.subheader("Confidence Scores")
        for label, prob in sorted(
            zip(model.classes_, probabilities),
            key=lambda x: -x[1]
        ):
            st.write(f"**{label.capitalize()}**: {prob:.4f}")
    else:
        st.warning("Please enter some text before classifying.")
