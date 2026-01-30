import os
import string
import pickle
import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# -----------------------------
# 1. Load dataset (CSV)
# -----------------------------
def load_dataset_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Expecting columns: category, content
    required_cols = {"category", "content"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, but found {set(df.columns)}"
        )

    # Clean NaNs / empty
    df = df.dropna(subset=["category", "content"])
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""]

    texts = df["content"].tolist()
    labels = df["category"].astype(str).tolist()

    print(f"Loaded {len(texts)} rows from CSV across {df['category'].nunique()} category.")
    print("Category counts:")
    print(df["category"].value_counts())

    return texts, labels

# -----------------------------
# 2. Text preprocessing
# -----------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)

    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and token.isalpha()
    ]

    return " ".join(tokens)

# -----------------------------
# 3. Train model + save as PKL
# -----------------------------
def train_and_save_model(csv_path, model_out_path="news_classifier.pkl"):
    texts, labels = load_dataset_csv(csv_path)
    if len(texts) == 0:
        raise ValueError("No rows found in CSV. Check the dataset file.")

    print("\nPreprocessing texts...")
    processed_texts = [preprocess(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    print("\nTraining the classifier...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    with open(model_out_path, "wb") as f:
        pickle.dump(model, f)

    print("\n" + "-" * 50)
    print(f"Saved trained model to: {model_out_path}")

    return model

# -----------------------------
# 4. Load model (for Streamlit or reuse)
# -----------------------------
def load_model(model_path="news_classifier.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

# -----------------------------
# 5. CLI execution
# -----------------------------
if __name__ == "__main__":
    CSV_PATH = "dataset.csv"   # change if needed
    MODEL_PATH = "news_classifier.pkl"

    model = train_and_save_model(CSV_PATH, MODEL_PATH)

    print("\n" + "-" * 50)
    print("TEST THE CLASSIFIER")
    print("Enter a news paragraph (or type 'quit' to exit):")

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        if user_input.strip():
            processed_input = preprocess(user_input)
            prediction = model.predict([processed_input])[0]
            probabilities = model.predict_proba([processed_input])[0]

            print(f"\nPredicted category: {prediction.upper()}")
            print("Confidence scores:")

            for label, prob in sorted(
                zip(model.classes_, probabilities),
                key=lambda x: -x[1]
            ):
                print(f"  {label}: {prob:.4f}")
