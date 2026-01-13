import os
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#Data on drive: 

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# -----------------------------
# 1. Load dataset
# -----------------------------
def load_dataset(data_path):
    texts = []
    labels = []

    categories = ["business", "entertainment", "health"]

    for category in categories:
        folder_path = os.path.join(data_path, category)
        print(f"Loading {category} documents from {folder_path}...")

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found!")
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()

                    if text:
                        texts.append(text)
                        labels.append(category)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"\nLoaded {len(texts)} documents across {len(set(labels))} categories.")
    return texts, labels


# -----------------------------
# 2. Text preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words
    ]
    return " ".join(tokens)


# -----------------------------
# 3. Train model (for Streamlit)
# -----------------------------
def train_and_return_model():
    DATA_PATH = "data"

    texts, labels = load_dataset(DATA_PATH)
    if len(texts) == 0:
        raise ValueError("No documents found. Check data path.")

    processed_texts = [preprocess(text) for text in texts]

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

    model.fit(X_train, y_train)

    return model


# -----------------------------
# 4. CLI execution (optional)
# -----------------------------
if __name__ == "__main__":

    DATA_PATH = "data"
    texts, labels = load_dataset(DATA_PATH)

    if len(texts) == 0:
        print("No documents loaded. Check your DATA_PATH.")
        exit()

    print("\nPreprocessing texts...")
    processed_texts = [preprocess(text) for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    print("\nTraining the classifier...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

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
