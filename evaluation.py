# evaluate_no_viz.py - Works without seaborn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model_simple(model, X_test, y_test):
    """Simple evaluation without seaborn dependency"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_true': y_test
    }

if __name__ == "__main__":
    from classifier import load_dataset, preprocess, train_and_return_model
    from sklearn.model_selection import train_test_split
    
    print("Loading data...")
    texts, labels = load_dataset("data")
    processed_texts = [preprocess(text) for text in texts]
    
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("Training model...")
    model = train_and_return_model()
    
    evaluate_model_simple(model, X_test, y_test)