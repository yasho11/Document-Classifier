# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pickle
import os

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """Plot a confusion matrix without seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, 
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()

def plot_classification_metrics(y_true, y_pred, classes):
    """Plot precision, recall, and f1-score for each class"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=classes)
    support = np.bincount([list(classes).index(label) for label in y_true])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon', edgecolor='black')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Classification Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend()
    
    # Add support counts below x-axis
    for i, sup in enumerate(support):
        ax.text(i, -0.1, f'n={sup}', ha='center', va='top', transform=ax.get_xaxis_transform())
    
    fig.tight_layout()
    plt.show()

def evaluate_model_with_plots(model, X_test, y_test):
    """Evaluate model with visualizations"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get unique classes in correct order
    classes = model.classes_
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, classes)
    
    # Plot 2: Classification Metrics
    plot_classification_metrics(y_test, y_pred, classes)
    
    return {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_true': y_test,
        'classes': classes
    }

def evaluate_model_simple(model, X_test, y_test):
    """Simple evaluation without visualizations"""
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

def load_model_from_pickle(model_path="news_classifier.pkl"):
    """Load a pre-trained model from pickle file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"Loaded model from: {model_path}")
    print(f"Model classes: {model.classes_}")
    return model

if __name__ == "__main__":
    from pre_train import load_dataset_csv, preprocess
    from sklearn.model_selection import train_test_split
    
    # Configuration
    CSV_PATH = "dataset.csv"  # Update this if needed
    MODEL_PATH = "news_classifier.pkl"  # Update this if needed
    
    # Load the pre-trained model
    print("Loading pre-trained model...")
    model = load_model_from_pickle(MODEL_PATH)
    
    # Load and preprocess data for evaluation
    print("\nLoading evaluation data...")
    texts, labels = load_dataset_csv(CSV_PATH)
    processed_texts = [preprocess(text) for text in texts]
    
    # Split data (use same random_state for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nDataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Choose evaluation mode
    print("\nChoose evaluation mode:")
    print("1. With visualizations (requires matplotlib)")
    print("2. Simple evaluation (text only)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            # Evaluate with plots
            results = evaluate_model_with_plots(model, X_test, y_test)
            
            # Save plots to files
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, results['y_pred'])
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(results['classes']))
            plt.xticks(tick_marks, results['classes'], rotation=45)
            plt.yticks(tick_marks, results['classes'])
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("\nConfusion matrix saved as 'confusion_matrix.png'")
            
        else:
            # Simple evaluation
            results = evaluate_model_simple(model, X_test, y_test)
            
    except Exception as e:
        print(f"\nError in visualization: {e}")
        print("Falling back to simple evaluation...")
        results = evaluate_model_simple(model, X_test, y_test)