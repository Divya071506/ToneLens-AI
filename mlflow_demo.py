import mlflow
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the same sentiment model used by ToneLens AI
print("Loading sentiment model...")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Small demonstration dataset with manually assigned labels
messages = [
    "I love this product!",
    "This is absolutely wonderful.",
    "Thank you for your excellent support.",
    "I hate this experience.",
    "This is terrible and disappointing.",
    "The service was awful."
]

actual_labels = [
    "POSITIVE",
    "POSITIVE",
    "POSITIVE",
    "NEGATIVE",
    "NEGATIVE",
    "NEGATIVE"
]

print("Predicting sentiment...")
predicted_labels = [
    result["label"]
    for result in classifier(messages, truncation=True)
]

# Calculate evaluation metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(
    actual_labels, predicted_labels, pos_label="POSITIVE", zero_division=0
)
recall = recall_score(
    actual_labels, predicted_labels, pos_label="POSITIVE", zero_division=0
)

# Create and log an MLflow experiment run
mlflow.set_experiment("ToneLens-AI-Experiment")

with mlflow.start_run(run_name="Sentiment-Model-Evaluation"):
    mlflow.log_param("model_name", "distilbert-base-uncased-finetuned-sst-2-english")
    mlflow.log_param("dataset_size", len(messages))
    mlflow.log_param("task", "binary_sentiment_classification")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("positive_precision", precision)
    mlflow.log_metric("positive_recall", recall)

    print("\n--- ToneLens AI Evaluation ---")
    print("Actual labels:   ", actual_labels)
    print("Predicted labels:", predicted_labels)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print("\nMLflow experiment run logged successfully!")
