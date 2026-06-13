from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import re
import emoji
import csv
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- 1. SETUP & MODEL LOADING ---

def ensure_nltk_resource(resource_paths, download_name):
    if isinstance(resource_paths, str):
        resource_paths = [resource_paths]

    for resource_path in resource_paths:
        try:
            find(resource_path)
            return
        except LookupError:
            continue

    nltk.download(download_name, quiet=True)

# Ensure NLTK resources are available for cleaning
ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
ensure_nltk_resource("corpora/stopwords", "stopwords")
ensure_nltk_resource(["corpora/wordnet", "corpora/wordnet.zip"], "wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.difference_update({'not', 'no', 'never'})
model_metrics = None

# Load all assets saved from your notebook
try:
    lr_model = joblib.load("model_lr.pkl")
    nb_model = joblib.load("model_nb.pkl")
    tfidf = joblib.load("tfidf.pkl")  # Used for Logistic Regression
    bow = joblib.load("bow.pkl")      # Used for Naive Bayes
    model_classes = joblib.load("classes.pkl") # Standard labels [Negative, Neutral, Positive]
except Exception as e:
    print(f"Error loading model files: {e}")

# --- 2. PREPROCESSING FUNCTION ---
# This must match your notebook's clean_text function exactly
def clean_text(text):
    text = emoji.demojize(text)
    text = text.replace(':', ' ').replace('_', ' ').lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

def load_data(file_path):
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            with open(file_path, newline="", encoding=encoding) as csv_file:
                return list(csv.DictReader(csv_file))
        except UnicodeDecodeError:
            continue
    with open(file_path, newline="", encoding="latin1", errors="replace") as csv_file:
        return list(csv.DictReader(csv_file))

def build_metrics():
    rows = load_data("Flipkart_Product.csv")
    processed_rows = []

    for row in rows:
        review = (row.get("Review") or "").strip()
        sentiment = (row.get("Sentiment") or "").strip()
        summary = (row.get("Summary") or "No Summary").strip() or "No Summary"

        if not review or not sentiment:
            continue

        final_review = clean_text(f"{summary} {review}")
        if not final_review:
            continue

        processed_rows.append({
            "text": final_review,
            "label": sentiment
        })

    texts = [row["text"] for row in processed_rows]
    labels = [row["label"] for row in processed_rows]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    X_test_tfidf = tfidf.transform(X_test_text)
    X_test_bow = bow.transform(X_test_text)

    lr_predictions = [model_classes[int(prediction)] for prediction in lr_model.predict(X_test_tfidf)]
    nb_predictions = [model_classes[int(prediction)] for prediction in nb_model.predict(X_test_bow)]

    labels_order = list(model_classes)

    def metric_bundle(predictions):
        matrix = confusion_matrix(y_test, predictions, labels=labels_order)
        correct = sum(1 for expected, predicted in zip(y_test, predictions) if expected == predicted)
        total = len(y_test)
        per_class_recall = {}
        for index, label in enumerate(labels_order):
            class_total = int(matrix[index].sum())
            per_class_recall[label] = round((matrix[index][index] / class_total) * 100, 2) if class_total else 0

        return {
            "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
            "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, predictions, average="weighted", zero_division=0) * 100, 2),
            "confusion_matrix": matrix.tolist(),
            "labels": labels_order,
            "per_class_recall": per_class_recall,
            "correct": correct,
            "incorrect": total - correct,
            "total": total
        }

    lr_metrics = metric_bundle(lr_predictions)
    nb_metrics = metric_bundle(nb_predictions)

    def build_conclusion(lr_metrics, nb_metrics):
        metric_names = ("accuracy", "precision", "recall", "f1")
        metric_labels = {
            "accuracy": "accuracy",
            "precision": "weighted precision",
            "recall": "weighted recall",
            "f1": "weighted F1"
        }

        lr_average = sum(lr_metrics[name] for name in metric_names) / len(metric_names)
        nb_average = sum(nb_metrics[name] for name in metric_names) / len(metric_names)
        score_gap = round(abs(lr_average - nb_average), 2)

        lr_wins = [name for name in metric_names if lr_metrics[name] > nb_metrics[name]]
        nb_wins = [name for name in metric_names if nb_metrics[name] > lr_metrics[name]]

        if score_gap < 0.25:
            preferred_model = "Tie"
        elif lr_average > nb_average:
            preferred_model = "Logistic Regression"
        else:
            preferred_model = "Naive Bayes"

        metric_summary = (
            f"Logistic Regression: {lr_metrics['accuracy']}% accuracy, "
            f"{lr_metrics['precision']}% precision, {lr_metrics['recall']}% recall, "
            f"{lr_metrics['f1']}% F1. Naive Bayes: {nb_metrics['accuracy']}% accuracy, "
            f"{nb_metrics['precision']}% precision, {nb_metrics['recall']}% recall, "
            f"{nb_metrics['f1']}% F1."
        )

        if preferred_model == "Tie":
            verdict = (
                "Both models are effectively tied on the held-out test split. "
                "The small metric differences are not large enough to justify calling one model clearly better."
            )
        else:
            winner_metrics = lr_metrics if preferred_model == "Logistic Regression" else nb_metrics
            loser_metrics = nb_metrics if preferred_model == "Logistic Regression" else lr_metrics
            winner_short = "LR" if preferred_model == "Logistic Regression" else "NB"
            opposing_wins = nb_wins if preferred_model == "Logistic Regression" else lr_wins
            leading_wins = lr_wins if preferred_model == "Logistic Regression" else nb_wins
            leading_labels = ", ".join(metric_labels[name] for name in leading_wins) or "the composite score"

            if opposing_wins:
                mixed_labels = ", ".join(metric_labels[name] for name in opposing_wins)
                verdict = (
                    f"Overall, {preferred_model} has the stronger evaluation profile, leading on {leading_labels}. "
                    f"The result is mixed because the other model leads on {mixed_labels}, so the conclusion should be read as an overall comparison, not a clean sweep."
                )
            else:
                verdict = (
                    f"Overall, {preferred_model} is the stronger model on this evaluation split, "
                    f"leading on {leading_labels} with {winner_metrics['correct']} correct predictions "
                    f"versus {loser_metrics['correct']} for the other model."
                )

            if score_gap < 1:
                verdict += " The overall gap is small, so both models remain competitive."
            else:
                verdict += f" Its average advantage across the four reported metrics is {score_gap} percentage points."

            best_class = max(
                labels_order,
                key=lambda label: winner_metrics["per_class_recall"][label] - loser_metrics["per_class_recall"][label]
            )
            class_gap = round(
                winner_metrics["per_class_recall"][best_class] - loser_metrics["per_class_recall"][best_class],
                2
            )
            if class_gap > 0:
                verdict += (
                    f" The clearest class-level gain is for {best_class} reviews, where {winner_short} recall is "
                    f"{winner_metrics['per_class_recall'][best_class]}% versus {loser_metrics['per_class_recall'][best_class]}%."
                )

        return preferred_model, f"{metric_summary} {verdict}"

    preferred_model, conclusion = build_conclusion(lr_metrics, nb_metrics)

    return {
        "lr": lr_metrics,
        "nb": nb_metrics,
        "conclusion": conclusion,
        "preferred_model": preferred_model
    }

try:
    model_metrics = build_metrics()
except Exception as e:
    print(f"Error building metrics: {e}")

# --- 3. PREDICTION ROUTE ---

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.route("/metrics", methods=["GET"])
def metrics():
    if model_metrics is None:
        return jsonify({"error": "Metrics are unavailable."}), 500
    return jsonify(model_metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        raw_review = data.get("review", "")
        
        # Clean the input text
        processed_review = clean_text(raw_review)

        # Vectorize using the correct tool for each model
        X_tfidf = tfidf.transform([processed_review])
        X_bow = bow.transform([processed_review])

        # Get Predictions (Indices: 0, 1, or 2)
        lr_pred_idx = int(lr_model.predict(X_tfidf)[0])
        nb_pred_idx = int(nb_model.predict(X_bow)[0])

        # Get Probabilities for Confidence
        lr_prob = max(lr_model.predict_proba(X_tfidf)[0]) * 100
        nb_prob = max(nb_model.predict_proba(X_bow)[0]) * 100

        return jsonify({
            "lr": {
                "sentiment": model_classes[lr_pred_idx],
                "confidence": round(lr_prob, 2)
            },
            "nb": {
                "sentiment": model_classes[nb_pred_idx],
                "confidence": round(nb_prob, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
