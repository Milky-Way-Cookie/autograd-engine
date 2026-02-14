import json
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from nn import MLP
from pipeline import (
    FEATURES,
    OUTPUT_DIR,
    compute_metrics,
    score_records,
    generate_applicants,
    split_dataset,
    compute_approval_rate,
    train_model,
    export_artifacts,
    export_reports,
)

app = Flask(__name__)
CORS(app)

MODEL_PATH = OUTPUT_DIR / "model.json"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
CARD_PATH = OUTPUT_DIR / "model_card.json"
AUDIT_PATH = OUTPUT_DIR / "audit_report.json"


def load_model_weights():
    """Load trained weights from model.json."""
    if not MODEL_PATH.exists():
        return None
    try:
        weights = json.loads(MODEL_PATH.read_text())
        return weights
    except Exception:
        return None


def load_metrics():
    """Load metrics from model_metrics.json."""
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return {}


def load_model_card():
    """Load model card from model_card.json."""
    if not CARD_PATH.exists():
        return {}
    try:
        return json.loads(CARD_PATH.read_text())
    except Exception:
        return {}


def load_audit_report():
    """Load audit report from audit_report.json."""
    if not AUDIT_PATH.exists():
        return []
    try:
        return json.loads(AUDIT_PATH.read_text())
    except Exception:
        return []


def reconstruct_model_from_weights(weights):
    """Rebuild an MLP and set its weights."""
    model = MLP(3, [4, 4, 1])
    params = model.parameters()
    if len(weights) != len(params):
        return None
    for param, weight in zip(params, weights):
        param.data = weight
    return model


def parse_applicant_description(description):
    """
    Parse natural language description flexibly.
    Extracts up to 3 dollar amounts and maps them to income, debt, and savings.
    Classifies by keywords, with fallback to order if needed.
    """
    desc = description.lower()

    # Define keyword categories
    income_keywords = [
        "income", "earn", "earns", "salary", "wage", "makes", "paid",
        "annual", "monthly", "per year", "yr"
    ]
    debt_keywords = [
        "debt", "owe", "owes", "liabilit", "obligation", "credit", "loan",
        "owed", "outstanding"
    ]
    savings_keywords = [
        "saving", "savings", "cash", "liquid", "asset", "reserve", "buffer",
        "account", "fund", "balance", "emergency"
    ]

    # Extract all numbers with their context
    number_pattern = r"\$?([\d,]+(?:\.\d+)?)\s*([km])?"
    matches = list(re.finditer(number_pattern, desc))

    if not matches or len(matches) < 3:
        return None

    # Build list of (value, context_start, context_end) tuples
    numbers_with_context = []
    for match in matches:
        num_str = match.group(1).replace(",", "")
        multiplier_char = match.group(2)
        try:
            value = float(num_str)
            if multiplier_char == "k":
                value *= 1000
            elif multiplier_char == "m":
                value *= 1000000

            start = max(0, match.start() - 40)
            end = min(len(desc), match.end() + 40)
            context = desc[start:end]

            numbers_with_context.append((value, context, match.start()))
        except ValueError:
            continue

    if len(numbers_with_context) < 3:
        return None

    # Sort by position in text
    numbers_with_context.sort(key=lambda x: x[2])

    # Try to classify each number
    classified = {}  # {category: value}
    unclassified_values = []

    for value, context, _ in numbers_with_context:
        income_score = sum(1 for kw in income_keywords if kw in context)
        debt_score = sum(1 for kw in debt_keywords if kw in context)
        savings_score = sum(1 for kw in savings_keywords if kw in context)

        # Find the best match
        scores = [
            (income_score, "income"),
            (debt_score, "debt"),
            (savings_score, "savings"),
        ]
        scores.sort(reverse=True, key=lambda x: x[0])
        
        best_score, best_category = scores[0]

        # Only classify if there's strong evidence
        if best_score > 0 and best_category not in classified:
            classified[best_category] = value
        else:
            unclassified_values.append(value)

    # Fill in any missing categories from unclassified values
    for category in ["income", "debt", "savings"]:
        if category not in classified and unclassified_values:
            classified[category] = unclassified_values.pop(0)

    # Ensure all three exist (fallback to defaults if needed)
    if "income" not in classified:
        classified["income"] = unclassified_values[0] if unclassified_values else 50000
    if "debt" not in classified:
        classified["debt"] = unclassified_values[1] if len(unclassified_values) > 1 else 10000
    if "savings" not in classified:
        classified["savings"] = unclassified_values[2] if len(unclassified_values) > 2 else 20000

    # Normalize
    normalized_income = min(1.0, classified["income"] / 150000.0)
    normalized_debt = min(1.0, classified["debt"] / 100000.0)
    normalized_savings = min(1.0, classified["savings"] / 150000.0)

    return {
        "income": normalized_income,
        "debtRatio": normalized_debt,
        "savingsBuffer": normalized_savings,
    }


@app.route("/api/status", methods=["GET"])
def status():
    """Return current model status."""
    weights = load_model_weights()
    metrics = load_metrics()
    card = load_model_card()
    return jsonify(
        {
            "model_loaded": weights is not None,
            "metrics": metrics,
            "card": card,
        }
    )


@app.route("/api/train", methods=["POST"])
def train():
    """Train a new model with specified parameters."""
    try:
        payload = request.get_json() or {}
        dataset_size = payload.get("dataset_size", 1200)
        epochs = payload.get("epochs", 220)
        learning_rate = payload.get("learning_rate", 0.03)
        seed = payload.get("seed", 7)
        train_ratio = payload.get("train_ratio", 0.8)

        records = generate_applicants(dataset_size, seed=seed)
        train_records, eval_records = split_dataset(
            records, train_ratio=train_ratio, seed=seed
        )
        approval_rate = compute_approval_rate(records)

        model = MLP(3, [4, 4, 1])
        train_model(model, train_records, epochs=epochs, learning_rate=learning_rate, verbose=False)

        scores = score_records(model, eval_records)
        labels = [1 if record["approved"] else 0 for record in eval_records]
        metrics = compute_metrics(scores, labels)

        export_artifacts(model, train_records, eval_records, metrics, seed, approval_rate)
        export_reports(metrics, train_records, eval_records, seed, approval_rate)

        new_metrics = load_metrics()
        new_card = load_model_card()
        audit = load_audit_report()

        return jsonify(
            {
                "status": "success",
                "message": f"Training complete. AUC: {metrics['auc']:.3f}",
                "metrics": new_metrics,
                "card": new_card,
                "audit_sample": audit[:5],
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/infer", methods=["POST"])
def infer():
    """Run inference on provided features."""
    try:
        payload = request.get_json() or {}
        income = float(payload.get("income", 0.5))
        debt_ratio = float(payload.get("debtRatio", 0.5))
        savings_buffer = float(payload.get("savingsBuffer", 0.5))

        weights = load_model_weights()
        if not weights:
            return jsonify({"status": "error", "message": "No model loaded"}), 400

        model = reconstruct_model_from_weights(weights)
        if not model:
            return jsonify({"status": "error", "message": "Failed to reconstruct model"}), 500

        inputs = [income, debt_ratio, savings_buffer]
        prediction = model(inputs)
        score = (prediction.data + 1) / 2
        score = max(0, min(1, score))

        decision = (
            "Approved"
            if score >= 0.58
            else "Manual Review"
            if score >= 0.48
            else "Declined"
        )

        return jsonify(
            {
                "status": "success",
                "score": round(score, 4),
                "raw_output": round(prediction.data, 4),
                "decision": decision,
                "risk": round(1 - score, 4),
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/parse-description", methods=["POST"])
def parse_description():
    """Parse a natural language description into features."""
    try:
        payload = request.get_json() or {}
        description = payload.get("description", "")

        if not description.strip():
            return jsonify(
                {"status": "error", "message": "No description provided"}
            ), 400

        features = parse_applicant_description(description)
        if not features:
            return jsonify(
                {
                    "status": "error",
                    "message": "Could not parse description. Try: 'earns $75000, has $10000 debt, $20000 savings'",
                }
            ), 400

        return jsonify({"status": "success", "features": features})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model", methods=["GET"])
def get_model_data():
    """Return all model data (weights, metrics, card, audit)."""
    return jsonify(
        {
            "weights": load_model_weights(),
            "metrics": load_metrics(),
            "card": load_model_card(),
            "audit": load_audit_report(),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
