import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

from nn import MLP


FEATURES = ["income", "debtRatio", "savingsBuffer"]
OUTPUT_DIR = Path(
    os.getenv(
        "MODEL_OUTPUT_DIR",
        Path(__file__).resolve().parent.parent / "visualizer" / "src",
    )
)
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def generate_applicants(count, seed=7):
    rng = random.Random(seed)
    records = []

    for _ in range(count):
        income = clamp(rng.betavariate(2.5, 2.0))
        debt_ratio = clamp(rng.gauss(0.45 - 0.3 * income, 0.18))
        savings_buffer = clamp(rng.gauss(0.25 + 0.5 * income - 0.25 * debt_ratio, 0.2))

        latent = 2.2 * income - 2.6 * debt_ratio + 1.8 * savings_buffer
        latent += rng.gauss(0, 0.35)
        approval_prob = sigmoid(latent)
        approved = rng.random() < approval_prob

        records.append(
            {
                "income": income,
                "debtRatio": debt_ratio,
                "savingsBuffer": savings_buffer,
                "approved": approved,
            }
        )

    return records


def split_dataset(records, train_ratio=0.8, seed=7):
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    cutoff = int(len(shuffled) * train_ratio)
    return shuffled[:cutoff], shuffled[cutoff:]


def compute_approval_rate(records):
    return sum(1 for record in records if record["approved"]) / max(1, len(records))


def train_model(model, train_records, epochs=220, learning_rate=0.03, verbose=True):
    log_interval = max(1, epochs // 5)
    for epoch in range(epochs):
        total_loss = 0.0
        for record in train_records:
            inputs = [record[feature] for feature in FEATURES]
            target = 1.0 if record["approved"] else -1.0

            prediction = model(inputs)
            loss = (prediction - target) ** 2
            total_loss += loss.data

            for param in model.parameters():
                param.grad = 0.0
            loss.backward()

            for param in model.parameters():
                param.data += -learning_rate * param.grad

        if verbose and epoch % log_interval == 0:
            avg_loss = total_loss / max(1, len(train_records))
            print(f"Epoch {epoch:03d} | avg loss {avg_loss:.4f}")


def compute_metrics(scores, labels):
    predictions = [1 if score >= 0.5 else 0 for score in scores]

    tp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(labels))
    auc = compute_auc(scores, labels)
    ks = compute_ks(scores, labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auc": auc,
        "ks": ks,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_auc(scores, labels):
    paired = list(zip(scores, labels))
    paired.sort(key=lambda x: x[0])

    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    if pos_count == 0 or neg_count == 0:
        return 0.5

    rank_sum = 0.0
    idx = 0
    while idx < len(paired):
        score = paired[idx][0]
        start = idx
        while idx < len(paired) and paired[idx][0] == score:
            idx += 1
        end = idx
        avg_rank = (start + end + 1) / 2.0
        positives = sum(1 for _, label in paired[start:end] if label == 1)
        rank_sum += positives * avg_rank

    auc = (rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return auc


def compute_ks(scores, labels):
    paired = list(zip(scores, labels))
    paired.sort(key=lambda x: x[0], reverse=True)

    pos_total = sum(labels)
    neg_total = len(labels) - pos_total
    if pos_total == 0 or neg_total == 0:
        return 0.0

    pos_cum = 0
    neg_cum = 0
    ks = 0.0
    for _, label in paired:
        if label == 1:
            pos_cum += 1
        else:
            neg_cum += 1
        ks = max(ks, abs(pos_cum / pos_total - neg_cum / neg_total))

    return ks


def score_records(model, records):
    outputs = []
    for record in records:
        inputs = [record[feature] for feature in FEATURES]
        prediction = model(inputs)
        score = (prediction.data + 1) / 2
        outputs.append(score)
    return outputs


def export_artifacts(model, train_records, eval_records, metrics, seed, approval_rate):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_params = [param.data for param in model.parameters()]
    (OUTPUT_DIR / "model.json").write_text(json.dumps(model_params, indent=2))

    model_card = {
        "model": "MLP (3x4x4x1)",
        "training": "Custom autograd engine",
        "calibration": "Tanh output mapped to [0, 1] score",
        "monitoring": ["Drift checks", "Bias review", "Latency tracking"],
        "dataset": {
            "train_size": len(train_records),
            "eval_size": len(eval_records),
            "seed": seed,
            "approval_rate": round(approval_rate, 3),
        },
        "features": FEATURES,
        "thresholds": {
            "approve": 0.58,
            "manual_review": 0.48,
        },
    }
    (OUTPUT_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2))

    metrics_payload = {
        "precision": round(metrics["precision"], 3),
        "recall": round(metrics["recall"], 3),
        "f1": round(metrics["f1"], 3),
        "accuracy": round(metrics["accuracy"], 3),
        "auc": round(metrics["auc"], 3),
        "ks": round(metrics["ks"], 3),
        "sample_count": len(eval_records),
        "positive_rate": round(sum(1 for r in eval_records if r["approved"]) / max(1, len(eval_records)), 3),
        "confusion": {
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        },
    }
    (OUTPUT_DIR / "model_metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    audit_rows = []
    scores = score_records(model, eval_records)
    for idx, (record, score) in enumerate(zip(eval_records, scores), start=1):
        decision = "Approved" if score >= 0.58 else "Manual Review" if score >= 0.48 else "Declined"
        audit_rows.append(
            {
                "id": idx,
                "income": round(record["income"], 2),
                "debtRatio": round(record["debtRatio"], 2),
                "savingsBuffer": round(record["savingsBuffer"], 2),
                "score": round(score, 4),
                "decision": decision,
                "actual": "Approved" if record["approved"] else "Declined",
            }
        )

    (OUTPUT_DIR / "audit_report.json").write_text(json.dumps(audit_rows, indent=2))


def export_reports(metrics, train_records, eval_records, seed, approval_rate):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_csv = [
        "metric,value",
        f"precision,{metrics['precision']:.4f}",
        f"recall,{metrics['recall']:.4f}",
        f"f1,{metrics['f1']:.4f}",
        f"accuracy,{metrics['accuracy']:.4f}",
        f"auc,{metrics['auc']:.4f}",
        f"ks,{metrics['ks']:.4f}",
        f"tp,{metrics['tp']}",
        f"tn,{metrics['tn']}",
        f"fp,{metrics['fp']}",
        f"fn,{metrics['fn']}",
    ]
    (REPORT_DIR / "metrics.csv").write_text("\n".join(metrics_csv))

    report_lines = [
        "# Credit Risk Pipeline Summary",
        "",
        f"Run timestamp (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset",
        f"- Total records: {len(train_records) + len(eval_records)}",
        f"- Train split: {len(train_records)}",
        f"- Eval split: {len(eval_records)}",
        f"- Approval rate: {approval_rate:.3f}",
        f"- Seed: {seed}",
        "",
        "## Model",
        "- Architecture: MLP (3x4x4x1)",
        "- Training: custom autograd engine",
        "- Output calibration: tanh mapped to [0, 1] score",
        "- Decision thresholds: approve >= 0.58, manual review >= 0.48",
        "",
        "## Evaluation",
        f"- AUC: {metrics['auc']:.3f}",
        f"- KS: {metrics['ks']:.3f}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- F1: {metrics['f1']:.3f}",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        "",
        "## Confusion Matrix",
        f"- TP: {metrics['tp']}",
        f"- TN: {metrics['tn']}",
        f"- FP: {metrics['fp']}",
        f"- FN: {metrics['fn']}",
        "",
        "## Notes",
        "- Data is synthetic but structured to mimic real-world credit drivers.",
        "- Exported artifacts feed the visualizer dashboard.",
    ]
    (REPORT_DIR / "summary.md").write_text("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Train a credit risk model and export artifacts for the visualizer."
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1200,
        help="Total number of applicants to generate (default: 1200)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=220,
        help="Number of training epochs (default: 220)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="Learning rate for gradient descent (default: 0.03)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility (default: 7)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress output",
    )
    args = parser.parse_args()

    records = generate_applicants(args.dataset_size, seed=args.seed)
    train_records, eval_records = split_dataset(records, train_ratio=args.train_ratio, seed=args.seed)
    approval_rate = compute_approval_rate(records)

    model = MLP(3, [4, 4, 1])
    train_model(model, train_records, epochs=args.epochs, learning_rate=args.learning_rate, verbose=not args.quiet)

    scores = score_records(model, eval_records)
    labels = [1 if record["approved"] else 0 for record in eval_records]
    metrics = compute_metrics(scores, labels)

    export_artifacts(model, train_records, eval_records, metrics, args.seed, approval_rate)
    export_reports(metrics, train_records, eval_records, args.seed, approval_rate)

    print("\nArtifacts exported to visualizer/src")
    print("Reports exported to engine/reports")
    print(f"Eval AUC: {metrics['auc']:.3f} | F1: {metrics['f1']:.3f}")
    print(f"\nConfig: {args.dataset_size} samples, {args.epochs} epochs, lr={args.learning_rate}, seed={args.seed}")


if __name__ == "__main__":
    main()
