# Credit Risk Pipeline Summary

Run timestamp (UTC): 2026-02-14 05:46:45

## Dataset
- Total records: 1200
- Train split: 960
- Eval split: 240
- Approval rate: 0.747
- Seed: 7479

## Model
- Architecture: MLP (3x4x4x1)
- Training: custom autograd engine
- Output calibration: tanh mapped to [0, 1] score
- Decision thresholds: approve >= 0.58, manual review >= 0.48

## Evaluation
- AUC: 0.775
- KS: 0.422
- Precision: 0.828
- Recall: 0.914
- F1: 0.869
- Accuracy: 0.787

## Confusion Matrix
- TP: 169
- TN: 20
- FP: 35
- FN: 16

## Notes
- Data is synthetic but structured to mimic real-world credit drivers.
- Exported artifacts feed the visualizer dashboard.