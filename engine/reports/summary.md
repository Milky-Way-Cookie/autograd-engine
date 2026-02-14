# Credit Risk Pipeline Summary

Run timestamp (UTC): 2026-02-14 06:35:22

## Dataset
- Total records: 1200
- Train split: 960
- Eval split: 240
- Approval rate: 0.714
- Seed: 2239

## Model
- Architecture: MLP (3x4x4x1)
- Training: custom autograd engine
- Output calibration: tanh mapped to [0, 1] score
- Decision thresholds: approve >= 0.58, manual review >= 0.48

## Evaluation
- AUC: 0.722
- KS: 0.383
- Precision: 0.748
- Recall: 0.970
- F1: 0.845
- Accuracy: 0.750

## Confusion Matrix
- TP: 163
- TN: 17
- FP: 55
- FN: 5

## Notes
- Data is synthetic but structured to mimic real-world credit drivers.
- Exported artifacts feed the visualizer dashboard.