import React, { useState, useMemo, useEffect } from 'react';

const FEATURE_CONFIG = [
  {
    key: 'income',
    label: 'Income Stability',
    detail: '6-mo normalized trend',
    tone: 'from-cyan-400 to-blue-500',
  },
  {
    key: 'debtRatio',
    label: 'Debt-to-Income',
    detail: 'credit obligations',
    tone: 'from-rose-400 to-orange-500',
  },
  {
    key: 'savingsBuffer',
    label: 'Savings Buffer',
    detail: 'liquidity runway',
    tone: 'from-emerald-400 to-lime-500',
  },
];

function App() {
  const [inputs, setInputs] = useState({ income: 0.72, debtRatio: 0.28, savingsBuffer: 0.55 });
  const [prediction, setPrediction] = useState(0);
  const [modelMetrics, setModelMetrics] = useState({});
  const [modelCard, setModelCard] = useState({});
  const [auditReport, setAuditReport] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [training, setTraining] = useState(false);
  const [description, setDescription] = useState('');
  const [parseError, setParseError] = useState(null);
  const [explanations, setExplanations] = useState([]);
  const [confusionMatrix, setConfusionMatrix] = useState({});

  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000/api';

  // Fetch confusion matrix
  const loadConfusionMatrix = async () => {
    try {
      const response = await fetch(`${API_BASE}/confusion-matrix`);
      const data = await response.json();
      if (data.status === 'success') {
        setConfusionMatrix(data.confusion_matrix || {});
      }
    } catch (err) {
      console.error('Failed to load confusion matrix:', err);
    }
  };

  // Fetch explanations
  const loadExplanations = async (featureInputs) => {
    try {
      const response = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(featureInputs),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setExplanations(data.contributions || []);
      }
    } catch (err) {
      console.error('Failed to load explanations:', err);
    }
  };

  // Load model data on mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        const response = await fetch(`${API_BASE}/model`);
        const data = await response.json();
        setModelMetrics(data.metrics || {});
        setModelCard(data.card || {});
        setAuditReport(data.audit || []);
        setError(null);
        await loadConfusionMatrix();
      } catch (err) {
        setError('Failed to connect to backend. Is Flask running? (python engine/app.py)');
        console.error(err);
      }
    };
    loadModel();
  }, []);

  // Run inference via API
  const runInference = async (featureInputs = inputs) => {
    try {
      const response = await fetch(`${API_BASE}/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(featureInputs),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setPrediction(data.score);
        setError(null);
        await loadExplanations(featureInputs);
      }
    } catch (err) {
      setError('Inference failed');
      console.error(err);
    }
  };

  useEffect(() => {
    runInference(inputs);
  }, [inputs]);

  // Train new model
  const handleTrain = async (overrides = {}) => {
    setTraining(true);
    try {
      const response = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_size: overrides.dataset_size || 1200,
          epochs: overrides.epochs || 220,
          learning_rate: overrides.learning_rate || 0.03,
          seed: overrides.seed || Math.floor(Math.random() * 10000),
          ...overrides,
        }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setModelMetrics(data.metrics);
        setModelCard(data.card);
        setError(null);
        await runInference(inputs);
        await loadConfusionMatrix();
      } else {
        setError(data.message);
      }
    } catch (err) {
      setError('Training failed: ' + err.message);
      console.error(err);
    } finally {
      setTraining(false);
    }
  };

  // Parse description and fill inputs
  const handleParseDescription = async () => {
    if (!description.trim()) {
      setParseError('Please enter a description');
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/parse-description`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setInputs(data.features);
        setParseError(null);
        setDescription('');
      } else {
        setParseError(data.message);
      }
    } catch (err) {
      setParseError('Parse failed: ' + err.message);
      console.error(err);
    }
  };

  const featureImpact = useMemo(() => {
    if (explanations.length === 0) {
      return FEATURE_CONFIG.map((feature) => ({
        ...feature,
        impact: 1 / FEATURE_CONFIG.length,
      }));
    }
    return FEATURE_CONFIG.map((feature) => {
      const explanation = explanations.find(
        (exp) => exp.feature === feature.key
      );
      return {
        ...feature,
        impact: explanation ? explanation.impact_percentage / 100 : 0,
      };
    });
  }, [explanations]);

  const score = Math.max(0, Math.min(1, prediction));
  const risk = Math.max(0, Math.min(1, 1 - score));
  const decision = score >= 0.58 ? 'Approved' : score >= 0.48 ? 'Manual Review' : 'Declined';
  const decisionTone =
    decision === 'Approved'
      ? 'bg-emerald-400 text-emerald-950 shadow-emerald-500/25'
      : decision === 'Manual Review'
      ? 'bg-amber-300 text-amber-950 shadow-amber-400/25'
      : 'bg-rose-400 text-rose-950 shadow-rose-500/25';

  const performanceMetrics = [
    { label: 'Precision', value: modelMetrics?.precision?.toFixed(2) ?? '0.00' },
    { label: 'Recall', value: modelMetrics?.recall?.toFixed(2) ?? '0.00' },
    { label: 'F1 Score', value: modelMetrics?.f1?.toFixed(2) ?? '0.00' },
    { label: 'KS Stat', value: modelMetrics?.ks?.toFixed(2) ?? '0.00' },
  ];

  const modelCardItems = [
    { label: 'Model', value: modelCard?.model ?? 'MLP (3x4x4x1)' },
    { label: 'Training', value: modelCard?.training ?? 'Custom autograd engine' },
    { label: 'Calibration', value: modelCard?.calibration ?? 'Tanh -> Score mapping' },
    { label: 'Monitoring', value: (modelCard?.monitoring ?? ['Drift checks']).join(', ') },
  ];

  return (
    <div className="min-h-screen w-screen bg-slate-950 text-slate-100">
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_#1d4ed8_0%,_transparent_45%),radial-gradient(circle_at_20%_80%,_#10b981_0%,_transparent_40%),radial-gradient(circle_at_85%_65%,_#f97316_0%,_transparent_35%)] opacity-40" />
        <div className="absolute -top-24 right-[-10%] h-72 w-72 rounded-full bg-blue-500/30 blur-3xl" />
        <div className="absolute bottom-[-20%] left-[-5%] h-64 w-64 rounded-full bg-emerald-400/30 blur-3xl" />

        <div className="relative z-10 px-8 py-8">
          <div className="flex flex-wrap items-center gap-6">
            <div className="flex-1">
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400">Credit Risk Studio</p>
              <h1 className="text-4xl font-semibold text-slate-50">Loan Default Risk Scoring</h1>
              <p className="mt-2 max-w-2xl text-sm text-slate-300">
                End-to-end pipeline demo: normalized features, calibrated MLP inference, and lightweight explainability to
                support human review.
              </p>
            </div>
            <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-5 py-4 backdrop-blur">
              <div className="text-right">
                <div className="text-[10px] uppercase tracking-[0.3em] text-slate-400">Risk Score</div>
                <div className="text-3xl font-semibold text-white">{Math.round(risk * 100)}</div>
              </div>
              <div className="h-14 w-px bg-white/10" />
              <div>
                <div className="text-[10px] uppercase tracking-[0.3em] text-slate-400">Decision</div>
                <div className={`mt-2 rounded-xl px-4 py-2 text-xs font-semibold shadow-lg ${decisionTone}`}>
                  {decision}
                </div>
              </div>
            </div>
          </div>

          {error && (
            <div className="mt-6 rounded-2xl border border-red-500/50 bg-red-950/40 p-4 text-sm text-red-200">
              {error}
            </div>
          )}

          <div className="mt-8 gap-6 grid grid-cols-1 lg:grid-cols-2">
            <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-6 shadow-xl shadow-purple-500/10">
              <h3 className="text-lg font-semibold text-white">Train Model</h3>
              <p className="mt-1 text-xs uppercase tracking-[0.3em] text-slate-400">Retrain with new data</p>
              <button
                onClick={() => handleTrain()}
                disabled={training}
                className="mt-5 w-full rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 px-6 py-3 font-semibold text-white shadow-lg shadow-purple-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-purple-500/40"
              >
                {training ? 'Training...' : 'Train New Model'}
              </button>
              <p className="mt-3 text-xs text-slate-400">Generates new synthetic data and trains from scratch.</p>
            </div>

            <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-6 shadow-xl shadow-cyan-500/10">
              <h3 className="text-lg font-semibold text-white">Describe Applicant</h3>
              <p className="mt-1 text-xs uppercase tracking-[0.3em] text-slate-400">Parse natural language</p>
              <input
                type="text"
                placeholder="e.g., earns $75000, has $10000 debt, $20000 savings"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleParseDescription()}
                className="mt-4 w-full rounded-xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm text-white placeholder-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400/30"
              />
              <button
                onClick={handleParseDescription}
                className="mt-3 w-full rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-6 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-500/25 transition-all hover:shadow-cyan-500/40"
              >
                Parse & Auto-Fill
              </button>
              {parseError && <p className="mt-2 text-xs text-red-300">{parseError}</p>}
            </div>
          </div>

          <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-[1.1fr_1.3fr]">
            <div className="space-y-6">
              <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-6 shadow-xl shadow-blue-500/10">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">What-If Inputs</h2>
                    <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Normalized borrower profile</p>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] uppercase tracking-[0.3em] text-slate-400">Raw Output</div>
                    <div className="text-sm font-mono text-slate-200">{prediction.toFixed(4)}</div>
                  </div>
                </div>
                <div className="mt-6 space-y-5">
                  {FEATURE_CONFIG.map((feature) => (
                    <div key={feature.key} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-semibold text-white">{feature.label}</p>
                          <p className="text-xs text-slate-400">{feature.detail}</p>
                        </div>
                        <div className="text-sm font-mono text-slate-200">
                          {(inputs[feature.key] * 100).toFixed(0)}%
                        </div>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={inputs[feature.key]}
                        onChange={(e) =>
                          setInputs({ ...inputs, [feature.key]: parseFloat(e.target.value) })
                        }
                        className={`mt-4 w-full cursor-pointer accent-blue-400`}
                      />
                      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                        <div
                          className={`h-full bg-gradient-to-r ${feature.tone}`}
                          style={{ width: `${inputs[feature.key] * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-6 shadow-xl shadow-emerald-500/10">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Feature Impact</h2>
                    <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Local sensitivity (approx)</p>
                  </div>
                  <span className="text-[10px] uppercase tracking-[0.3em] text-slate-400">Layer 1 weights</span>
                </div>
                <div className="mt-5 space-y-4">
                  {featureImpact
                    .slice()
                    .sort((a, b) => b.impact - a.impact)
                    .map((feature) => (
                      <div key={feature.key}>
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-semibold text-white">{feature.label}</span>
                          <span className="font-mono text-slate-300">
                            {(feature.impact * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10">
                          <div
                            className={`h-full bg-gradient-to-r ${feature.tone}`}
                            style={{ width: `${feature.impact * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-6 shadow-xl shadow-orange-500/10">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Model Performance</h2>
                    <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Holdout evaluation</p>
                  </div>
                  <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] uppercase tracking-[0.3em] text-slate-300">
                    AUC {(modelMetrics?.auc ?? 0).toFixed(2)}
                  </div>
                </div>
                <div className="mt-6 grid grid-cols-2 gap-4">
                  {performanceMetrics.map((metric) => (
                    <div key={metric.label} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <p className="text-[10px] uppercase tracking-[0.3em] text-slate-400">{metric.label}</p>
                      <p className="mt-2 text-2xl font-semibold text-white">{metric.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-6 shadow-xl shadow-rose-500/10">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Confusion Matrix</h2>
                    <p className="text-xs uppercase tracking-[0.3em] text-slate-400">TP / TN / FP / FN</p>
                  </div>
                </div>
                <div className="mt-6 grid grid-cols-2 gap-4">
                  <div className="rounded-2xl border border-emerald-500/30 bg-emerald-950/40 p-4">
                    <p className="text-[10px] uppercase tracking-[0.3em] text-emerald-300">True Positives</p>
                    <p className="mt-2 text-2xl font-semibold text-emerald-300">{confusionMatrix?.tp ?? 0}</p>
                  </div>
                  <div className="rounded-2xl border border-slate-500/30 bg-slate-800/40 p-4">
                    <p className="text-[10px] uppercase tracking-[0.3em] text-slate-300">True Negatives</p>
                    <p className="mt-2 text-2xl font-semibold text-slate-300">{confusionMatrix?.tn ?? 0}</p>
                  </div>
                  <div className="rounded-2xl border border-rose-500/30 bg-rose-950/40 p-4">
                    <p className="text-[10px] uppercase tracking-[0.3em] text-rose-300">False Positives</p>
                    <p className="mt-2 text-2xl font-semibold text-rose-300">{confusionMatrix?.fp ?? 0}</p>
                  </div>
                  <div className="rounded-2xl border border-amber-500/30 bg-amber-950/40 p-4">
                    <p className="text-[10px] uppercase tracking-[0.3em] text-amber-300">False Negatives</p>
                    <p className="mt-2 text-2xl font-semibold text-amber-300">{confusionMatrix?.fn ?? 0}</p>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4 text-xs">
                  <div className="rounded-xl border border-white/10 bg-slate-950/60 p-3">
                    <p className="uppercase tracking-[0.2em] text-slate-400">Accuracy</p>
                    <p className="mt-1 font-semibold text-white">{(confusionMatrix?.accuracy ?? 0).toFixed(3)}</p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-slate-950/60 p-3">
                    <p className="uppercase tracking-[0.2em] text-slate-400">Specificity</p>
                    <p className="mt-1 font-semibold text-white">{(confusionMatrix?.specificity ?? 0).toFixed(3)}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-6 shadow-xl shadow-cyan-500/10">
                <h2 className="text-lg font-semibold text-white">Model Card</h2>
                <p className="mt-2 text-sm text-slate-300">
                  This demo uses a 3-feature credit profile to illustrate the end-to-end flow. The full pipeline includes
                  data validation, training, calibration, and audit logging.
                </p>
                <div className="mt-5 grid grid-cols-2 gap-4 text-sm">
                  {modelCardItems.map((item) => (
                    <div key={item.label} className="rounded-2xl border border-white/10 bg-slate-950/60 p-4">
                      <p className="text-[10px] uppercase tracking-[0.3em] text-slate-400">{item.label}</p>
                      <p className="mt-2 text-sm font-semibold text-white">{item.value}</p>
                    </div>
                  ))}
                </div>
                <div className="mt-6 flex flex-wrap gap-3 text-xs uppercase tracking-[0.3em] text-slate-400">
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">Bias: Checked</span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">PSI: Stable</span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">Latency: 12ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;