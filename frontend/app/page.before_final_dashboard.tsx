"use client";

import { useEffect, useMemo, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

type Item = {
  id: number;
  name: string;
};

type Prediction = {
  prediction?: string;
  probability?: number;
  confidence?: number;
  interaction?: boolean;
};

type HistoryItem = {
  id: number;
  drug: string;
  microbe: string;
  prediction: string;
  probability: number;
  timestamp: string;
};

type Evaluation = {
  model: string;
  dataset: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  };
};

type DatasetStats = {
  dataset?: string;
  total_records?: number;
  unique_drugs?: number;
  unique_microbes?: number;
  positive_interactions?: number | null;
  negative_interactions?: number | null;
};

type ComparisonMetrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
};

type Comparison = {
  title?: string;
  description?: string;
  models?: {
    GCN?: ComparisonMetrics;
    HaGAT?: ComparisonMetrics;
  };
  improvement?: ComparisonMetrics;
};

export default function Home() {
  // ============================================================
  // DATA
  // ============================================================

  const [drugs, setDrugs] = useState<Item[]>([]);
  const [microbes, setMicrobes] = useState<Item[]>([]);

  const [drugSearch, setDrugSearch] = useState("");
  const [microbeSearch, setMicrobeSearch] = useState("");

  const [selectedDrug, setSelectedDrug] = useState<Item | null>(null);
  const [selectedMicrobe, setSelectedMicrobe] =
    useState<Item | null>(null);

  const [loadingData, setLoadingData] = useState(true);

  // ============================================================
  // PREDICTION
  // ============================================================

  const [prediction, setPrediction] =
    useState<Prediction | null>(null);

  const [loadingPrediction, setLoadingPrediction] =
    useState(false);

  const [error, setError] = useState("");

  // ============================================================
  // EVALUATION
  // ============================================================

  const [evaluation, setEvaluation] =
    useState<Evaluation | null>(null);

  const [evaluationLoading, setEvaluationLoading] =
    useState(true);

  const [evaluationError, setEvaluationError] =
    useState("");

  // ============================================================
  // DATASET STATS
  // ============================================================

  const [datasetStats, setDatasetStats] =
    useState<DatasetStats | null>(null);

  // ============================================================
  // COMPARISON
  // ============================================================

  const [comparison, setComparison] =
    useState<Comparison | null>(null);

  const [comparisonLoading, setComparisonLoading] =
    useState(true);

  const [comparisonError, setComparisonError] =
    useState("");

  // ============================================================
  // HISTORY
  // ============================================================

  const [history, setHistory] = useState<HistoryItem[]>([]);

  // ============================================================
  // LOAD HISTORY
  // ============================================================

  useEffect(() => {
    try {
      const saved = localStorage.getItem(
        "drugMicrobePredictionHistory"
      );

      if (saved) {
        setHistory(JSON.parse(saved));
      }
    } catch (err) {
      console.error("History error:", err);
    }
  }, []);

  // ============================================================
  // LOAD DRUGS + MICROBES
  // ============================================================

  useEffect(() => {
    async function loadData() {
      try {
        setLoadingData(true);

        const [drugRes, microbeRes] = await Promise.all([
          fetch(`${API_URL}/drugs`),
          fetch(`${API_URL}/microbes`),
        ]);

        if (!drugRes.ok || !microbeRes.ok) {
          throw new Error(
            "Failed to load drugs or microbes."
          );
        }

        const drugData = await drugRes.json();
        const microbeData = await microbeRes.json();

        setDrugs(drugData);
        setMicrobes(microbeData);
      } catch (err) {
        console.error(err);

        setError(
          "Unable to connect to the Drug-Microbe AI backend."
        );
      } finally {
        setLoadingData(false);
      }
    }

    loadData();
  }, []);

  // ============================================================
  // LOAD EVALUATION
  // ============================================================

  useEffect(() => {
    async function loadEvaluation() {
      try {
        setEvaluationLoading(true);

        const response = await fetch(
          `${API_URL}/evaluation`
        );

        if (!response.ok) {
          throw new Error(
            "Failed to load evaluation."
          );
        }

        const data = await response.json();

        setEvaluation(data);
      } catch (err) {
        console.error(err);

        setEvaluationError(
          "Unable to load model evaluation."
        );
      } finally {
        setEvaluationLoading(false);
      }
    }

    loadEvaluation();
  }, []);

  // ============================================================
  // LOAD DATASET STATS
  // ============================================================

  useEffect(() => {
    async function loadStats() {
      try {
        const response = await fetch(
          `${API_URL}/dataset-stats`
        );

        if (!response.ok) return;

        const data = await response.json();

        setDatasetStats(data);
      } catch (err) {
        console.error(
          "Dataset stats error:",
          err
        );
      }
    }

    loadStats();
  }, []);

  // ============================================================
  // LOAD MODEL COMPARISON
  // ============================================================

  useEffect(() => {
    async function loadComparison() {
      try {
        setComparisonLoading(true);
        setComparisonError("");

        const response = await fetch(
          `${API_URL}/comparison`
        );

        if (!response.ok) {
          throw new Error(
            "Comparison endpoint unavailable."
          );
        }

        const data = await response.json();

        setComparison(data);
      } catch (err) {
        console.error(err);

        setComparisonError(
          "GCN vs HaGAT comparison is not available yet."
        );
      } finally {
        setComparisonLoading(false);
      }
    }

    loadComparison();
  }, []);

  // ============================================================
  // SEARCH
  // ============================================================

  const filteredDrugs = useMemo(() => {
    const query = drugSearch.toLowerCase().trim();

    if (!query) {
      return drugs.slice(0, 10);
    }

    return drugs
      .filter((drug) =>
        drug.name.toLowerCase().includes(query)
      )
      .slice(0, 10);
  }, [drugs, drugSearch]);

  const filteredMicrobes = useMemo(() => {
    const query =
      microbeSearch.toLowerCase().trim();

    if (!query) {
      return microbes.slice(0, 10);
    }

    return microbes
      .filter((microbe) =>
        microbe.name
          .toLowerCase()
          .includes(query)
      )
      .slice(0, 10);
  }, [microbes, microbeSearch]);

  // ============================================================
  // SAVE HISTORY
  // ============================================================

  function saveHistory(
    drug: Item,
    microbe: Item,
    result: Prediction
  ) {
    let probability =
      typeof result.probability === "number"
        ? result.probability
        : 0;

    if (probability <= 1) {
      probability *= 100;
    }

    const interaction =
      result.interaction ??
      result.prediction
        ?.toLowerCase()
        .includes("interaction") ??
      false;

    const item: HistoryItem = {
      id: Date.now(),
      drug: drug.name,
      microbe: microbe.name,
      prediction:
        result.prediction ??
        (interaction
          ? "Interaction"
          : "No Interaction"),
      probability,
      timestamp:
        new Date().toLocaleString(),
    };

    setHistory((previous) => {
      const updated = [item, ...previous]
        .slice(0, 10);

      localStorage.setItem(
        "drugMicrobePredictionHistory",
        JSON.stringify(updated)
      );

      return updated;
    });
  }

  // ============================================================
  // PREDICT
  // ============================================================

  async function handlePrediction() {
    if (
      !selectedDrug ||
      !selectedMicrobe
    ) {
      setError(
        "Please select both a drug and a microorganism."
      );

      return;
    }

    try {
      setLoadingPrediction(true);
      setError("");
      setPrediction(null);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            drug_id: selectedDrug.id,
            microbe_id: selectedMicrobe.id,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Prediction failed."
        );
      }

      setPrediction(data);

      saveHistory(
        selectedDrug,
        selectedMicrobe,
        data
      );
    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Prediction failed."
      );
    } finally {
      setLoadingPrediction(false);
    }
  }

  // ============================================================
  // CLEAR HISTORY
  // ============================================================

  function clearHistory() {
    localStorage.removeItem(
      "drugMicrobePredictionHistory"
    );

    setHistory([]);
  }

  // ============================================================
  // PREDICTION VALUES
  // ============================================================

  let probabilityPercent = 0;

  if (
    prediction &&
    typeof prediction.probability ===
      "number"
  ) {
    probabilityPercent =
      prediction.probability <= 1
        ? prediction.probability * 100
        : prediction.probability;
  }

  const isInteraction =
    prediction?.interaction ??
    prediction?.prediction
      ?.toLowerCase()
      .includes("interaction") ??
    false;

  const predictionText =
    prediction?.prediction ??
    (isInteraction
      ? "Interaction"
      : "No Interaction");

  // ============================================================
  // FORMAT
  // ============================================================

  function percent(value: number) {
    return `${(value * 100).toFixed(2)}%`;
  }

  // ============================================================
  // COMPARISON METRICS
  // ============================================================

  const comparisonMetrics = [
    {
      name: "Accuracy",
      key: "accuracy" as keyof ComparisonMetrics,
    },
    {
      name: "Precision",
      key: "precision" as keyof ComparisonMetrics,
    },
    {
      name: "Recall",
      key: "recall" as keyof ComparisonMetrics,
    },
    {
      name: "F1 Score",
      key: "f1" as keyof ComparisonMetrics,
    },
    {
      name: "ROC-AUC",
      key: "roc_auc" as keyof ComparisonMetrics,
    },
  ];

  // ============================================================
  // UI
  // ============================================================

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto max-w-7xl px-6 py-10">

        {/* HEADER */}

        <header className="mb-10">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-300">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            AI-powered interaction prediction
          </div>

          <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
            DrugMicrobe AI
          </h1>

          <p className="mt-3 max-w-2xl text-slate-400">
            Predict potential drug–microorganism
            interactions using our trained HaGAT
            graph neural network.
          </p>

          <div className="mt-3 text-sm text-slate-500">
            Heterogeneous Graph Attention Network
          </div>
        </header>

        {/* DATASET OVERVIEW */}

        {datasetStats && (
          <section className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Dataset Records
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-300">
                {datasetStats.total_records?.toLocaleString() ?? "—"}
              </p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Unique Drugs
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-300">
                {datasetStats.unique_drugs?.toLocaleString() ?? "—"}
              </p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Unique Microbes
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-300">
                {datasetStats.unique_microbes?.toLocaleString() ?? "—"}
              </p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Graph Nodes
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-300">
                {(
                  (datasetStats.unique_drugs ?? 0) +
                  (datasetStats.unique_microbes ?? 0)
                ).toLocaleString()}
              </p>
            </div>

          </section>
        )}

        {/* PREDICTION */}

        <div className="grid gap-6 lg:grid-cols-2">

          {/* INPUT */}

          <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">

            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Prediction Engine
            </p>

            <h2 className="mt-2 text-2xl font-semibold">
              Drug–Microbe Interaction Prediction
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Enter the identifiers for a drug and
              microbe.
            </p>

            {/* DRUG */}

            <div className="mt-7">
              <label className="mb-2 block text-sm font-medium text-slate-300">
                Drug ID
              </label>

              <input
                value={
                  selectedDrug
                    ? selectedDrug.name
                    : drugSearch
                }
                onChange={(event) => {
                  setSelectedDrug(null);
                  setDrugSearch(
                    event.target.value
                  );
                }}
                placeholder={
                  loadingData
                    ? "Loading drugs..."
                    : "Search drug..."
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3.5 text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-400"
              />

              {!selectedDrug && (
                <div className="mt-2 max-h-52 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950">

                  {filteredDrugs.map((drug) => (
                    <button
                      key={drug.id}
                      onClick={() => {
                        setSelectedDrug(drug);
                        setDrugSearch("");
                      }}
                      className="block w-full border-b border-slate-800 px-4 py-3 text-left transition hover:bg-slate-800"
                    >
                      <div className="font-medium">
                        {drug.name}
                      </div>

                      <div className="mt-1 text-xs text-slate-500">
                        ID: {drug.id}
                      </div>
                    </button>
                  ))}

                  {filteredDrugs.length === 0 &&
                    !loadingData && (
                      <div className="p-4 text-sm text-slate-500">
                        No drug found.
                      </div>
                    )}
                </div>
              )}
            </div>

            {/* MICROBE */}

            <div className="mt-6">
              <label className="mb-2 block text-sm font-medium text-slate-300">
                Microbe ID
              </label>

              <input
                value={
                  selectedMicrobe
                    ? selectedMicrobe.name
                    : microbeSearch
                }
                onChange={(event) => {
                  setSelectedMicrobe(null);
                  setMicrobeSearch(
                    event.target.value
                  );
                }}
                placeholder={
                  loadingData
                    ? "Loading microbes..."
                    : "Search microorganism..."
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3.5 text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-400"
              />

              {!selectedMicrobe && (
                <div className="mt-2 max-h-52 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950">

                  {filteredMicrobes.map((microbe) => (
                    <button
                      key={microbe.id}
                      onClick={() => {
                        setSelectedMicrobe(
                          microbe
                        );
                        setMicrobeSearch("");
                      }}
                      className="block w-full border-b border-slate-800 px-4 py-3 text-left transition hover:bg-slate-800"
                    >
                      <div className="font-medium">
                        {microbe.name}
                      </div>

                      <div className="mt-1 text-xs text-slate-500">
                        ID: {microbe.id}
                      </div>
                    </button>
                  ))}

                  {filteredMicrobes.length === 0 &&
                    !loadingData && (
                      <div className="p-4 text-sm text-slate-500">
                        No microorganism found.
                      </div>
                    )}
                </div>
              )}
            </div>

            {/* SELECTED PAIR */}

            {(selectedDrug ||
              selectedMicrobe) && (
              <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-4">

                <p className="text-xs uppercase tracking-wider text-slate-500">
                  Selected Pair
                </p>

                {selectedDrug && (
                  <div className="mt-3 flex justify-between gap-4">
                    <span className="text-sm text-slate-500">
                      Drug
                    </span>

                    <span className="text-right text-sm font-medium text-cyan-300">
                      {selectedDrug.name}
                    </span>
                  </div>
                )}

                {selectedMicrobe && (
                  <div className="mt-2 flex justify-between gap-4">
                    <span className="text-sm text-slate-500">
                      Microbe
                    </span>

                    <span className="text-right text-sm font-medium text-cyan-300">
                      {selectedMicrobe.name}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* ERROR */}

            {error && (
              <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
                {error}
              </div>
            )}

            {/* BUTTON */}

            <button
              onClick={handlePrediction}
              disabled={
                loadingPrediction ||
                !selectedDrug ||
                !selectedMicrobe
              }
              className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-400 px-5 py-3.5 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loadingPrediction ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-950 border-t-transparent" />
                  Running HaGAT...
                </>
              ) : (
                "Predict Interaction"
              )}
            </button>
          </section>

          {/* RESULT */}

          <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">

            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Analysis Result
            </p>

            <h2 className="mt-2 text-2xl font-semibold">
              Prediction
            </h2>

            {!prediction ? (
              <div className="flex min-h-[450px] items-center justify-center text-center">
                <div>
                  <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-3xl border border-slate-800 bg-slate-950 text-4xl">
                    🧬
                  </div>

                  <p className="mt-5 font-medium text-slate-300">
                    Ready for analysis
                  </p>

                  <p className="mt-2 max-w-sm text-sm text-slate-500">
                    Select a drug and microorganism,
                    then run the HaGAT prediction.
                  </p>
                </div>
              </div>
            ) : (
              <div className="mt-7">

                <div className="rounded-3xl border border-slate-700 bg-slate-950 p-7 text-center">

                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                    Predicted Outcome
                  </p>

                  <h3
                    className={`mt-3 text-3xl font-bold ${
                      isInteraction
                        ? "text-emerald-400"
                        : "text-amber-400"
                    }`}
                  >
                    {predictionText}
                  </h3>

                  <div className="mt-8 text-6xl font-bold">
                    {probabilityPercent.toFixed(1)}%
                  </div>

                  <p className="mt-2 text-sm text-slate-500">
                    Interaction Probability
                  </p>

                  <div className="mt-7 h-3 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        isInteraction
                          ? "bg-emerald-400"
                          : "bg-amber-400"
                      }`}
                      style={{
                        width: `${Math.min(
                          Math.max(
                            probabilityPercent,
                            0
                          ),
                          100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="mt-5 rounded-2xl border border-slate-800 bg-slate-950 p-5">

                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Evaluated Biological Pair
                  </p>

                  <p className="mt-3 font-semibold">
                    💊 {selectedDrug?.name}
                  </p>

                  <p className="my-2 text-center text-slate-600">
                    ↓
                  </p>

                  <p className="font-semibold">
                    🦠 {selectedMicrobe?.name}
                  </p>
                </div>

                <div className="mt-5 grid grid-cols-3 gap-3">

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4 text-center">
                    <p className="text-xs text-slate-500">
                      Model
                    </p>

                    <p className="mt-2 font-semibold text-cyan-300">
                      HaGAT
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4 text-center">
                    <p className="text-xs text-slate-500">
                      Embedding
                    </p>

                    <p className="mt-2 font-semibold">
                      64D
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4 text-center">
                    <p className="text-xs text-slate-500">
                      Attention
                    </p>

                    <p className="mt-2 font-semibold">
                      4 Heads
                    </p>
                  </div>

                </div>
              </div>
            )}
          </section>
        </div>

        {/* ============================================================
            EVALUATION DASHBOARD
        ============================================================ */}

        <section className="mt-6 rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">

          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
            Model Performance
          </p>

          <h2 className="mt-2 text-2xl font-semibold">
            Evaluation Dashboard
          </h2>

          <p className="mt-2 text-sm text-slate-400">
            Performance metrics of the trained HaGAT
            model on the evaluation dataset.
          </p>

          {evaluationLoading && (
            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-8 text-center">
              <div className="mx-auto h-6 w-6 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />

              <p className="mt-4 text-sm text-slate-500">
                Loading evaluation metrics...
              </p>
            </div>
          )}

          {!evaluationLoading &&
            evaluationError && (
              <div className="mt-6 rounded-2xl border border-red-500/30 bg-red-500/10 p-5 text-sm text-red-300">
                {evaluationError}
              </div>
            )}

          {!evaluationLoading &&
            !evaluationError &&
            evaluation && (
              <div className="mt-7">

                {/* METRICS */}

                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">

                  {[
                    {
                      label: "Accuracy",
                      value:
                        evaluation.metrics.accuracy,
                    },
                    {
                      label: "Precision",
                      value:
                        evaluation.metrics.precision,
                    },
                    {
                      label: "Recall",
                      value:
                        evaluation.metrics.recall,
                    },
                    {
                      label: "F1 Score",
                      value:
                        evaluation.metrics.f1,
                    },
                    {
                      label: "ROC-AUC",
                      value:
                        evaluation.metrics.roc_auc,
                    },
                  ].map((metric) => (
                    <div
                      key={metric.label}
                      className="rounded-2xl border border-slate-800 bg-slate-950 p-5"
                    >
                      <p className="text-sm text-slate-500">
                        {metric.label}
                      </p>

                      <p className="mt-3 text-3xl font-bold text-cyan-300">
                        {(metric.value * 100).toFixed(2)}%
                      </p>

                      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="h-full rounded-full bg-cyan-400 transition-all duration-700"
                          style={{
                            width: `${Math.min(
                              Math.max(
                                metric.value * 100,
                                0
                              ),
                              100
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}

                </div>

                {/* MODEL INFORMATION */}

                <div className="mt-6 grid gap-4 md:grid-cols-3">

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">
                    <p className="text-xs uppercase tracking-wider text-slate-500">
                      Model
                    </p>

                    <p className="mt-2 text-lg font-semibold text-cyan-300">
                      {evaluation.model}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">
                    <p className="text-xs uppercase tracking-wider text-slate-500">
                      Dataset
                    </p>

                    <p className="mt-2 text-lg font-semibold">
                      {evaluation.dataset}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">
                    <p className="text-xs uppercase tracking-wider text-slate-500">
                      Architecture
                    </p>

                    <p className="mt-2 text-lg font-semibold">
                      Heterogeneous GAT
                    </p>
                  </div>

                </div>

                {/* SUMMARY */}

                <div className="mt-6 rounded-2xl border border-cyan-500/20 bg-cyan-500/5 p-5">

                  <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
                    Evaluation Summary
                  </p>

                  <p className="mt-3 leading-7 text-slate-300">
                    The trained HaGAT model achieved an
                    accuracy of{" "}
                    <span className="font-semibold text-cyan-300">
                      {percent(
                        evaluation.metrics.accuracy
                      )}
                    </span>
                    , precision of{" "}
                    <span className="font-semibold text-cyan-300">
                      {percent(
                        evaluation.metrics.precision
                      )}
                    </span>
                    , recall of{" "}
                    <span className="font-semibold text-cyan-300">
                      {percent(
                        evaluation.metrics.recall
                      )}
                    </span>
                    , F1 score of{" "}
                    <span className="font-semibold text-cyan-300">
                      {percent(
                        evaluation.metrics.f1
                      )}
                    </span>
                    , and ROC-AUC of{" "}
                    <span className="font-semibold text-cyan-300">
                      {percent(
                        evaluation.metrics.roc_auc
                      )}
                    </span>
                    .
                  </p>

                </div>

                {/* =====================================================
                    NEW: CONFUSION MATRIX + ROC CURVE
                ====================================================== */}

                <div className="mt-6 grid gap-6 lg:grid-cols-2">

                  {/* CONFUSION MATRIX */}

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">

                    <div className="mb-5">
                      <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
                        Classification Analysis
                      </p>

                      <h3 className="mt-2 text-xl font-semibold">
                        Confusion Matrix
                      </h3>

                      <p className="mt-2 text-sm text-slate-500">
                        Prediction classification results
                        of the evaluated model.
                      </p>
                    </div>

                    <div className="overflow-hidden rounded-xl border border-slate-800 bg-black/20">

                      <img
                        src={`${API_URL}/experiments/confusion_matrix.png`}
                        alt="HaGAT confusion matrix"
                        className="h-auto w-full object-contain"
                        loading="lazy"
                        onError={(event) => {
                          event.currentTarget.style.display =
                            "none";

                          const parent =
                            event.currentTarget
                              .parentElement;

                          if (parent) {
                            parent.innerHTML =
                              '<div class="flex min-h-[300px] items-center justify-center p-6 text-center text-sm text-slate-500">Confusion matrix image could not be loaded.</div>';
                          }
                        }}
                      />

                    </div>

                  </div>

                  {/* ROC CURVE */}

                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">

                    <div className="mb-5">
                      <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
                        Threshold Analysis
                      </p>

                      <h3 className="mt-2 text-xl font-semibold">
                        ROC Curve
                      </h3>

                      <p className="mt-2 text-sm text-slate-500">
                        Receiver operating characteristic
                        curve for the trained model.
                      </p>
                    </div>

                    <div className="overflow-hidden rounded-xl border border-slate-800 bg-black/20">

                      <img
                        src={`${API_URL}/experiments/roc_curve.png`}
                        alt="HaGAT ROC curve"
                        className="h-auto w-full object-contain"
                        loading="lazy"
                        onError={(event) => {
                          event.currentTarget.style.display =
                            "none";

                          const parent =
                            event.currentTarget
                              .parentElement;

                          if (parent) {
                            parent.innerHTML =
                              '<div class="flex min-h-[300px] items-center justify-center p-6 text-center text-sm text-slate-500">ROC curve image could not be loaded.</div>';
                          }
                        }}
                      />

                    </div>

                  </div>

                </div>

              </div>
            )}
        </section>

        {/* ============================================================
            GCN VS HAGAT
        ============================================================ */}

        <section className="mt-6 rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">

          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
            Research Comparison
          </p>

          <h2 className="mt-2 text-2xl font-semibold">
            GCN vs HaGAT
          </h2>

          <p className="mt-2 text-sm text-slate-400">
            Comparison of the heterogeneous graph
            attention model against a GCN baseline.
          </p>

          {comparisonLoading && (
            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-8 text-center text-slate-500">
              Loading model comparison...
            </div>
          )}

          {!comparisonLoading &&
            comparisonError && (
              <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-5 text-sm text-slate-500">
                {comparisonError}
              </div>
            )}

          {!comparisonLoading &&
            !comparisonError &&
            comparison?.models?.GCN &&
            comparison?.models?.HaGAT && (
              <div className="mt-7">

                {/* TABLE */}

                <div className="overflow-x-auto rounded-2xl border border-slate-800">

                  <table className="w-full min-w-[700px] text-left">

                    <thead className="bg-slate-950">
                      <tr className="border-b border-slate-800">

                        <th className="px-5 py-4">
                          Metric
                        </th>

                        <th className="px-5 py-4">
                          GCN
                        </th>

                        <th className="px-5 py-4 text-cyan-300">
                          HaGAT
                        </th>

                        <th className="px-5 py-4">
                          Difference
                        </th>

                      </tr>
                    </thead>

                    <tbody>

                      {comparisonMetrics.map(
                        (metric) => {
                          const gcn =
                            comparison.models!
                              .GCN![
                              metric.key
                            ];

                          const hagat =
                            comparison.models!
                              .HaGAT![
                              metric.key
                            ];

                          const difference =
                            (hagat - gcn) * 100;

                          return (
                            <tr
                              key={metric.name}
                              className="border-b border-slate-800 last:border-0"
                            >
                              <td className="px-5 py-4 font-medium">
                                {metric.name}
                              </td>

                              <td className="px-5 py-4 text-slate-400">
                                {percent(gcn)}
                              </td>

                              <td className="px-5 py-4 font-semibold text-cyan-300">
                                {percent(hagat)}
                              </td>

                              <td
                                className={`px-5 py-4 font-semibold ${
                                  difference >= 0
                                    ? "text-emerald-400"
                                    : "text-red-400"
                                }`}
                              >
                                {difference >= 0
                                  ? "+"
                                  : ""}
                                {difference.toFixed(
                                  2
                                )} pp
                              </td>
                            </tr>
                          );
                        }
                      )}

                    </tbody>
                  </table>
                </div>

                {/* BARS */}

                <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-5">

                  <p className="font-semibold">
                    Model Performance Overview
                  </p>

                  <div className="mt-6 space-y-6">

                    {comparisonMetrics.map(
                      (metric) => {
                        const gcn =
                          comparison.models!
                            .GCN![
                            metric.key
                          ] * 100;

                        const hagat =
                          comparison.models!
                            .HaGAT![
                            metric.key
                          ] * 100;

                        return (
                          <div
                            key={metric.name}
                          >

                            <div className="mb-2 flex justify-between">

                              <span className="text-sm text-slate-400">
                                {metric.name}
                              </span>

                              <span className="text-xs text-slate-500">
                                GCN{" "}
                                {gcn.toFixed(2)}%
                                {" · "}
                                HaGAT{" "}
                                {hagat.toFixed(2)}%
                              </span>

                            </div>

                            <div className="space-y-2">

                              <div className="flex items-center gap-3">

                                <span className="w-12 text-xs text-slate-500">
                                  GCN
                                </span>

                                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">

                                  <div
                                    className="h-full rounded-full bg-slate-600"
                                    style={{
                                      width: `${gcn}%`,
                                    }}
                                  />

                                </div>
                              </div>

                              <div className="flex items-center gap-3">

                                <span className="w-12 text-xs text-cyan-400">
                                  HaGAT
                                </span>

                                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">

                                  <div
                                    className="h-full rounded-full bg-cyan-400"
                                    style={{
                                      width: `${hagat}%`,
                                    }}
                                  />

                                </div>
                              </div>

                            </div>
                          </div>
                        );
                      }
                    )}

                  </div>
                </div>

              </div>
            )}

        </section>

        {/* ============================================================
            PREDICTION HISTORY
        ============================================================ */}

        <section className="mt-6 rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">

          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Research Activity
              </p>

              <h2 className="mt-2 text-2xl font-semibold">
                Prediction History
              </h2>

              <p className="mt-2 text-sm text-slate-400">
                Your latest predictions are stored
                locally in this browser.
              </p>
            </div>

            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="rounded-xl border border-red-500/30 px-4 py-2 text-sm text-red-400 transition hover:bg-red-500/10"
              >
                Clear History
              </button>
            )}

          </div>

          {history.length === 0 ? (
            <div className="mt-6 rounded-2xl border border-dashed border-slate-800 bg-slate-950 p-8 text-center">

              <p className="text-slate-400">
                No predictions yet.
              </p>

              <p className="mt-2 text-sm text-slate-600">
                Your predictions will appear here.
              </p>

            </div>
          ) : (
            <div className="mt-6 space-y-3">

              {history.map((item) => {
                const positive =
                  item.prediction
                    .toLowerCase()
                    .includes(
                      "interaction"
                    );

                return (
                  <div
                    key={item.id}
                    className="rounded-2xl border border-slate-800 bg-slate-950 p-4"
                  >

                    <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">

                      <div>

                        <div className="flex flex-wrap items-center gap-2">

                          <span className="font-semibold">
                            💊 {item.drug}
                          </span>

                          <span className="text-slate-600">
                            →
                          </span>

                          <span className="font-semibold">
                            🦠 {item.microbe}
                          </span>

                        </div>

                        <p className="mt-2 text-xs text-slate-600">
                          {item.timestamp}
                        </p>

                      </div>

                      <div className="flex items-center gap-5">

                        <div className="text-right">

                          <p
                            className={`text-sm font-semibold ${
                              positive
                                ? "text-emerald-400"
                                : "text-amber-400"
                            }`}
                          >
                            {item.prediction}
                          </p>

                          <p className="mt-1 text-xs text-slate-500">
                            Probability
                          </p>

                        </div>

                        <div className="min-w-[70px] text-right text-xl font-bold">
                          {item.probability.toFixed(
                            1
                          )}
                          %
                        </div>

                      </div>

                    </div>
                  </div>
                );
              })}

            </div>
          )}

        </section>

        {/* FOOTER */}

        <footer className="mt-10 border-t border-slate-800 pt-6 text-center text-sm text-slate-500">
          DrugMicrobe AI · Graph Neural Network Research Project
        </footer>

      </div>
    </main>
  );
}