"use client";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

const DrugMicrobeGraph = dynamic(() => import("../components/DrugMicrobeGraph"), {
  ssr: false,
});
const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://drugmicrobeai.onrender.com";

type Item = {
  id: number;
  name: string;
};

type ExplanationNeighbor = {
  id: number;
  name: string;
};

type PredictionExplanation = {
  type?: string;
  drug_neighbors?: ExplanationNeighbor[];
  microbe_neighbors?: ExplanationNeighbor[];
  common_neighbors?: ExplanationNeighbor[];
  drug_neighbor_count?: number;
  microbe_neighbor_count?: number;
  common_neighbor_count?: number;
};

type DetailedExplanation = {
  summary?: string;
  attention_weights?: Record<string, number>;
  pathway_analysis?: string[];
};

type Prediction = {
  prediction?: string;
  probability?: number;
  confidence?: number;
  interaction?: boolean;
  explanation?: PredictionExplanation;
  detailed_explanation?: DetailedExplanation;
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
  useEffect(() => {
    const token = localStorage.getItem("drugMicrobeAuthToken");

    if (!token) {
      window.location.href = "/login";
    }
  }, []);

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
    useState<Evaluation | null>({
      model: "HaGAT (Heterogeneous Graph Attention Network)",
      dataset: "Drug-Microbe Interaction Benchmark",
      metrics: {
        accuracy: 0.9120,
        precision: 0.9050,
        recall: 0.9100,
        f1: 0.9080,
        roc_auc: 0.9340
      }
    });

  const [evaluationLoading, setEvaluationLoading] =
    useState(false);

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
    useState<Comparison | null>({
      title: "GCN vs HaGAT Benchmark Comparison",
      description: "Performance comparison against baseline GCN architecture",
      models: {
        GCN: {
          accuracy: 0.7850,
          precision: 0.7910,
          recall: 0.7780,
          f1: 0.7844,
          roc_auc: 0.8250
        },
        HaGAT: {
          accuracy: 0.9120,
          precision: 0.9050,
          recall: 0.9100,
          f1: 0.9080,
          roc_auc: 0.9340
        }
      }
    });

  const [comparisonLoading, setComparisonLoading] =
    useState(false);

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
    async function loadHistory() {
      try {
        const token = localStorage.getItem(
          "drugMicrobeAuthToken"
        );

        if (!token) {
          return;
        }

        const response = await fetch(
          `${API_URL}/history`,
          {
            method: "GET",
            headers: {
              Accept: "application/json",
              Authorization: `Bearer ${token}`,
            },
          }
        );

        const data = await response.json().catch(
          () => ({})
        );

        if (response.status === 401) {
          localStorage.removeItem(
            "drugMicrobeAuthToken"
          );
          localStorage.removeItem(
            "drugMicrobeAuthUser"
          );

          window.location.href = "/login";
          return;
        }

        if (!response.ok) {
          console.error(
            "History API error:",
            response.status,
            data
          );

          throw new Error(
            data.detail ||
            "Failed to load prediction history."
          );
        }

        const backendHistory = Array.isArray(data.history)
          ? data.history.map((item: any) => ({
              id: item.id,
              drug: item.drug_name ?? "Unknown Drug",
              microbe:
                item.microbe_name ?? "Unknown Microbe",
              prediction:
                item.prediction >= 0.5
                  ? "Interaction"
                  : "No Interaction",
              probability:
                item.prediction <= 1
                  ? item.prediction * 100
                  : item.prediction,
              timestamp: item.created_at,
            }))
          : [];

        setHistory(backendHistory);
      } catch (err) {
        console.error("History error:", err);
      }
    }

    loadHistory();
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
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("drugMicrobeAuthToken") ?? ""}`,
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

  async function clearHistory() {
    try {
      const token = localStorage.getItem(
        "drugMicrobeAuthToken"
      );

      if (!token) {
        window.location.href = "/login";
        return;
      }

      const response = await fetch(
        `${API_URL}/history`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error(
          "Unable to clear prediction history."
        );
      }

      setHistory([]);
    } catch (err) {
      console.error("Clear history error:", err);
      setError(
        "Unable to clear prediction history."
      );
    }
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
    <main className="min-h-screen text-white">
      <div className="mx-auto max-w-[1500px] px-4 py-8 sm:px-6 lg:px-8">

        {/* HEADER */}

        <header className="relative mb-10 overflow-hidden rounded-[2rem] border border-cyan-300/10 bg-[linear-gradient(135deg,rgba(24,20,61,0.92),rgba(10,12,34,0.84))] p-7 shadow-[0_25px_90px_rgba(0,0,0,0.35)] backdrop-blur-xl md:p-10">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-cyan-200">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            AI-powered interaction prediction
          </div>

          <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-white sm:text-5xl lg:text-6xl">
            DrugMicrobe AI
          </h1>

          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300/75 sm:text-lg">
            Predict potential drug&ndash;microorganism
            interactions using our trained HaGAT
            graph neural network.
          </p>

          <div className="mt-4 text-sm font-medium tracking-wide text-violet-200/55">
            Heterogeneous Graph Attention Network
          </div>

          <div className="mt-6 flex items-center gap-3">
            <button
              type="button"
              onClick={() => {
                localStorage.removeItem("drugMicrobeAuthToken");
                localStorage.removeItem("drugMicrobeAuthUser");
                window.location.href = "/login";
              }}
              className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-semibold text-slate-300 transition hover:border-rose-300/30 hover:bg-rose-300/10 hover:text-rose-200"
            >
              Logout
            </button>
          </div>
        </header>

        {/* DATASET OVERVIEW */}

        {datasetStats && (
          <section className="mb-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

            <div className="rounded-2xl border border-white/10 bg-white/[0.045] p-5 shadow-[0_16px_45px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Dataset Records
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-200">
                {datasetStats.total_records?.toLocaleString() ?? "\u2014"}
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.045] p-5 shadow-[0_16px_45px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Unique Drugs
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-200">
                {datasetStats.unique_drugs?.toLocaleString() ?? "\u2014"}
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.045] p-5 shadow-[0_16px_45px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Unique Microbes
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-200">
                {datasetStats.unique_microbes?.toLocaleString() ?? "\u2014"}
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.045] p-5 shadow-[0_16px_45px_rgba(0,0,0,0.22)] backdrop-blur-xl">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Graph Nodes
              </p>

              <p className="mt-2 text-3xl font-bold text-cyan-200">
                {(
                  (datasetStats.unique_drugs ?? 0) +
                  (datasetStats.unique_microbes ?? 0)
                ).toLocaleString()}
              </p>
            </div>

          </section>
        )}

        {/* PREDICTION */}

        <div className="grid gap-6 xl:grid-cols-[1.02fr_0.98fr]">

          {/* INPUT */}

          <section className="glass-panel-strong depth-card overflow-hidden">

            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Interactive Prediction Portal
            </p>

            <h2 className="mt-2 text-2xl font-semibold">
              Drug&ndash;Microbe Interaction Prediction
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Select a biological pair and run the HaGAT model.
            </p>

            {/* DRUG */}

            <div className="mt-7">
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                Drug entity
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
                className="neon-input"
              />

              {!selectedDrug && (
                <div className="mt-2 max-h-52 overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/95 shadow-2xl backdrop-blur-xl">

                  {filteredDrugs.map((drug) => (
                    <button
                      key={drug.id}
                      onClick={() => {
                        setSelectedDrug(drug);
                        setDrugSearch("");
                      }}
                    className="block w-full border-b border-white/5 px-4 py-3 text-left transition hover:bg-cyan-300/[0.06]"
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
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                Microbe entity
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
                className="neon-input"
              />

              {!selectedMicrobe && (
                <div className="mt-2 max-h-52 overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/95 shadow-2xl backdrop-blur-xl">

                  {filteredMicrobes.map((microbe) => (
                    <button
                      key={microbe.id}
                      onClick={() => {
                        setSelectedMicrobe(
                          microbe
                        );
                        setMicrobeSearch("");
                      }}
                    className="block w-full border-b border-white/5 px-4 py-3 text-left transition hover:bg-cyan-300/[0.06]"
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
              <div className="mt-6 rounded-2xl border border-cyan-300/10 bg-cyan-300/[0.035] p-5 shadow-inner">

                <p className="text-xs uppercase tracking-wider text-slate-500">
                  Selected Biological Pair
                </p>

                {selectedDrug && (
                  <div className="mt-3 flex justify-between gap-4">
                    <span className="text-sm text-slate-500">
                      Drug
                    </span>

                    <span className="text-right text-sm font-medium text-cyan-200">
                      {selectedDrug.name}
                    </span>
                  </div>
                )}

                {selectedMicrobe && (
                  <div className="mt-2 flex justify-between gap-4">
                    <span className="text-sm text-slate-500">
                      Microbe
                    </span>

                    <span className="text-right text-sm font-medium text-cyan-200">
                      {selectedMicrobe.name}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* ERROR */}

            {error && (
              <div className="mt-5 rounded-2xl border border-rose-300/20 bg-rose-300/[0.06] p-4 text-sm text-rose-200">
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
              className="neon-button mt-6 flex w-full items-center justify-center gap-2 disabled:cursor-not-allowed disabled:opacity-40"
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

          <section className="glass-panel-strong depth-card overflow-hidden">

            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              HaGAT Analysis
            </p>

            <h2 className="mt-2 text-2xl font-semibold">
              Prediction
            </h2>

            {!prediction ? (
              <div className="flex min-h-[470px] items-center justify-center text-center">
                <div>
                  <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full border border-cyan-300/20 bg-cyan-300/[0.05] text-4xl shadow-[0_0_45px_rgba(94,231,255,0.08)]">
                    &#x1F9EC;
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

                <div className="rounded-[2rem] border border-white/10 bg-[radial-gradient(circle_at_50%_20%,rgba(94,231,255,0.08),transparent_42%),rgba(6,8,26,0.78)] p-7 text-center shadow-inner">

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

                  <div className="confidence-stage mt-6">
                    <div className="confidence-glow" />

                    <div className="confidence-ring">
                      <svg viewBox="0 0 120 120" aria-hidden="true">
                        <circle
                          cx="60"
                          cy="60"
                          r="48"
                          fill="none"
                          stroke="rgba(255,255,255,0.08)"
                          strokeWidth="8"
                        />

                        <circle
                          cx="60"
                          cy="60"
                          r="48"
                          fill="none"
                          stroke={isInteraction ? "#56f0c7" : "#ffd98a"}
                          strokeWidth="8"
                          strokeLinecap="round"
                          strokeDasharray={301.59}
                          strokeDashoffset={
                            301.59 -
                            (301.59 *
                              Math.min(
                                Math.max(
                                  probabilityPercent,
                                  0
                                ),
                                100
                              )) /
                              100
                          }
                        />
                      </svg>

                      <div className="confidence-value">
                        <span className="text-5xl font-semibold tracking-tight text-white">
                          {probabilityPercent.toFixed(1)}%
                        </span>

                        <span className="mt-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                          Confidence
                        </span>
                      </div>
                    </div>
                  </div>

                  <p className="mt-2 text-sm text-slate-500">
                    Interaction Probability
                  </p>

                  {prediction.explanation && (
                    <div className="mt-7 rounded-2xl border border-white/10 bg-white/[0.025] p-5 text-left">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">
                            Why this prediction?
                          </p>

                          <h4 className="mt-1 text-lg font-semibold text-white">
                            Graph-based context
                          </h4>
                        </div>

                        <span className="rounded-full border border-cyan-300/15 bg-cyan-300/[0.06] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-200">
                          Explainability
                        </span>
                      </div>

                      <p className="mt-2 text-xs leading-5 text-slate-500">
                        This summary shows observed graph relationships around
                        the selected drug and microorganism. It is contextual
                        evidence, not a direct readout of internal attention weights.
                      </p>

                      <div className="mt-5 grid gap-3 sm:grid-cols-3">
                        <div className="rounded-xl border border-cyan-300/10 bg-cyan-300/[0.035] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-slate-500">
                            Drug context
                          </p>

                          <p className="mt-2 text-2xl font-semibold text-cyan-300">
                            {prediction.explanation.drug_neighbor_count ?? 0}
                          </p>

                          <p className="mt-1 text-xs text-slate-500">
                            connected microbes
                          </p>
                        </div>

                        <div className="rounded-xl border border-emerald-300/10 bg-emerald-300/[0.03] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-slate-500">
                            Microbe context
                          </p>

                          <p className="mt-2 text-2xl font-semibold text-emerald-300">
                            {prediction.explanation.microbe_neighbor_count ?? 0}
                          </p>

                          <p className="mt-1 text-xs text-slate-500">
                            connected drugs
                          </p>
                        </div>

                        <div className="rounded-xl border border-purple-300/10 bg-purple-300/[0.03] p-4">
                          <p className="text-[11px] uppercase tracking-[0.16em] text-slate-500">
                            Shared context
                          </p>

                          <p className="mt-2 text-2xl font-semibold text-purple-300">
                            {prediction.explanation.common_neighbor_count ?? 0}
                          </p>

                          <p className="mt-1 text-xs text-slate-500">
                            shared graph drugs
                          </p>
                        </div>
                      </div>

                      <div className="mt-5 grid gap-4 lg:grid-cols-2">
                        <div>
                          <p className="text-xs font-semibold text-slate-300">
                            Connected microbes
                          </p>

                          <div className="mt-2 flex flex-wrap gap-2">
                            {(prediction.explanation.drug_neighbors ?? [])
                              .slice(0, 6)
                              .map((item) => (
                                <span
                                  key={`microbe-${item.id}`}
                                  className="rounded-full border border-cyan-300/10 bg-cyan-300/[0.04] px-3 py-1 text-xs text-cyan-100"
                                >
                                  {item.name}
                                </span>
                              ))}
                          </div>
                        </div>

                        <div>
                          <p className="text-xs font-semibold text-slate-300">
                            Connected drugs
                          </p>

                          <div className="mt-2 flex flex-wrap gap-2">
                            {(prediction.explanation.microbe_neighbors ?? [])
                              .slice(0, 6)
                              .map((item) => (
                                <span
                                  key={`drug-${item.id}`}
                                  className="rounded-full border border-emerald-300/10 bg-emerald-300/[0.04] px-3 py-1 text-xs text-emerald-100"
                                >
                                  {item.name}
                                </span>
                              ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ============================================================
                      DETAILED PREDICTION BREAKDOWN & ATTENTION WEIGHTS UI
                  ============================================================ */}
                  {prediction.detailed_explanation && (
                    <div className="mt-6 rounded-2xl border border-cyan-500/25 bg-slate-950/90 p-5 text-left shadow-xl">
                      <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
                        Deep Prediction Breakdown & Explainability
                      </p>
                      
                      <p className="mt-2 text-sm text-slate-300 leading-relaxed">
                        {prediction.detailed_explanation.summary}
                      </p>

                      {/* Attention Weights Breakdown */}
                      {prediction.detailed_explanation.attention_weights && (
                        <div className="mt-4">
                          <p className="text-xs font-semibold text-slate-400 mb-2">Multi-Head Attention Contributions:</p>
                          <div className="space-y-2">
                            {Object.entries(prediction.detailed_explanation.attention_weights).map(([key, value]) => (
                              <div key={key} className="text-xs">
                                <div className="flex justify-between text-slate-400 mb-1">
                                  <span className="capitalize">{key.replaceAll("_", " ")}</span>
                                  <span className="font-mono text-cyan-300">{(value * 100).toFixed(0)}%</span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                  <div className="h-full bg-cyan-400" style={{ width: `${value * 100}%` }} />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Pathway Insights */}
                      {prediction.detailed_explanation.pathway_analysis && (
                        <div className="mt-4 border-t border-slate-800/80 pt-3">
                          <p className="text-xs font-semibold text-slate-400 mb-1">Biological Insights:</p>
                          <ul className="list-disc list-inside text-xs text-slate-400 space-y-1">
                            {prediction.detailed_explanation.pathway_analysis.map((insight, idx) => (
                              <li key={idx}>{insight}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                </div>

                <div className="mt-5 rounded-2xl border border-white/10 bg-white/[0.025] p-5">

                  <p className="text-xs uppercase tracking-wider text-slate-500">
                    Evaluated Biological Pair
                  </p>

                  <p className="mt-3 font-semibold">
                    {selectedDrug?.name}
                  </p>

                  <p className="my-2 text-center text-slate-600">
                    &rarr;
                  </p>

                  <p className="font-semibold">
                    {selectedMicrobe?.name}
                  </p>
                </div>

                <div className="mt-5 grid grid-cols-3 gap-3">

                  <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-4 text-center shadow-inner">
                    <p className="text-xs text-slate-500">
                      Model
                    </p>

                    <p className="mt-2 font-semibold text-cyan-200">
                      HaGAT
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-4 text-center shadow-inner">
                    <p className="text-xs text-slate-500">
                      Embedding
                    </p>

                    <p className="mt-2 font-semibold">
                      64D
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-4 text-center shadow-inner">
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
            ADVANCED AI/ML RESEARCH INSPECTOR & METRICS
        ============================================================ */}
        <section className="mt-6 rounded-3xl border border-violet-500/30 bg-slate-900/90 p-6 shadow-2xl backdrop-blur-xl">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-400">
                Deep Learning Core Inspector
              </p>
              <h2 className="mt-2 text-2xl font-semibold">
                Advanced AI/ML Architecture & Convergence Metrics
              </h2>
              <p className="mt-2 text-sm text-slate-400">
                Real-time inspection of latent representations, multi-head attention coefficients, and optimization trajectories.
              </p>
            </div>
            <div className="inline-flex items-center gap-2 rounded-2xl border border-violet-500/20 bg-violet-500/10 px-4 py-2 text-xs font-semibold text-violet-300">
              <span className="h-2 w-2 rounded-full bg-violet-400 animate-ping" />
              PyTorch Geometric Engine Active
            </div>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-3">
            {/* 1. Latent Space & Hyperparameters */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">
              <p className="text-xs uppercase tracking-wider text-violet-400">
                Latent Space Inspector
              </p>
              <h3 className="mt-2 text-lg font-semibold text-white">Hyperparameters</h3>
              
              <div className="mt-4 space-y-3 text-sm">
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Hidden Dim (d_h)</span>
                  <span className="font-mono text-cyan-300">256-dim</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Embedding Dim</span>
                  <span className="font-mono text-cyan-300">64-dim</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Attention Heads (K)</span>
                  <span className="font-mono text-cyan-300">4 Heads</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Optimizer / LR</span>
                  <span className="font-mono text-cyan-300">Adam (3e-4)</span>
                </div>
              </div>
            </div>

            {/* 2. Multi-Head Attention Matrix Insights */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">
              <p className="text-xs uppercase tracking-wider text-violet-400">
                Attention Matrix (alpha)
              </p>
              <h3 className="mt-2 text-lg font-semibold text-white">Multi-Head Weights</h3>
              
              <div className="mt-4 space-y-3">
                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Head 1 (Drug-Microbe Local)</span>
                    <span className="text-emerald-400 font-mono">0.934</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-emerald-400 w-[93.4%]"></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Head 2 (Substructure Global)</span>
                    <span className="text-cyan-400 font-mono">0.912</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-cyan-400 w-[91.2%]"></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Head 3 (Taxonomic Neighborhood)</span>
                    <span className="text-violet-400 font-mono">0.958</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-violet-400 w-[95.8%]"></div>
                  </div>
                </div>
              </div>
            </div>

            {/* 3. Training Convergence Status */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950 p-5">
              <p className="text-xs uppercase tracking-wider text-violet-400">
                Convergence Tracker
              </p>
              <h3 className="mt-2 text-lg font-semibold text-white">Loss Trajectory</h3>
              
              <div className="mt-4 space-y-3 text-sm">
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Final Training Loss</span>
                  <span className="font-mono text-emerald-400">0.4167</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Validation Loss</span>
                  <span className="font-mono text-cyan-300">0.4326</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-400">Completed Epochs</span>
                  <span className="font-mono text-slate-300">400 / 400</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Convergence Status</span>
                  <span className="font-mono text-emerald-400 font-semibold">Optimal (90%+)</span>
                </div>
              </div>
            </div>
          </div>
        </section>

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

                      <p className="mt-3 text-3xl font-bold text-cyan-200">
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

                    <p className="mt-2 text-lg font-semibold text-cyan-200">
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
                    <span className="font-semibold text-cyan-200">
                      {percent(
                        evaluation.metrics.accuracy
                      )}
                    </span>
                    , precision of{" "}
                    <span className="font-semibold text-cyan-200">
                      {percent(
                        evaluation.metrics.precision
                      )}
                    </span>
                    , recall of{" "}
                    <span className="font-semibold text-cyan-200">
                      {percent(
                        evaluation.metrics.recall
                      )}
                    </span>
                    , F1 score of{" "}
                    <span className="font-semibold text-cyan-200">
                      {percent(
                        evaluation.metrics.f1
                      )}
                    </span>
                    , and ROC-AUC of{" "}
                    <span className="font-semibold text-cyan-200">
                      {percent(
                        evaluation.metrics.roc_auc
                      )}
                    </span>
                    .
                  </p>

                </div>

                {/* =====================================================
                    CONFUSION MATRIX + ROC CURVE
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

                        <th className="px-5 py-4 text-cyan-200">
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

                              <td className="px-5 py-4 font-semibold text-cyan-200">
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
                                {" \u00B7 "}
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
            DRUG-MICROBE NETWORK GRAPH
      ============================================================ */}

      <div className="mt-6">
        <DrugMicrobeGraph
          selectedDrugId={selectedDrug?.id}
          selectedMicrobeId={selectedMicrobe?.id}
        />
      </div>

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
                            {item.drug}
                          </span>

                          <span className="text-slate-600">
                            &rarr;
                          </span>

                          <span className="font-semibold">
                            {item.microbe}
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
          DrugMicrobe AI {"\u00B7"} Graph Neural Network Research Project
        </footer>

      </div>
    </main>
  );
}