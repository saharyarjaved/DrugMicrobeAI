"use client";

import { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_URL = "http://127.0.0.1:8000";

type GraphNode = {
  id: string;
  type: "drug" | "microbe";
  name: string;
};

type GraphLink = {
  source: string | GraphNode;
  target: string | GraphNode;
};

type GraphData = {
  nodes: GraphNode[];
  links: GraphLink[];
  statistics?: {
    drug_nodes: number;
    microbe_nodes: number;
    total_nodes: number;
    interactions: number;
  };
};

export default function DrugMicrobeGraph() {
  const graphRef = useRef<any>(null);

  const [graphData, setGraphData] = useState<GraphData>({
    nodes: [],
    links: [],
  });

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadGraph() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(`${API_URL}/graph`);

        if (!response.ok) {
          throw new Error(
            `Graph API failed with status ${response.status}`
          );
        }

        const data = await response.json();

        setGraphData({
      nodes: Array.isArray(data.nodes) ? data.nodes : [],
      links: Array.isArray(data.links)
        ? data.links
        : Array.isArray(data.edges)
          ? data.edges
          : [],
      statistics: data.statistics ?? data.stats,
    });
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to load graph."
        );
      } finally {
        setLoading(false);
      }
    }

    loadGraph();
  }, []);

  if (loading) {
    return (
      <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
        <div className="flex min-h-[500px] items-center justify-center">
          <div className="text-center">
            <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />

            <p className="mt-4 text-sm text-slate-400">
              Loading Drugâ€“Microbe graph...
            </p>
          </div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-3xl border border-red-900/50 bg-slate-900/80 p-6 shadow-2xl">
        <h2 className="text-xl font-semibold text-red-400">
          Graph unavailable
        </h2>

        <p className="mt-2 text-sm text-slate-400">
          {error}
        </p>

        <p className="mt-4 text-xs text-slate-600">
          Make sure the FastAPI backend is running on
          127.0.0.1:8000.
        </p>
      </section>
    );
  }

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/80 shadow-2xl">
      {/* Header */}
      <div className="border-b border-slate-800 p-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Heterogeneous Network
            </p>

            <h2 className="mt-2 text-2xl font-semibold text-white">
              Drugâ€“Microbe Interaction Graph
            </h2>

            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Interactive visualization of drugs, microorganisms,
              and their known interaction relationships.
            </p>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-center">
              <p className="text-xs text-slate-500">
                Drugs
              </p>

              <p className="mt-1 text-lg font-bold text-cyan-300">
                {graphData.statistics?.drug_nodes ??
                  graphData.nodes.filter(
                    (node) => node.type === "drug"
                  ).length}
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-center">
              <p className="text-xs text-slate-500">
                Microbes
              </p>

              <p className="mt-1 text-lg font-bold text-emerald-300">
                {graphData.statistics?.microbe_nodes ??
                  graphData.nodes.filter(
                    (node) => node.type === "microbe"
                  ).length}
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-center">
              <p className="text-xs text-slate-500">
                Nodes
              </p>

              <p className="mt-1 text-lg font-bold text-white">
                {graphData.statistics?.total_nodes ??
                  graphData.nodes.length}
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-center">
              <p className="text-xs text-slate-500">
                Interactions
              </p>

              <p className="mt-1 text-lg font-bold text-purple-300">
                {graphData.statistics?.interactions ??
                  (graphData.links?.length ?? 0)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-5 border-b border-slate-800 px-6 py-4 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-cyan-400" />
          Drug
        </div>

        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-emerald-400" />
          Microbe
        </div>

        <div className="flex items-center gap-2">
          <span className="h-px w-5 bg-slate-500" />
          Interaction
        </div>

        <span className="ml-auto">
          Drag nodes Â· Scroll to zoom Â· Click for details
        </span>
      </div>

      {/* Graph */}
      <div className="relative h-[600px] w-full bg-slate-950">
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeId="id"
          nodeLabel={(node: any) =>
            `${node.type === "drug" ? "Drug" : "Microbe"}: ${
              node.name
            }`
          }
          linkColor={() => "rgba(100, 116, 139, 0.28)"}
          linkWidth={1}
          linkDirectionalParticles={0}
          backgroundColor="#020617"
          nodeRelSize={5}
          cooldownTicks={100}
          d3VelocityDecay={0.35}
          onNodeClick={(node: any) => {
            if (graphRef.current) {
              graphRef.current.centerAt(
                node.x,
                node.y,
                500
              );

              graphRef.current.zoom(4, 500);
            }
          }}
          nodeCanvasObject={(
            node: any,
            ctx: CanvasRenderingContext2D,
            globalScale: number
          ) => {
            const isDrug = node.type === "drug";

            const color = isDrug
              ? "#22d3ee"
              : "#34d399";

            const radius = isDrug ? 4 : 5;

            ctx.beginPath();

            ctx.arc(
              node.x,
              node.y,
              radius,
              0,
              2 * Math.PI
            );

            ctx.fillStyle = color;
            ctx.fill();

            ctx.strokeStyle = isDrug
              ? "rgba(34,211,238,0.35)"
              : "rgba(52,211,153,0.35)";

            ctx.lineWidth = 1;

            ctx.stroke();

            /*
             * Only display labels when zoomed in.
             * This prevents 1,574 labels from covering
             * the entire graph.
             */
            if (globalScale >= 2.5) {
              const label = String(node.name);

              const fontSize = 10 / globalScale;

              ctx.font = `${fontSize}px Sans-Serif`;

              ctx.textAlign = "center";
              ctx.textBaseline = "middle";

              ctx.fillStyle = "#e2e8f0";

              ctx.fillText(
                label.length > 30
                  ? `${label.substring(0, 30)}...`
                  : label,
                node.x,
                node.y + radius + 4
              );
            }
          }}
        />

        {/* Graph info */}
        <div className="pointer-events-none absolute bottom-4 left-4 rounded-xl border border-slate-800 bg-slate-900/90 px-4 py-3 backdrop-blur">
          <p className="text-xs text-slate-500">
            Network
          </p>

          <p className="mt-1 text-sm font-medium text-slate-300">
            {graphData.nodes.length.toLocaleString()} nodes Â·{" "}
            {(graphData.links?.length ?? 0).toLocaleString()} edges
          </p>
        </div>
      </div>
    </section>
  );
}

