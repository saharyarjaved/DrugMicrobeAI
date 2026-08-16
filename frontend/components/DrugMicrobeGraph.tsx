"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://drugmicrobeai.onrender.com";

type GraphNode = {
  id: string;
  type: "drug" | "microbe";
  name: string;
  x?: number;
  y?: number;
};

type GraphLink = {
  source: string | GraphNode;
  target: string | GraphNode;
};

type GraphStatistics = {
  drug_nodes?: number;
  microbe_nodes?: number;
  total_nodes?: number;
  interactions?: number;
};

type GraphData = {
  nodes: GraphNode[];
  links: GraphLink[];
  statistics?: GraphStatistics;
};

function getNodeId(value: string | GraphNode) {
  return typeof value === "string" ? value : value.id;
}

export default function DrugMicrobeGraph({
  selectedDrugId,
  selectedMicrobeId,
}: {
  selectedDrugId?: number;
  selectedMicrobeId?: number;
}) {
  const graphRef = useRef<any>(null);

  const [graphData, setGraphData] = useState<GraphData>({
    nodes: [],
    links: [],
  });

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [search, setSearch] = useState("");
  const [selectedNode, setSelectedNode] =
    useState<GraphNode | null>(null);

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

  const matchingNodes = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return [];
    }

    return graphData.nodes
      .filter((node) =>
        node.name.toLowerCase().includes(query)
      )
      .slice(0, 8);
  }, [search, graphData.nodes]);

  const highlightedNodeIds = useMemo(() => {
    const ids = new Set<string>();

    // Highlight pair selected from prediction panel
    if (
      typeof selectedDrugId === "number" &&
      typeof selectedMicrobeId === "number"
    ) {
      ids.add(`drug-${selectedDrugId}`);
      ids.add(`microbe-${selectedMicrobeId}`);
    }

    if (selectedNode) {
      ids.add(selectedNode.id);

      graphData.links.forEach((link) => {
        const sourceId = getNodeId(link.source);
        const targetId = getNodeId(link.target);

        if (sourceId === selectedNode.id) {
          ids.add(targetId);
        }

        if (targetId === selectedNode.id) {
          ids.add(sourceId);
        }
      });
    }

    return ids;
  }, [selectedNode, graphData.links]);

  const highlightedLinks = useMemo(() => {
    const links = new Set<string>();

    graphData.links.forEach((link, index) => {
      const sourceId = getNodeId(link.source);
      const targetId = getNodeId(link.target);

      // Highlight selected node connections
      if (selectedNode) {
        if (
          sourceId === selectedNode.id ||
          targetId === selectedNode.id
        ) {
          links.add(String(index));
        }
      }

      // Highlight selected prediction pair
      if (
        typeof selectedDrugId === "number" &&
        typeof selectedMicrobeId === "number"
      ) {
        const isPair =
          (sourceId === `drug-${selectedDrugId}` &&
            targetId === `microbe-${selectedMicrobeId}`) ||
          (sourceId === `microbe-${selectedMicrobeId}` &&
            targetId === `drug-${selectedDrugId}`);

        if (isPair) {
          links.add(String(index));
        }
      }
    });

    return links;
  }, [
    selectedNode,
    selectedDrugId,
    selectedMicrobeId,
    graphData.links,
  ]);

  const selectNode = (node: GraphNode) => {
    setSelectedNode(node);

    setTimeout(() => {
      if (
        graphRef.current &&
        typeof node.x === "number" &&
        typeof node.y === "number"
      ) {
        graphRef.current.centerAt(
          node.x,
          node.y,
          600
        );

        graphRef.current.zoom(5, 600);
      }
    }, 100);
  };

  useEffect(() => {
  if (
    typeof selectedDrugId !== "number" ||
    typeof selectedMicrobeId !== "number" ||
    !graphRef.current
  ) {
    return;
  }

  const drugNode = graphData.nodes.find(
    (node) => node.id === `drug-${selectedDrugId}`
  );

  const microbeNode = graphData.nodes.find(
    (node) => node.id === `microbe-${selectedMicrobeId}`
  );

  if (
    !drugNode ||
    !microbeNode ||
    typeof drugNode.x !== "number" ||
    typeof drugNode.y !== "number" ||
    typeof microbeNode.x !== "number" ||
    typeof microbeNode.y !== "number"
  ) {
    return;
  }

  const centerX = (drugNode.x + microbeNode.x) / 2;
  const centerY = (drugNode.y + microbeNode.y) / 2;

  const distance = Math.sqrt(
    Math.pow(microbeNode.x - drugNode.x, 2) +
    Math.pow(microbeNode.y - drugNode.y, 2)
  );

  const zoomLevel =
    distance > 250
      ? 2.5
      : distance > 100
        ? 3.5
        : 5;

  graphRef.current.centerAt(
    centerX,
    centerY,
    800
  );

  graphRef.current.zoom(
    zoomLevel,
    800
  );
}, [
  selectedDrugId,
  selectedMicrobeId,
  graphData.nodes,
]);
const selectedInteractionCount = useMemo(() => {
    if (!selectedNode) {
      return 0;
    }

    return graphData.links.filter((link) => {
      const sourceId = getNodeId(link.source);
      const targetId = getNodeId(link.target);

      return (
        sourceId === selectedNode.id ||
        targetId === selectedNode.id
      );
    }).length;
  }, [selectedNode, graphData.links]);

  if (loading) {
    return (
      <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
          Heterogeneous Network
        </p>

        <h2 className="mt-2 text-2xl font-semibold text-white">
          Drug-Microbe Interaction Graph
        </h2>

        <p className="mt-4 text-sm text-slate-400">
          Loading Drug-Microbe graph...
        </p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-3xl border border-red-900/50 bg-slate-900/80 p-6">
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
    <section className="overflow-hidden rounded-[2rem] border border-cyan-300/10 bg-[#080a1d] shadow-[0_28px_90px_rgba(0,0,0,0.35)]">

      {/* HEADER */}
      <div className="border-b border-white/10 bg-[#0a0c24] p-6">

        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">

          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Heterogeneous Network
            </p>

            <h2 className="mt-2 text-2xl font-semibold text-white">
              Drug-Microbe Interaction Graph
            </h2>

            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Explore relationships between drugs and microorganisms.
              Search or select a node to inspect its interaction network.
            </p>
          </div>

          {/* SEARCH */}
          <div className="relative w-full lg:w-[360px]">

            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search drug or microorganism..."
              className="neon-input w-full text-sm"
            />

            {matchingNodes.length > 0 && (
              <div className="absolute left-0 right-0 top-full z-50 mt-2 overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">

                {matchingNodes.map((node) => (
                  <button
                    key={node.id}
                    onClick={() => {
                      selectNode(node);
                      setSearch(node.name);
                    }}
                    className="flex w-full items-center gap-3 border-b border-slate-800 px-4 py-3 text-left transition hover:bg-slate-800"
                  >

                    <span
                      className={`h-3 w-3 rounded-full ${
                        node.type === "drug"
                          ? "bg-cyan-400"
                          : "bg-emerald-400"
                      }`}
                    />

                    <div className="min-w-0">
                      <p className="truncate text-sm text-white">
                        {node.name}
                      </p>

                      <p className="text-xs text-slate-500">
                        {node.type === "drug"
                          ? "Drug"
                          : "Microorganism"}
                      </p>
                    </div>

                  </button>
                ))}

              </div>
            )}

          </div>

        </div>

        {/* STATISTICS */}
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">

          <div className="depth-card rounded-2xl border border-white/10 bg-white/[0.035] px-4 py-3 text-center">
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

          <div className="depth-card rounded-2xl border border-white/10 bg-white/[0.035] px-4 py-3 text-center">
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

          <div className="depth-card rounded-2xl border border-white/10 bg-white/[0.035] px-4 py-3 text-center">
            <p className="text-xs text-slate-500">
              Nodes
            </p>

            <p className="mt-1 text-lg font-bold text-white">
              {graphData.statistics?.total_nodes ??
                graphData.nodes.length}
            </p>
          </div>

          <div className="depth-card rounded-2xl border border-white/10 bg-white/[0.035] px-4 py-3 text-center">
            <p className="text-xs text-slate-500">
              Interactions
            </p>

            <p className="mt-1 text-lg font-bold text-purple-300">
              {graphData.statistics?.interactions ??
                graphData.links.length}
            </p>
          </div>

        </div>

      </div>

      {/* LEGEND */}
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
          Drag nodes - Scroll to zoom - Click a node
        </span>

      </div>

      {/* SELECTED NODE */}
      {selectedNode && (
        <div className="border-b border-cyan-300/10 bg-cyan-300/[0.045] px-6 py-4 backdrop-blur-xl">

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">

            <div>
              <p className="text-xs uppercase tracking-wider text-cyan-400">
                Selected Node
              </p>

              <p className="mt-1 text-lg font-semibold text-white">
                {selectedNode.name}
              </p>

              <p className="text-xs text-slate-500">
                {selectedNode.type === "drug"
                  ? "Drug"
                  : "Microorganism"}{" "}
                ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· ID: {selectedNode.id}
              </p>
            </div>

            <div className="flex items-center gap-3">

              <div className="depth-card rounded-xl border border-cyan-300/10 bg-white/[0.035] px-4 py-2 text-center">
                <p className="text-xs text-slate-500">
                  Connected interactions
                </p>

                <p className="text-lg font-bold text-cyan-300">
                  {selectedInteractionCount}
                </p>
              </div>

              <button
                onClick={() => setSelectedNode(null)}
                className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-slate-300 transition hover:border-cyan-300/20 hover:bg-cyan-300/[0.05]"
              >
                Clear
              </button>

            </div>

          </div>

        </div>
      )}

      {/* GRAPH */}
      <div className="relative h-[600px] w-full overflow-hidden bg-[#050619] isolate">

        <div className="pointer-events-none absolute inset-0 opacity-40">
          <div className="absolute -left-20 top-10 h-72 w-72 rounded-full bg-cyan-400/10 blur-[90px]" />
          <div className="absolute right-0 top-24 h-80 w-80 rounded-full bg-violet-400/10 blur-[100px]" />
          <div className="absolute bottom-0 left-1/3 h-64 w-64 rounded-full bg-emerald-400/5 blur-[90px]" />
        </div>

        <div
          className="pointer-events-none absolute inset-0 opacity-[0.08]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(148,163,184,0.35) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.35) 1px, transparent 1px)",
            backgroundSize: "42px 42px",
          }}
        />

        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeId="id"

          nodeLabel={(node: any) =>
            `${node.type === "drug" ? "Drug" : "Microbe"}: ${node.name}`
          }

          linkColor={(link: any) => {
            const index =
              graphData.links.indexOf(link);

            return highlightedLinks.has(String(index))
              ? "#22d3ee"
              : selectedNode ||
                (typeof selectedDrugId === "number" &&
                  typeof selectedMicrobeId === "number")
                ? "rgba(100,116,139,0.035)"
                : "rgba(100,116,139,0.28)";
          }}

          linkWidth={(link: any) => {
            const index =
              graphData.links.indexOf(link);

            return highlightedLinks.has(String(index))
              ? 4
              : selectedNode ||
                (typeof selectedDrugId === "number" &&
                  typeof selectedMicrobeId === "number")
                ? 0.4
                : 1;
          }}

          linkCanvasObject={(link: any, ctx: CanvasRenderingContext2D) => {
            const index =
              graphData.links.indexOf(link);

            if (!highlightedLinks.has(String(index))) {
              return;
            }

            const source = link.source as any;
            const target = link.target as any;

            if (
              typeof source?.x !== "number" ||
              typeof source?.y !== "number" ||
              typeof target?.x !== "number" ||
              typeof target?.y !== "number"
            ) {
              return;
            }

            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const length = Math.sqrt(
              dx * dx + dy * dy
            );

            if (!length) {
              return;
            }

            const ux = dx / length;
            const uy = dy / length;

            // Soft outer glow
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.lineTo(target.x, target.y);

            ctx.strokeStyle = "rgba(94,231,255,0.22)";
            ctx.lineWidth = 10;
            ctx.shadowColor = "rgba(94,231,255,0.65)";
            ctx.shadowBlur = 18;
            ctx.stroke();

            // Bright interaction beam
            ctx.beginPath();
            ctx.moveTo(
              source.x + ux * 6,
              source.y + uy * 6
            );
            ctx.lineTo(
              target.x - ux * 6,
              target.y - uy * 6
            );

            ctx.strokeStyle = "#5ee7ff";
            ctx.lineWidth = 3;
            ctx.shadowColor = "rgba(94,231,255,0.95)";
            ctx.shadowBlur = 10;
            ctx.stroke();

            // Small center pulse
            const centerX =
              (source.x + target.x) / 2;
            const centerY =
              (source.y + target.y) / 2;

            ctx.beginPath();
            ctx.arc(
              centerX,
              centerY,
              4.5,
              0,
              2 * Math.PI
            );

            ctx.fillStyle = "#ffffff";
            ctx.shadowColor = "rgba(255,255,255,0.9)";
            ctx.shadowBlur = 12;
            ctx.fill();

            ctx.restore();
          }}

          backgroundColor="#050619"

          nodeRelSize={5}

          cooldownTicks={100}

          d3VelocityDecay={0.35}

          onNodeClick={(node: any) => {
            selectNode(node);
          }}

          nodeCanvasObject={(
            node: any,
            ctx: CanvasRenderingContext2D,
            globalScale: number
          ) => {

            const isDrug =
              node.type === "drug";

            const isPredictionDrug =
              typeof selectedDrugId === "number" &&
              node.id === `drug-${selectedDrugId}`;

            const isPredictionMicrobe =
              typeof selectedMicrobeId === "number" &&
              node.id === `microbe-${selectedMicrobeId}`;

            const isPredictionPair =
              isPredictionDrug ||
              isPredictionMicrobe;

            const isSelectedNode =
              node.id === selectedNode?.id;

            const isHighlighted =
              !selectedNode &&
              !isPredictionPair
                ? true
                : highlightedNodeIds.has(node.id) ||
                  isPredictionPair;

            const color = isDrug
              ? "#5ee7ff"
              : "#56f0c7";

            const radius =
              isSelectedNode
                ? 8
                : isPredictionPair
                  ? 8
                  : isDrug
                    ? 4
                    : 5;

            // Neon depth glow
            ctx.save();

            if (
              isPredictionPair ||
              isSelectedNode
            ) {
              ctx.shadowColor = isDrug
                ? "rgba(94,231,255,0.75)"
                : "rgba(86,240,199,0.70)";
              ctx.shadowBlur = isSelectedNode ? 26 : 20;
            } else {
              ctx.shadowColor = isDrug
                ? "rgba(94,231,255,0.30)"
                : "rgba(86,240,199,0.26)";
              ctx.shadowBlur = 10;
            }

            ctx.beginPath();
            ctx.arc(
              node.x,
              node.y,
              radius,
              0,
              2 * Math.PI
            );

            ctx.fillStyle = isHighlighted
              ? color
              : "rgba(71,85,105,0.18)";

            ctx.fill();

            // Small specular highlight gives the node a 3D feel
            if (
              isHighlighted ||
              isPredictionPair ||
              isSelectedNode
            ) {
              ctx.shadowBlur = 0;
              ctx.beginPath();
              ctx.arc(
                node.x - radius * 0.28,
                node.y - radius * 0.28,
                Math.max(radius * 0.28, 1.2),
                0,
                2 * Math.PI
              );
              ctx.fillStyle = "rgba(255,255,255,0.72)";
              ctx.fill();
            }

            ctx.restore();

            // Main node
            ctx.beginPath();

            ctx.arc(
              node.x,
              node.y,
              radius,
              0,
              2 * Math.PI
            );

            ctx.fillStyle = isHighlighted
              ? color
              : "rgba(71,85,105,0.18)";

            ctx.fill();

            // White outline for selected/prediction nodes
            if (
              isSelectedNode ||
              isPredictionPair
            ) {
              ctx.strokeStyle = "#ffffff";
              ctx.lineWidth = 2;
              ctx.stroke();
            }

            // Labels
            if (
              globalScale >= 2.0 ||
              isPredictionPair ||
              isSelectedNode
            ) {

              const label =
                String(node.name);

              const fontSize =
                (isPredictionPair ||
                  isSelectedNode
                  ? 12
                  : 10) /
                globalScale;

              ctx.font =
                `${fontSize}px Sans-Serif`;

              ctx.textAlign = "center";
              ctx.textBaseline = "middle";

              ctx.fillStyle =
                isPredictionPair ||
                isSelectedNode
                  ? "#ffffff"
                  : isHighlighted
                    ? "#e2e8f0"
                    : "rgba(148,163,184,0.18)";

              const shortLabel =
                label.length > 30
                  ? `${label.substring(0, 30)}...`
                  : label;

              ctx.fillText(
                shortLabel,
                node.x,
                node.y + radius + 8
              );
            }
          }}        />

        {/* GRAPH INFO */}
        <div className="pointer-events-none absolute bottom-4 left-4 rounded-2xl border border-white/10 bg-slate-950/85 px-4 py-3 shadow-2xl backdrop-blur-xl">

          <p className="text-xs text-slate-500">
            Network
          </p>

          <p className="mt-1 text-sm font-medium text-slate-300">
            {graphData.nodes.length.toLocaleString()} nodes ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â·{" "}
            {graphData.links.length.toLocaleString()} edges
          </p>

        </div>

      </div>

    </section>
  );
}

