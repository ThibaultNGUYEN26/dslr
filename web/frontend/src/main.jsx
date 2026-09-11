import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Calculator,
  ChevronLeft,
  ChevronRight,
  Database,
  Grid3X3,
  LineChart,
  RefreshCw,
  ScatterChart,
  X,
} from "lucide-react";
import "./styles.css";

function formatStatistic(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return Number(value).toFixed(6);
}

function formatModelTime(value) {
  if (!value) {
    return "unknown time";
  }
  return new Date(value).toLocaleString();
}

const LOSS_COLORS = {
  Average: "#182026",
  Batch: "#1f4e8c",
  Stochastic: "#b31b1b",
  "Mini-batch": "#1f7a4d",
  Gryffindor: "#b31b1b",
  Hufflepuff: "#d6a600",
  Ravenclaw: "#1f4e8c",
  Slytherin: "#1f7a4d",
};

function LossChart({ data, series, title, combined = false, onClick = null }) {
  const width = 1000;
  const height = 440;
  const margin = { top: 24, right: 28, bottom: 56, left: 78 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const allValues = series.flatMap((item) => item.values);
  const minimum = Math.min(...allValues);
  const maximum = Math.max(...allValues);
  const lossRange = maximum - minimum || 1;
  const epochCount = data.epochs;

  function xPosition(index) {
    return margin.left + (index / Math.max(1, epochCount - 1)) * plotWidth;
  }

  function yPosition(value) {
    return margin.top + ((maximum - value) / lossRange) * plotHeight;
  }

  const yTicks = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    return maximum - ratio * lossRange;
  });
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) =>
    Math.round(1 + ratio * Math.max(0, epochCount - 1))
  );

  const Element = onClick ? "button" : "section";

  return (
    <Element
      className={`${combined ? "loss-panel combined" : "loss-panel"}${onClick ? " loss-card" : ""}`}
      aria-label={`${title} chart`}
      onClick={onClick || undefined}
      type={onClick ? "button" : undefined}
    >
      <h2 className="loss-title">{title}</h2>
      <div className="loss-legend">
        {series.map((item) => (
          <span key={item.name}>
            <i style={{ backgroundColor: LOSS_COLORS[item.name] }} />
            {item.name}
            {item.kind === "optimizer" &&
              (item.strategy === "batch"
                ? " (full dataset)"
                : ` (${data.optimizerConfigs[item.strategy].batch_size} samples)`)}
          </span>
        ))}
      </div>
      <div className="loss-chart-scroll">
        <svg
          className="loss-chart"
          role="img"
          viewBox={`0 0 ${width} ${height}`}
          aria-label={`Binary cross-entropy over ${epochCount} training epochs`}
        >
          {yTicks.map((value) => {
            const y = yPosition(value);
            return (
              <g key={value}>
                <line className="chart-grid-line" x1={margin.left} x2={width - margin.right} y1={y} y2={y} />
                <text className="chart-tick" x={margin.left - 12} y={y + 4} textAnchor="end">
                  {value.toFixed(3)}
                </text>
              </g>
            );
          })}
          {xTicks.map((epoch) => {
            const x = xPosition(epoch - 1);
            return (
              <text className="chart-tick" key={epoch} x={x} y={height - 24} textAnchor="middle">
                {epoch}
              </text>
            );
          })}
          <line className="chart-axis" x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} />
          <line className="chart-axis" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
          {series.map((item) => (
            <polyline
              className={item.kind === "average" ? "loss-line average" : "loss-line"}
              key={item.name}
              points={item.values
                .map((value, index) => `${xPosition(index)},${yPosition(value)}`)
                .join(" ")}
              stroke={LOSS_COLORS[item.name]}
            />
          ))}
          <text className="chart-label" x={margin.left + plotWidth / 2} y={height - 2} textAnchor="middle">
            Epoch
          </text>
          <text
            className="chart-label"
            textAnchor="middle"
            transform={`translate(18 ${margin.top + plotHeight / 2}) rotate(-90)`}
          >
            Binary cross-entropy
          </text>
        </svg>
      </div>
    </Element>
  );
}

function TrainingLossCharts({ data, onSelect }) {
  const optimizerCharts = data.series.map((series) => ({
    title: `${series.name} loss`,
    series: [
      { ...series, kind: "average", name: "Average" },
      ...data.houseSeries[series.strategy],
    ],
  }));

  return (
    <div className="loss-grid">
      <LossChart
        combined
        data={data}
        onClick={() =>
          onSelect({ title: "Optimizer comparison", series: data.series })
        }
        series={data.series}
        title="Optimizer comparison"
      />
      {optimizerCharts.map((chart) => (
        <LossChart
          data={data}
          key={chart.title}
          onClick={() => onSelect(chart)}
          series={chart.series}
          title={chart.title}
        />
      ))}
    </div>
  );
}

function RegressionChart({ data, series, title, onClick = null }) {
  const width = 1000;
  const height = 440;
  const margin = { top: 24, right: 28, bottom: 56, left: 78 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xValues = series.flatMap((item) => item.points.map((point) => point.x));
  const minimum = Math.min(...xValues);
  const maximum = Math.max(...xValues);
  const xRange = maximum - minimum || 1;

  function xPosition(value) {
    return margin.left + ((value - minimum) / xRange) * plotWidth;
  }

  function yPosition(value) {
    return margin.top + (1 - value) * plotHeight;
  }

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map(
    (ratio) => minimum + ratio * xRange
  );
  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const Element = onClick ? "button" : "section";

  return (
    <Element
      className={`loss-panel${onClick ? " loss-card" : ""}`}
      aria-label={`${title} chart`}
      onClick={onClick || undefined}
      type={onClick ? "button" : undefined}
    >
      <h2 className="loss-title">{title}</h2>
      <div className="loss-legend">
        <span><i className="observed-dot" />Observed labels</span>
        {series.map((item) => (
          <span key={item.name}>
            <i style={{ backgroundColor: LOSS_COLORS[item.name] }} />
            {item.name}
          </span>
        ))}
      </div>
      <div className="loss-chart-scroll">
        <svg
          className="loss-chart regression-chart"
          role="img"
          viewBox={`0 0 ${width} ${height}`}
          aria-label={`${data.house} probability from the full model score`}
        >
          {yTicks.map((value) => {
            const y = yPosition(value);
            return (
              <g key={value}>
                <line className="chart-grid-line" x1={margin.left} x2={width - margin.right} y1={y} y2={y} />
                <text className="chart-tick" x={margin.left - 12} y={y + 4} textAnchor="end">
                  {value.toFixed(2)}
                </text>
              </g>
            );
          })}
          {xTicks.map((value) => (
            <text className="chart-tick" key={value} x={xPosition(value)} y={height - 24} textAnchor="middle">
              {value.toFixed(1)}
            </text>
          ))}
          <line className="chart-axis" x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} />
          <line className="chart-axis" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
          {series.map((item) => (
            <g key={`${item.name}-points`}>
              {item.points.map((point, index) => (
                <circle
                  className="regression-point"
                  cx={xPosition(point.x)}
                  cy={yPosition(point.y)}
                  key={`${point.x}-${index}`}
                  r="2.3"
                  style={{ fill: LOSS_COLORS[item.name] }}
                />
              ))}
            </g>
          ))}
          {series.map((item) => (
            <polyline
              className="regression-line"
              key={item.name}
              points={item.values.map((point) => `${xPosition(point.x)},${yPosition(point.y)}`).join(" ")}
              stroke={LOSS_COLORS[item.name]}
            />
          ))}
          <text className="chart-label" x={margin.left + plotWidth / 2} y={height - 2} textAnchor="middle">
            Decision score z (all {data.featureCount} features)
          </text>
          <text
            className="chart-label"
            textAnchor="middle"
            transform={`translate(18 ${margin.top + plotHeight / 2}) rotate(-90)`}
          >
            P({data.house})
          </text>
        </svg>
      </div>
    </Element>
  );
}

function RegressionCharts({ data, onSelect }) {
  const charts = [
    { title: "Optimizer comparison", series: data.series },
    ...data.series.map((series) => ({
      title: `${series.name} regression`,
      series: [series],
    })),
  ];

  return (
    <div className="loss-grid">
      {charts.map((chart) => (
        <RegressionChart
          data={data}
          key={chart.title}
          onClick={() => onSelect(chart)}
          series={chart.series}
          title={chart.title}
        />
      ))}
    </div>
  );
}

function App() {
  const [payload, setPayload] = useState(null);
  const [pairPlot, setPairPlot] = useState(null);
  const [scatterPlot, setScatterPlot] = useState(null);
  const [trainingLoss, setTrainingLoss] = useState(null);
  const [regressionData, setRegressionData] = useState(null);
  const [datasetList, setDatasetList] = useState([]);
  const [datasetRows, setDatasetRows] = useState(null);
  const [describeData, setDescribeData] = useState(null);
  const [activeDataset, setActiveDataset] = useState("train");
  const [datasetQuery, setDatasetQuery] = useState("");
  const [datasetPageSize, setDatasetPageSize] = useState(50);
  const [datasetOffset, setDatasetOffset] = useState(0);
  const [error, setError] = useState("");
  const [datasetError, setDatasetError] = useState("");
  const [describeError, setDescribeError] = useState("");
  const [trainingLossError, setTrainingLossError] = useState("");
  const [regressionError, setRegressionError] = useState("");
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [describeLoading, setDescribeLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState("histograms");
  const [pairMode, setPairMode] = useState("focused");
  const [pairFeatureX, setPairFeatureX] = useState("");
  const [pairFeatureY, setPairFeatureY] = useState("");
  const [scatterMode, setScatterMode] = useState("top");
  const [histogramSort, setHistogramSort] = useState("dataset");
  const [scatterSort, setScatterSort] = useState("dataset");
  const [regressionHouse, setRegressionHouse] = useState("");
  const [regressionLoading, setRegressionLoading] = useState(false);
  const [trainingRefreshLoading, setTrainingRefreshLoading] = useState(false);
  const [selectedHistogram, setSelectedHistogram] = useState(null);
  const [selectedScatterPair, setSelectedScatterPair] = useState(null);
  const [selectedLossGraph, setSelectedLossGraph] = useState(null);
  const [selectedRegressionGraph, setSelectedRegressionGraph] = useState(null);

  async function loadData() {
    setLoading(true);
    setError("");
    try {
      const [
        histogramResponse,
        pairPlotResponse,
        scatterPlotResponse,
        datasetsResponse,
        trainingLossResponse,
      ] = await Promise.all([
        fetch("/api/histograms"),
        fetch("/api/pair-plot"),
        fetch("/api/scatter-plot"),
        fetch("/api/datasets"),
        fetch("/api/training-loss", { cache: "no-store" }),
      ]);
      const histogramData = await histogramResponse.json();
      const pairPlotData = await pairPlotResponse.json();
      const scatterPlotData = await scatterPlotResponse.json();
      const datasetsData = await datasetsResponse.json();
      const trainingLossData = await trainingLossResponse.json();
      if (!histogramResponse.ok) {
        throw new Error(histogramData.error || "Unable to load histograms");
      }
      if (!pairPlotResponse.ok) {
        throw new Error(pairPlotData.error || "Unable to load pair plot");
      }
      if (!scatterPlotResponse.ok) {
        throw new Error(scatterPlotData.error || "Unable to load scatter plot");
      }
      if (!datasetsResponse.ok) {
        throw new Error(datasetsData.error || "Unable to load datasets");
      }
      setPayload(histogramData);
      setPairPlot(pairPlotData);
      setScatterPlot(scatterPlotData);
      setDatasetList(datasetsData.datasets);
      if (trainingLossResponse.ok) {
        setTrainingLoss(trainingLossData);
        setTrainingLossError("");
      } else {
        setTrainingLoss(null);
        setTrainingLossError(
          trainingLossData.error || "Unable to load training loss"
        );
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function refreshTrainingVisuals() {
    setTrainingRefreshLoading(true);
    setTrainingLossError("");
    setRegressionError("");
    try {
      const cacheKey = Date.now().toString();
      const regressionParams = new URLSearchParams({ refresh: cacheKey });
      if (regressionHouse) {
        regressionParams.set("house", regressionHouse);
      }
      const [lossResponse, regressionResponse] = await Promise.all([
        fetch(`/api/training-loss?refresh=${cacheKey}`, { cache: "no-store" }),
        fetch(`/api/regression-curves?${regressionParams}`, { cache: "no-store" }),
      ]);
      const lossData = await lossResponse.json();
      const nextRegressionData = await regressionResponse.json();
      if (!lossResponse.ok) {
        throw new Error(lossData.error || "Unable to refresh training loss");
      }
      if (!regressionResponse.ok) {
        throw new Error(
          nextRegressionData.error || "Unable to refresh regression curves"
        );
      }
      setTrainingLoss(lossData);
      setRegressionData(nextRegressionData);
      setRegressionHouse(nextRegressionData.house);
    } catch (requestError) {
      setTrainingLossError(requestError.message);
      setRegressionError(requestError.message);
    } finally {
      setTrainingRefreshLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    async function loadDatasetRows() {
      setDatasetLoading(true);
      setDatasetError("");
      try {
        const params = new URLSearchParams({
          limit: String(datasetPageSize),
          offset: String(datasetOffset),
        });
        if (datasetQuery.trim()) {
          params.set("q", datasetQuery.trim());
        }

        const response = await fetch(`/api/datasets/${activeDataset}?${params}`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Unable to load dataset rows");
        }
        setDatasetRows(data);
      } catch (requestError) {
        setDatasetError(requestError.message);
      } finally {
        setDatasetLoading(false);
      }
    }

    loadDatasetRows();
  }, [activeDataset, datasetOffset, datasetPageSize, datasetQuery]);

  useEffect(() => {
    async function loadDescribeData() {
      setDescribeLoading(true);
      setDescribeError("");
      try {
        const response = await fetch(`/api/describe/${activeDataset}`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Unable to describe dataset");
        }
        setDescribeData(data);
      } catch (requestError) {
        setDescribeError(requestError.message);
      } finally {
        setDescribeLoading(false);
      }
    }

    loadDescribeData();
  }, [activeDataset]);

  useEffect(() => {
    async function loadRegressionData() {
      setRegressionLoading(true);
      setRegressionError("");
      try {
        const params = new URLSearchParams();
        if (regressionHouse) {
          params.set("house", regressionHouse);
        }
        const query = params.toString();
        const response = await fetch(
          `/api/regression-curves${query ? `?${query}` : ""}`,
          { cache: "no-store" }
        );
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Unable to load regression curves");
        }
        setRegressionData(data);
        setRegressionHouse((current) => current || data.house);
      } catch (requestError) {
        setRegressionError(requestError.message);
      } finally {
        setRegressionLoading(false);
      }
    }

    loadRegressionData();
  }, [regressionHouse]);

  useEffect(() => {
    if (!pairPlot || pairPlot.features.length < 2) {
      return;
    }
    setPairFeatureX((currentFeature) => currentFeature || pairPlot.features[0]);
    setPairFeatureY((currentFeature) => currentFeature || pairPlot.features[1]);
  }, [pairPlot]);

  useEffect(() => {
    function closeOnEscape(event) {
      if (event.key === "Escape") {
        setSelectedHistogram(null);
        setSelectedScatterPair(null);
        setSelectedLossGraph(null);
        setSelectedRegressionGraph(null);
      }
    }

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  const features = useMemo(() => {
    if (!payload) {
      return [];
    }
    let nextFeatures = payload.features;

    if (histogramSort === "az") {
      nextFeatures = [...nextFeatures].sort((left, right) =>
        left.name.localeCompare(right.name)
      );
    }
    if (histogramSort === "za") {
      nextFeatures = [...nextFeatures].sort((left, right) =>
        right.name.localeCompare(left.name)
      );
    }
    if (histogramSort === "similarity") {
      nextFeatures = [...nextFeatures].sort(
        (left, right) => left.similarityScore - right.similarityScore
      );
    }
    if (histogramSort === "dissimilarity") {
      nextFeatures = [...nextFeatures].sort(
        (left, right) => right.similarityScore - left.similarityScore
      );
    }
    return nextFeatures;
  }, [histogramSort, payload]);

  const pairPlotImage = useMemo(() => {
    if (!pairPlot) {
      return "";
    }
    if (pairMode === "all") {
      return pairPlot.allFeaturesImage;
    }
    if (pairMode === "focused" && pairFeatureX && pairFeatureY) {
      const params = new URLSearchParams({
        features: `${pairFeatureX},${pairFeatureY}`,
      });
      return `/api/pair-plot.png?${params}`;
    }
    return pairPlot.defaultImage;
  }, [pairFeatureX, pairFeatureY, pairMode, pairPlot]);

  const scatterPairs = useMemo(() => {
    if (!scatterPlot) {
      return [];
    }
    let nextPairs = scatterPlot.pairs;

    if (scatterSort === "az") {
      nextPairs = [...nextPairs].sort((left, right) =>
        `${left.featureX} ${left.featureY}`.localeCompare(
          `${right.featureX} ${right.featureY}`
        )
      );
    }
    if (scatterSort === "za") {
      nextPairs = [...nextPairs].sort((left, right) =>
        `${right.featureX} ${right.featureY}`.localeCompare(
          `${left.featureX} ${left.featureY}`
        )
      );
    }
    if (scatterSort === "pearsonDesc") {
      nextPairs = [...nextPairs].sort(
        (left, right) => right.correlation - left.correlation
      );
    }
    if (scatterSort === "pearsonAsc") {
      nextPairs = [...nextPairs].sort(
        (left, right) => left.correlation - right.correlation
      );
    }

    if (scatterMode === "all") {
      return nextPairs;
    }
    return nextPairs.slice(0, 12);
  }, [scatterMode, scatterPlot, scatterSort]);

  const datasetEnd = datasetRows
    ? Math.min(datasetRows.offset + datasetRows.rows.length, datasetRows.total)
    : 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">DSLR data science</p>
          <h1>Hogwarts data explorer</h1>
        </div>
      </header>

      <section className="view-tabs" aria-label="Plot views">
        <button
          className={activeView === "datasets" ? "tab active" : "tab"}
          onClick={() => setActiveView("datasets")}
        >
          <Database size={18} />
          Datasets
        </button>
        <button
          className={activeView === "describe" ? "tab active" : "tab"}
          onClick={() => setActiveView("describe")}
        >
          <Calculator size={18} />
          Describe
        </button>
        <button
          className={activeView === "histograms" ? "tab active" : "tab"}
          onClick={() => setActiveView("histograms")}
        >
          <BarChart3 size={18} />
          Histograms
        </button>
        <button
          className={activeView === "scatterPlot" ? "tab active" : "tab"}
          onClick={() => setActiveView("scatterPlot")}
        >
          <ScatterChart size={18} />
          Scatter Plot
        </button>
        <button
          className={activeView === "pairPlot" ? "tab active" : "tab"}
          onClick={() => setActiveView("pairPlot")}
        >
          <Grid3X3 size={18} />
          Pair Plot
        </button>
        <button
          className={activeView === "trainingLoss" ? "tab active" : "tab"}
          onClick={() => setActiveView("trainingLoss")}
        >
          <LineChart size={18} />
          Training Loss
        </button>
        <button
          className={activeView === "regression" ? "tab active" : "tab"}
          onClick={() => setActiveView("regression")}
        >
          <Activity size={18} />
          Regression Curves
        </button>
      </section>

      {activeView === "histograms" && (
        <section className="toolbar" aria-label="Histogram controls">
          <div className="summary">
            <BarChart3 size={20} />
            <span>
              {payload
                ? `${payload.features.length} courses`
                : "Loading courses"}
            </span>
          </div>
          <div className="select-group">
            <select
              value={histogramSort}
              onChange={(event) => setHistogramSort(event.target.value)}
              aria-label="Sort histogram courses"
            >
              <option value="dataset">Dataset order</option>
              <option value="similarity">Most similar</option>
              <option value="dissimilarity">Least similar</option>
              <option value="az">A to Z</option>
              <option value="za">Z to A</option>
            </select>
          </div>
        </section>
      )}

      {activeView === "pairPlot" && (
        <section className="toolbar" aria-label="Pair plot controls">
          <div className="summary">
            <Grid3X3 size={20} />
            <span>
              {pairMode === "focused"
                ? `${pairFeatureX || "Feature"} x ${pairFeatureY || "Feature"}`
                : pairMode === "selected"
                  ? "Selected features"
                  : "All numeric features"}
            </span>
          </div>
          <div className="toolbar-actions">
            {pairPlot && pairMode === "focused" && (
              <div className="select-group selected-pair-controls">
                <select
                  value={pairFeatureX}
                  onChange={(event) => setPairFeatureX(event.target.value)}
                  aria-label="Select pair plot column feature"
                >
                  {pairPlot.features.map((feature) => (
                    <option
                      disabled={feature === pairFeatureY}
                      key={feature}
                      value={feature}
                    >
                      Column: {feature}
                    </option>
                  ))}
                </select>
                <select
                  value={pairFeatureY}
                  onChange={(event) => setPairFeatureY(event.target.value)}
                  aria-label="Select pair plot row feature"
                >
                  {pairPlot.features.map((feature) => (
                    <option
                      disabled={feature === pairFeatureX}
                      key={feature}
                      value={feature}
                    >
                      Row: {feature}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <div className="segmented" role="group" aria-label="Pair plot feature mode">
              <button
                className={pairMode === "focused" ? "active" : ""}
                onClick={() => setPairMode("focused")}
              >
                Focused
              </button>
              <button
                className={pairMode === "selected" ? "active" : ""}
                onClick={() => setPairMode("selected")}
              >
                Selected
              </button>
              <button
                className={pairMode === "all" ? "active" : ""}
                onClick={() => setPairMode("all")}
              >
                All
              </button>
            </div>
          </div>
        </section>
      )}

      {activeView === "scatterPlot" && (
        <section className="toolbar" aria-label="Scatter plot controls">
          <div className="summary">
            <ScatterChart size={20} />
            <span>
              {!scatterPlot
                ? "Loading scatter plot"
                : scatterMode === "all"
                  ? `${scatterPairs.length} of ${scatterPlot.pairs.length} feature pairs`
                  : `${scatterPairs.length} of ${scatterPlot.pairs.length} feature pairs`}
            </span>
          </div>
          <div className="segmented" role="group" aria-label="Scatter plot pair mode">
            <button
              className={scatterMode === "top" ? "active" : ""}
              onClick={() => setScatterMode("top")}
            >
              Top
            </button>
            <button
              className={scatterMode === "all" ? "active" : ""}
              onClick={() => setScatterMode("all")}
            >
              All
            </button>
          </div>
          <div className="select-group">
            <select
              value={scatterSort}
              onChange={(event) => setScatterSort(event.target.value)}
              aria-label="Sort scatter plot pairs"
            >
              <option value="dataset">Dataset order</option>
              <option value="pearsonDesc">Pearson r high to low</option>
              <option value="pearsonAsc">Pearson r low to high</option>
              <option value="az">A to Z</option>
              <option value="za">Z to A</option>
            </select>
          </div>
        </section>
      )}

      {activeView === "datasets" && (
        <section className="toolbar" aria-label="Dataset controls">
          <div className="summary">
            <Database size={20} />
            <span>
              {datasetRows
                ? `${datasetRows.path} | ${datasetRows.total} rows`
                : "Loading dataset"}
            </span>
          </div>
          <div className="filter-row">
            <select
              value={activeDataset}
              onChange={(event) => {
                setActiveDataset(event.target.value);
                setDatasetOffset(0);
              }}
              aria-label="Select dataset"
            >
              {datasetList.map((dataset) => (
                <option key={dataset.name} value={dataset.name}>
                  {dataset.name}
                </option>
              ))}
            </select>
            <input
              value={datasetQuery}
              onChange={(event) => {
                setDatasetQuery(event.target.value);
                setDatasetOffset(0);
              }}
              placeholder="Search rows"
              aria-label="Search dataset rows"
            />
            <select
              value={datasetPageSize}
              onChange={(event) => {
                setDatasetPageSize(Number(event.target.value));
                setDatasetOffset(0);
              }}
              aria-label="Rows per page"
            >
              <option value={25}>25 rows</option>
              <option value={50}>50 rows</option>
              <option value={100}>100 rows</option>
              <option value={200}>200 rows</option>
            </select>
          </div>
        </section>
      )}

      {activeView === "describe" && (
        <section className="toolbar" aria-label="Describe controls">
          <div className="summary">
            <Calculator size={20} />
            <span>
              {describeData
                ? `${describeData.path} | ${describeData.features.length} numeric features`
                : "Loading statistics"}
            </span>
          </div>
          <select
            value={activeDataset}
            onChange={(event) => setActiveDataset(event.target.value)}
            aria-label="Select dataset to describe"
          >
            {datasetList.map((dataset) => (
              <option key={dataset.name} value={dataset.name}>
                {dataset.name}
              </option>
            ))}
          </select>
        </section>
      )}

      {activeView === "trainingLoss" && (
        <section className="toolbar" aria-label="Training loss summary">
          <div className="summary">
            <LineChart size={20} />
            <span>
              {trainingLoss
                ? `${trainingLoss.series.length} optimizers | ${trainingLoss.epochs} epochs | trained ${formatModelTime(trainingLoss.modelUpdatedAt)}`
                : "Training history unavailable"}
            </span>
          </div>
          <button
            className="icon-button refresh-button"
            disabled={trainingRefreshLoading}
            onClick={refreshTrainingVisuals}
            type="button"
          >
            <RefreshCw className={trainingRefreshLoading ? "spinning" : ""} size={18} />
            {trainingRefreshLoading ? "Refreshing" : "Refresh training data"}
          </button>
        </section>
      )}

      {activeView === "regression" && (
        <section className="toolbar" aria-label="Regression curve controls">
          <div className="summary">
            <Activity size={20} />
            <span>
              One-vs-rest probability from the full {regressionData?.featureCount || 10}-feature model
              {regressionData
                ? ` | trained ${formatModelTime(regressionData.modelUpdatedAt)}`
                : ""}
            </span>
          </div>
          <div className="toolbar-actions">
            {regressionData && (
              <div className="select-group">
              <select
                value={regressionHouse}
                onChange={(event) => setRegressionHouse(event.target.value)}
                aria-label="Select regression house"
              >
                {regressionData.houses.map((house) => (
                  <option key={house} value={house}>{house}</option>
                ))}
              </select>
              </div>
            )}
            <button
              className="icon-button refresh-button"
              disabled={trainingRefreshLoading}
              onClick={refreshTrainingVisuals}
              type="button"
            >
              <RefreshCw className={trainingRefreshLoading ? "spinning" : ""} size={18} />
              {trainingRefreshLoading ? "Refreshing" : "Refresh training data"}
            </button>
          </div>
        </section>
      )}

      {error && <p className="error">{error}</p>}
      {datasetError && activeView === "datasets" && (
        <p className="error">{datasetError}</p>
      )}
      {describeError && activeView === "describe" && (
        <p className="error">{describeError}</p>
      )}
      {trainingLossError && activeView === "trainingLoss" && (
        <p className="error">{trainingLossError}</p>
      )}
      {regressionError && activeView === "regression" && (
        <p className="error">{regressionError}</p>
      )}
      {(loading ||
        (describeLoading && activeView === "describe") ||
        (regressionLoading && activeView === "regression")) && (
        <p className="state">Loading data...</p>
      )}

      {!loading && !error && activeView === "histograms" && (
        <section className="histogram-grid" aria-label="Course histograms">
          {features.map((feature) => (
            <button
              className="histogram-card plot-card"
              key={feature.name}
              onClick={() => setSelectedHistogram(feature)}
              type="button"
            >
              <div className="card-title">
                <span>{feature.name}</span>
                <span className="card-meta">
                  similarity {feature.similarityScore.toFixed(6)}
                </span>
              </div>
              <img
                src={feature.image}
                alt={`${feature.name} histogram by Hogwarts house`}
                loading="lazy"
              />
            </button>
          ))}
        </section>
      )}

      {!loading && !error && activeView === "pairPlot" && pairPlot && (
        <section
          className={pairMode === "focused" ? "plot-panel focused-pair-panel" : "plot-panel"}
          aria-label="Pair plot"
        >
          <img src={pairPlotImage} alt="Hogwarts course pair plot" />
        </section>
      )}

      {!loading && !error && activeView === "scatterPlot" && scatterPlot && (
        <section className="histogram-grid" aria-label="Scatter plots">
          {scatterPairs.map((pair) => (
            <button
              className="histogram-card plot-card"
              key={`${pair.featureX}-${pair.featureY}`}
              onClick={() => setSelectedScatterPair(pair)}
              type="button"
            >
              <div className="card-title">
                <span>{pair.featureX}</span>
                <span className="card-meta">
                  vs {pair.featureY} | Pearson r {pair.correlation.toFixed(6)}
                </span>
              </div>
              <img
                src={pair.image}
                alt={`${pair.featureX} versus ${pair.featureY} scatter plot`}
                loading="lazy"
              />
            </button>
          ))}
        </section>
      )}

      {!loading &&
        !error &&
        !trainingLossError &&
        activeView === "trainingLoss" &&
        trainingLoss && (
          <TrainingLossCharts data={trainingLoss} onSelect={setSelectedLossGraph} />
        )}

      {!loading &&
        !error &&
        !regressionLoading &&
        !regressionError &&
        activeView === "regression" &&
        regressionData && (
          <RegressionCharts
            data={regressionData}
            onSelect={setSelectedRegressionGraph}
          />
        )}

      {!loading &&
        !error &&
        !describeLoading &&
        !describeError &&
        activeView === "describe" &&
        describeData && (
          <section className="dataset-panel" aria-label="Descriptive statistics table">
            <div className="table-scroll">
              <table className="describe-table">
                <thead>
                  <tr>
                    <th>Statistic</th>
                    {describeData.features.map((feature) => (
                      <th key={feature.name}>{feature.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {describeData.statistics.map((statistic) => (
                    <tr key={statistic}>
                      <th scope="row">{statistic}</th>
                      {describeData.features.map((feature) => (
                        <td key={feature.name}>
                          {formatStatistic(feature.values[statistic])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

      {!loading && !error && activeView === "datasets" && datasetRows && (
        <section className="dataset-panel" aria-label="Dataset table">
          <div className="dataset-pager">
            <span>
              {datasetRows.total === 0
                ? "No rows"
                : `${datasetRows.offset + 1}-${datasetEnd} of ${datasetRows.total}`}
            </span>
            <div className="pager-actions">
              <button
                className="icon-button"
                disabled={datasetLoading || datasetRows.offset === 0}
                onClick={() =>
                  setDatasetOffset(Math.max(0, datasetOffset - datasetPageSize))
                }
                title="Previous page"
                type="button"
              >
                <ChevronLeft size={18} />
              </button>
              <button
                className="icon-button"
                disabled={datasetLoading || datasetEnd >= datasetRows.total}
                onClick={() => setDatasetOffset(datasetOffset + datasetPageSize)}
                title="Next page"
                type="button"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  {datasetRows.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {datasetRows.rows.map((row, rowIndex) => (
                  <tr key={`${datasetRows.offset}-${rowIndex}`}>
                    {datasetRows.columns.map((column) => (
                      <td key={column}>{row[column] || ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {selectedHistogram && (
        <div
          className="modal-backdrop"
          onClick={() => setSelectedHistogram(null)}
          role="presentation"
        >
          <section
            className="plot-modal"
            aria-label={`${selectedHistogram.name} enlarged histogram`}
            aria-modal="true"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">Histogram</p>
                <h2>{selectedHistogram.name}</h2>
              </div>
              <button
                className="icon-button"
                onClick={() => setSelectedHistogram(null)}
                title="Close"
                type="button"
              >
                <X size={18} />
              </button>
            </header>
            <img
              src={selectedHistogram.image}
              alt={`${selectedHistogram.name} histogram by Hogwarts house`}
            />
          </section>
        </div>
      )}

      {selectedScatterPair && (
        <div
          className="modal-backdrop"
          onClick={() => setSelectedScatterPair(null)}
          role="presentation"
        >
          <section
            className="plot-modal scatter-modal"
            aria-label={`${selectedScatterPair.featureX} versus ${selectedScatterPair.featureY} enlarged scatter plot`}
            aria-modal="true"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">Scatter plot</p>
                <h2>{selectedScatterPair.featureX}</h2>
                <p className="modal-subtitle">
                  vs {selectedScatterPair.featureY} | Pearson r{" "}
                  {selectedScatterPair.correlation.toFixed(6)}
                </p>
              </div>
              <button
                className="icon-button"
                onClick={() => setSelectedScatterPair(null)}
                title="Close"
                type="button"
              >
                <X size={18} />
              </button>
            </header>
            <img
              src={selectedScatterPair.image}
              alt={`${selectedScatterPair.featureX} versus ${selectedScatterPair.featureY} scatter plot`}
            />
          </section>
        </div>
      )}

      {selectedLossGraph && trainingLoss && (
        <div
          className="modal-backdrop"
          onClick={() => setSelectedLossGraph(null)}
          role="presentation"
        >
          <section
            className="plot-modal loss-modal"
            aria-label={`${selectedLossGraph.title} enlarged`}
            aria-modal="true"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">Training loss</p>
                <h2>{selectedLossGraph.title}</h2>
              </div>
              <button
                className="icon-button"
                onClick={() => setSelectedLossGraph(null)}
                title="Close"
                type="button"
              >
                <X size={18} />
              </button>
            </header>
            <LossChart
              data={trainingLoss}
              series={selectedLossGraph.series}
              title={selectedLossGraph.title}
            />
          </section>
        </div>
      )}

      {selectedRegressionGraph && regressionData && (
        <div
          className="modal-backdrop"
          onClick={() => setSelectedRegressionGraph(null)}
          role="presentation"
        >
          <section
            className="plot-modal loss-modal"
            aria-label={`${selectedRegressionGraph.title} enlarged`}
            aria-modal="true"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">Logistic regression</p>
                <h2>{selectedRegressionGraph.title}</h2>
                <p className="modal-subtitle">
                  {regressionData.house} | all {regressionData.featureCount} model features
                </p>
              </div>
              <button
                className="icon-button"
                onClick={() => setSelectedRegressionGraph(null)}
                title="Close"
                type="button"
              >
                <X size={18} />
              </button>
            </header>
            <RegressionChart
              data={regressionData}
              series={selectedRegressionGraph.series}
              title={selectedRegressionGraph.title}
            />
          </section>
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
