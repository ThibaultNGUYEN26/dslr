import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Database,
  Grid3X3,
  ScatterChart,
  X,
} from "lucide-react";
import "./styles.css";

function App() {
  const [payload, setPayload] = useState(null);
  const [pairPlot, setPairPlot] = useState(null);
  const [scatterPlot, setScatterPlot] = useState(null);
  const [datasetList, setDatasetList] = useState([]);
  const [datasetRows, setDatasetRows] = useState(null);
  const [activeDataset, setActiveDataset] = useState("train");
  const [datasetQuery, setDatasetQuery] = useState("");
  const [datasetPageSize, setDatasetPageSize] = useState(50);
  const [datasetOffset, setDatasetOffset] = useState(0);
  const [error, setError] = useState("");
  const [datasetError, setDatasetError] = useState("");
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState("histograms");
  const [pairMode, setPairMode] = useState("focused");
  const [pairFeatureX, setPairFeatureX] = useState("");
  const [pairFeatureY, setPairFeatureY] = useState("");
  const [scatterMode, setScatterMode] = useState("top");
  const [histogramSort, setHistogramSort] = useState("dataset");
  const [scatterSort, setScatterSort] = useState("dataset");
  const [selectedHistogram, setSelectedHistogram] = useState(null);
  const [selectedScatterPair, setSelectedScatterPair] = useState(null);

  async function loadData() {
    setLoading(true);
    setError("");
    try {
      const [
        histogramResponse,
        pairPlotResponse,
        scatterPlotResponse,
        datasetsResponse,
      ] = await Promise.all([
        fetch("/api/histograms"),
        fetch("/api/pair-plot"),
        fetch("/api/scatter-plot"),
        fetch("/api/datasets"),
      ]);
      const histogramData = await histogramResponse.json();
      const pairPlotData = await pairPlotResponse.json();
      const scatterPlotData = await scatterPlotResponse.json();
      const datasetsData = await datasetsResponse.json();
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
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
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
          <p className="eyebrow">DSLR visualization</p>
          <h1>Hogwarts course plots</h1>
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

      {error && <p className="error">{error}</p>}
      {datasetError && activeView === "datasets" && (
        <p className="error">{datasetError}</p>
      )}
      {loading && <p className="state">Loading plots...</p>}

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
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
