(() => {
  "use strict";

  const data = window.STRESS_DATA;
  const $ = id => document.getElementById(id);
  const fmt = (value, digits = 1) => Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
  const set = (id, value) => { $(id).textContent = value; };
  const clamp = value => Math.max(0, Math.min(100, Number(value)));

  function showError(message) {
    const banner = $("error-banner");
    banner.textContent = message;
    banner.hidden = false;
  }

  function clearError() {
    $("error-banner").hidden = true;
  }

  function signed(value) {
    return `${value > 0 ? "+" : ""}${fmt(value, 1)} pts vs ${data.bootstrap.latest_year}`;
  }

  function readableFeature(value) {
    return value.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
  }

  function updateHorizon() {
    const years = Number($("year-input").value) - data.bootstrap.latest_year;
    $("horizon-label").textContent = `+${Math.max(1, years)} years`;
  }

  function renderRegistry() {
    $("model-table").innerHTML = data.bootstrap.diagnostics.map(row => `
      <tr>
        <td>${row.target}</td>
        <td>${row.model}</td>
        <td>${fmt(row.mae, row.target === "SAIFI" ? 3 : 2)}</td>
        <td>${fmt(row.rmse, row.target === "SAIFI" ? 3 : 2)}</td>
        <td>${fmt(row.r2, 3)}</td>
      </tr>
    `).join("");

    const electricity = [
      ...data.bootstrap.main_inputs.saidi.slice(0, 3),
      ...data.bootstrap.main_inputs.saifi.slice(0, 2)
    ];
    const water = [
      ...data.bootstrap.main_inputs.drought.slice(0, 3),
      ...data.bootstrap.main_inputs.compliance.slice(0, 2)
    ];
    $("electricity-inputs").innerHTML = electricity.map(v => `<li>${readableFeature(v)}</li>`).join("");
    $("water-inputs").innerHTML = water.map(v => `<li>${readableFeature(v)}</li>`).join("");
  }

  function updateMetric(prefix, metric) {
    set(`${prefix}-score`, fmt(metric.score, 1));
    set(`${prefix}-band`, metric.band.label);
    set(`${prefix}-change`, signed(metric.change));
    set(`${prefix}-range`, `${fmt(metric.range[0], 1)}–${fmt(metric.range[1], 1)}`);
    $(`${prefix}-bar`).style.width = `${clamp(metric.score)}%`;
  }

  function projectedSeries(state, year) {
    const rows = [];
    for (let current = data.bootstrap.min_year; current <= year; current += 1) {
      const p = data.projections[state][String(current)];
      rows.push({
        year: current,
        electricity: p.electricity.score,
        water: p.water.score,
        electricity_low: p.electricity.range[0],
        electricity_high: p.electricity.range[1],
        water_low: p.water.range[0],
        water_high: p.water.range[1],
        type: "projected"
      });
    }
    return rows;
  }

  function render() {
    clearError();
    try {
      const state = $("state-select").value;
      const year = Number($("year-input").value);
      if (!data.states[state]) throw new Error("Choose a valid state.");
      if (!Number.isInteger(year) || year < data.bootstrap.min_year || year > data.bootstrap.max_year) {
        throw new Error(`Choose a year from ${data.bootstrap.min_year} to ${data.bootstrap.max_year}.`);
      }

      const projection = data.projections[state][String(year)];
      const rank = data.rankings[String(year)][state];
      const stateInfo = data.states[state];

      set("result-state", stateInfo.name);
      set("result-year", year);
      set("scenario-horizon", `${String(projection.years_ahead).padStart(2, "0")}Y`);
      updateMetric("electricity", projection.electricity);
      updateMetric("water", projection.water);

      set("saidi-value", fmt(projection.electricity.saidi, 1));
      set("saifi-value", fmt(projection.electricity.saifi, 3));
      set("drought-value", fmt(projection.water.drought, 1));
      set("compliance-value", fmt(projection.water.compliance, 3));
      set("electricity-rank", `${rank.electricity} / ${rank.total}`);
      set("water-rank", `${rank.water} / ${rank.total}`);

      const compositions = [
        ["duration", projection.electricity.duration_stress],
        ["frequency", projection.electricity.frequency_stress],
        ["drought", projection.water.drought_stress],
        ["compliance", projection.water.compliance_stress]
      ];
      for (const [name, value] of compositions) {
        set(`${name}-stress`, fmt(value, 1));
        $(`${name}-fill`).style.width = `${clamp(value)}%`;
      }

      drawChart([
        ...data.historical[state],
        ...projectedSeries(state, year)
      ]);
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
    }
  }

  function svg(name, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function linePath(points, x, y) {
    return points
      .filter(point => point.value !== null && point.value !== undefined)
      .map((point, index) => `${index ? "L" : "M"} ${x(point.year)} ${y(point.value)}`)
      .join(" ");
  }

  function areaPath(points, x, y, lowKey, highKey) {
    const valid = points.filter(point => point[lowKey] != null && point[highKey] != null);
    if (!valid.length) return "";
    const top = valid.map((p, i) => `${i ? "L" : "M"} ${x(p.year)} ${y(p[highKey])}`).join(" ");
    const bottom = valid.slice().reverse().map(p => `L ${x(p.year)} ${y(p[lowKey])}`).join(" ");
    return `${top} ${bottom} Z`;
  }

  function drawChart(series) {
    const chart = $("trend-chart");
    chart.innerHTML = "";

    const width = 920;
    const height = 360;
    const pad = {left: 54, right: 20, top: 24, bottom: 42};
    const lastYear = Math.max(...series.map(row => row.year));
    const firstYear = Math.max(Math.min(...series.map(row => row.year)), data.bootstrap.latest_year - 10);
    const visible = series.filter(row => row.year >= firstYear);
    const x = year => pad.left + ((year - firstYear) / Math.max(1, lastYear - firstYear)) * (width - pad.left - pad.right);
    const y = value => height - pad.bottom - (Number(value) / 100) * (height - pad.top - pad.bottom);

    [0, 25, 50, 75, 100].forEach(value => {
      chart.appendChild(svg("line", {x1: pad.left, y1: y(value), x2: width - pad.right, y2: y(value), class: "chart-grid"}));
      const label = svg("text", {x: pad.left - 12, y: y(value) + 4, "text-anchor": "end", class: "chart-label"});
      label.textContent = value;
      chart.appendChild(label);
    });

    const step = lastYear - firstYear > 12 ? 2 : 1;
    for (let year = firstYear; year <= lastYear; year += step) {
      const label = svg("text", {x: x(year), y: height - 15, "text-anchor": "middle", class: "chart-label"});
      label.textContent = year;
      chart.appendChild(label);
    }

    const cutoff = x(data.bootstrap.latest_year);
    chart.appendChild(svg("line", {x1: cutoff, y1: pad.top, x2: cutoff, y2: height - pad.bottom, class: "cutoff"}));
    const cutoffLabel = svg("text", {x: cutoff - 6, y: pad.top + 10, "text-anchor": "end", class: "chart-label"});
    cutoffLabel.textContent = "observed cutoff";
    chart.appendChild(cutoffLabel);

    const observed = visible.filter(row => row.type === "observed");
    const projected = visible.filter(row => row.type === "projected");
    const anchor = observed.length ? observed[observed.length - 1] : null;
    const projectedWithAnchor = anchor ? [anchor, ...projected] : projected;

    const eArea = areaPath(projected, x, y, "electricity_low", "electricity_high");
    const wArea = areaPath(projected, x, y, "water_low", "water_high");
    if (eArea) chart.appendChild(svg("path", {d: eArea, class: "area-e"}));
    if (wArea) chart.appendChild(svg("path", {d: wArea, class: "area-w"}));

    const paths = [
      [observed.map(r => ({year: r.year, value: r.electricity})), "line-e"],
      [observed.map(r => ({year: r.year, value: r.water})), "line-w"],
      [projectedWithAnchor.map(r => ({year: r.year, value: r.electricity})), "line-ep"],
      [projectedWithAnchor.map(r => ({year: r.year, value: r.water})), "line-wp"]
    ];
    paths.forEach(([points, className]) => {
      const path = linePath(points, x, y);
      if (path) chart.appendChild(svg("path", {d: path, class: className}));
    });
  }

  function init() {
    if (!data || !data.bootstrap || !data.projections) {
      showError("Projection data could not be loaded. Keep static/data.js beside this page.");
      return;
    }

    $("state-select").innerHTML = data.bootstrap.states.map(state =>
      `<option value="${state.abbreviation}" ${state.abbreviation === "CA" ? "selected" : ""}>${state.name} (${state.abbreviation})</option>`
    ).join("");

    $("year-input").min = data.bootstrap.min_year;
    $("year-input").max = data.bootstrap.max_year;
    $("year-input").value = Math.min(2030, data.bootstrap.max_year);
    set("training-status", String(data.bootstrap.training.status || "complete").toUpperCase());
    set("latest-observation", data.bootstrap.latest_year);
    set("warning-text", data.bootstrap.warning);
    renderRegistry();
    updateHorizon();
    render();

    $("projection-form").addEventListener("submit", event => {
      event.preventDefault();
      render();
    });
    $("year-input").addEventListener("input", updateHorizon);
    $("state-select").addEventListener("change", render);
  }

  init();
})();
