/**
 * Chart.js rendering module for Historical prices, Indicators, and LSTM Forecasts.
 */

let historyChartInstance = null;
let rsiChartInstance = null;
let macdChartInstance = null;
let forecastChartInstance = null;

// Global Chart.js dark theme defaults
Chart.defaults.color = "#94a3b8";
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = "#0e1526";
Chart.defaults.plugins.tooltip.borderColor = "rgba(255, 255, 255, 0.1)";
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.titleFont = { family: "'JetBrains Mono', monospace", weight: "bold" };
Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono', monospace" };
Chart.defaults.plugins.tooltip.padding = 10;

/**
 * Renders the main historical price and moving averages chart.
 */
function renderHistoricalChart(records, currency = "USD") {
    const ctx = document.getElementById("historicalPriceChart");
    if (!ctx) return;

    if (historyChartInstance) {
        historyChartInstance.destroy();
    }

    const labels = records.map(r => r.date);
    const closePrices = records.map(r => r.close);
    const sma20 = records.map(r => r.sma_20);
    const sma50 = records.map(r => r.sma_50);
    const volumes = records.map(r => r.volume);

    const currSymbol = currency === "INR" ? "₹" : "$";

    historyChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Close Price",
                    data: closePrices,
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.08)",
                    borderWidth: 2,
                    pointRadius: records.length > 100 ? 0 : 2,
                    pointHoverRadius: 5,
                    fill: true,
                    tension: 0.15,
                    yAxisID: "y",
                },
                {
                    label: "SMA 20",
                    data: sma20,
                    borderColor: "#f59e0b",
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.15,
                    yAxisID: "y",
                },
                {
                    label: "SMA 50",
                    data: sma50,
                    borderColor: "#a855f7",
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.15,
                    yAxisID: "y",
                },
                {
                    label: "Volume",
                    data: volumes,
                    type: "bar",
                    backgroundColor: "rgba(148, 163, 184, 0.18)",
                    borderWidth: 0,
                    yAxisID: "yVolume",
                }
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false,
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.04)" },
                    ticks: { maxTicksLimit: 10, font: { family: "'JetBrains Mono', monospace", size: 11 } },
                },
                y: {
                    position: "right",
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: {
                        callback: (val) => `${currSymbol}${val.toLocaleString()}`,
                        font: { family: "'JetBrains Mono', monospace", size: 11 },
                    },
                },
                yVolume: {
                    position: "left",
                    grid: { display: false },
                    ticks: {
                        display: false,
                        max: Math.max(...volumes) * 4, // Keeps volume bars unobtrusive at chart bottom
                    },
                }
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: {
                        boxWidth: 12,
                        boxHeight: 12,
                        font: { size: 11 },
                    },
                },
            },
        },
    });
}

/**
 * Renders Relative Strength Index (RSI) chart with 70 / 30 guide lines.
 */
function renderRsiChart(records) {
    const ctx = document.getElementById("rsiChart");
    if (!ctx) return;

    if (rsiChartInstance) {
        rsiChartInstance.destroy();
    }

    const labels = records.map(r => r.date);
    const rsiValues = records.map(r => r.rsi_14);

    rsiChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "RSI (14)",
                    data: rsiValues,
                    borderColor: "#60a5fa",
                    borderWidth: 1.8,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                x: { display: false },
                y: {
                    min: 0,
                    max: 100,
                    position: "right",
                    grid: {
                        color: (ctx) => (ctx.tick.value === 70 || ctx.tick.value === 30) ? "rgba(239, 68, 68, 0.3)" : "rgba(255, 255, 255, 0.04)",
                    },
                    ticks: {
                        stepSize: 30,
                        callback: val => (val === 70 ? "70 (OB)" : val === 30 ? "30 (OS)" : val),
                        font: { family: "'JetBrains Mono', monospace", size: 10 }
                    },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });

    // Update status badge
    const latestRsi = rsiValues[rsiValues.length - 1];
    const badge = document.getElementById("rsiStatusBadge");
    if (badge && latestRsi !== undefined) {
        if (latestRsi >= 70) {
            badge.className = "badge bg-danger-subtle text-danger";
            badge.textContent = `Overbought (${latestRsi.toFixed(1)})`;
        } else if (latestRsi <= 30) {
            badge.className = "badge bg-success-subtle text-success";
            badge.textContent = `Oversold (${latestRsi.toFixed(1)})`;
        } else {
            badge.className = "badge bg-secondary-subtle text-secondary";
            badge.textContent = `Neutral (${latestRsi.toFixed(1)})`;
        }
    }
}

/**
 * Renders MACD line, signal line, and histogram bars.
 */
function renderMacdChart(records) {
    const ctx = document.getElementById("macdChart");
    if (!ctx) return;

    if (macdChartInstance) {
        macdChartInstance.destroy();
    }

    const labels = records.map(r => r.date);
    const macd = records.map(r => r.macd);
    const signal = records.map(r => r.macd_signal);
    const hist = records.map(r => r.macd_hist);

    const histColors = hist.map(h => (h >= 0 ? "rgba(16, 185, 129, 0.6)" : "rgba(239, 68, 68, 0.6)"));

    macdChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "MACD",
                    data: macd,
                    borderColor: "#38bdf8",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: "Signal",
                    data: signal,
                    borderColor: "#f59e0b",
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: "Histogram",
                    data: hist,
                    type: "bar",
                    backgroundColor: histColors,
                    borderWidth: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                x: { display: false },
                y: {
                    position: "right",
                    grid: { color: "rgba(255, 255, 255, 0.04)" },
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });

    const latestHist = hist[hist.length - 1];
    const badge = document.getElementById("macdStatusBadge");
    if (badge && latestHist !== undefined) {
        if (latestHist > 0) {
            badge.className = "badge bg-success-subtle text-success";
            badge.textContent = "Bullish Momentum";
        } else {
            badge.className = "badge bg-danger-subtle text-danger";
            badge.textContent = "Bearish Momentum";
        }
    }
}

/**
 * Renders the Historical vs LSTM Multi-Step Forecast Comparison Chart.
 * Displays past prices, predicted future steps, and 95% uncertainty envelope.
 */
function renderForecastComparisonChart(histTail, predictions, currency = "USD") {
    const ctx = document.getElementById("forecastComparisonChart");
    if (!ctx) return;

    if (forecastChartInstance) {
        forecastChartInstance.destroy();
    }

    const currSymbol = currency === "INR" ? "₹" : "$";

    // Combine timeline: historical tail dates + future dates
    const histDates = histTail.map(h => h.date);
    const histPrices = histTail.map(h => h.close);

    const predDates = predictions.map(p => p.date);
    const allLabels = [...histDates, ...predDates];

    // Bridge point: connect last historical point to first predicted point
    const lastHistPrice = histPrices[histPrices.length - 1];

    const histDataPoints = [...histPrices, ...new Array(predDates.length).fill(null)];

    const forecastDataPoints = [
        ...new Array(histPrices.length - 1).fill(null),
        lastHistPrice,
        ...predictions.map(p => p.predicted_price),
    ];

    const upperBandPoints = [
        ...new Array(histPrices.length - 1).fill(null),
        lastHistPrice,
        ...predictions.map(p => p.upper_bound),
    ];

    const lowerBandPoints = [
        ...new Array(histPrices.length - 1).fill(null),
        lastHistPrice,
        ...predictions.map(p => p.lower_bound),
    ];

    forecastChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: "Historical Close",
                    data: histDataPoints,
                    borderColor: "#94a3b8",
                    borderWidth: 2,
                    pointRadius: 2,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: "LSTM Forecast",
                    data: forecastDataPoints,
                    borderColor: "#38bdf8",
                    borderWidth: 2.5,
                    borderDash: [5, 4],
                    pointRadius: 4,
                    pointBackgroundColor: "#38bdf8",
                    fill: false,
                    tension: 0.15,
                },
                {
                    label: "Upper 95% Uncertainty",
                    data: upperBandPoints,
                    borderColor: "rgba(56, 189, 248, 0.4)",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: "+1", // Fill to next dataset (lower band)
                    backgroundColor: "rgba(56, 189, 248, 0.12)",
                    tension: 0.15,
                },
                {
                    label: "Lower 95% Uncertainty",
                    data: lowerBandPoints,
                    borderColor: "rgba(56, 189, 248, 0.4)",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.15,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.04)" },
                    ticks: { maxTicksLimit: 12, font: { family: "'JetBrains Mono', monospace", size: 11 } },
                },
                y: {
                    position: "right",
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: {
                        callback: val => `${currSymbol}${val.toLocaleString()}`,
                        font: { family: "'JetBrains Mono', monospace", size: 11 },
                    },
                },
            },
            plugins: {
                legend: {
                    display: false, // Custom legend rendered in HTML
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            if (context.parsed.y === null) return "";
                            return `${context.dataset.label}: ${currSymbol}${context.parsed.y.toFixed(2)}`;
                        },
                    },
                },
            },
        },
    });
}
