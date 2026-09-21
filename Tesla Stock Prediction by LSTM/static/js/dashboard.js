/**
 * Dashboard state and API controller.
 */

let activeTicker = "";
let currentCurrency = "USD";
let activePeriod = "1y";

document.addEventListener("DOMContentLoaded", () => {
    activeTicker = document.body.dataset.activeTicker || "TCS.NS";

    // Setup Epochs slider display
    const epochsInput = document.getElementById("inputEpochs");
    const epochsDisplay = document.getElementById("epochsDisplay");
    if (epochsInput && epochsDisplay) {
        epochsInput.addEventListener("input", (e) => {
            epochsDisplay.textContent = e.target.value;
        });
    }

    // Bind Timeframe buttons
    const periodButtons = document.querySelectorAll(".timeframe-btn-group button");
    periodButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            periodButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activePeriod = btn.dataset.period;
            loadHistoricalData(activePeriod);
        });
    });

    // Bind Forecast Button
    const btnRunPrediction = document.getElementById("btnRunPrediction");
    if (btnRunPrediction) {
        btnRunPrediction.addEventListener("click", triggerPrediction);
    }

    // Bind Start Training Button inside Modal
    const btnStartTraining = document.getElementById("btnStartTraining");
    if (btnStartTraining) {
        btnStartTraining.addEventListener("click", triggerModelTraining);
    }

    // Initial Load
    initDashboard();
});

async function initDashboard() {
    await loadStockProfile();
    await loadHistoricalData(activePeriod);
    await checkModelStatus();
}

/**
 * Loads company overview and current price statistics.
 */
async function loadStockProfile() {
    const skeleton = document.getElementById("stockInfoSkeleton");
    const content = document.getElementById("stockInfoContent");
    if (skeleton) skeleton.classList.remove("d-none");
    if (content) content.classList.add("opacity-50");

    try {
        const res = await fetch(`/api/stock/${encodeURIComponent(activeTicker)}`);
        const json = await res.json();

        if (!json.success || !json.data) {
            showToast(json.error || "Failed to load stock data", true);
            return;
        }

        const data = json.data;
        currentCurrency = data.currency;

        document.getElementById("infoTicker").textContent = data.ticker;
        document.getElementById("infoName").textContent = data.name;
        document.getElementById("infoExchange").textContent = data.exchange;
        document.getElementById("infoMarket").textContent = data.market;

        const currSymbol = data.currency === "INR" ? "₹" : "$";
        document.getElementById("infoCurrencyPrefix").textContent = currSymbol;
        document.getElementById("infoPrice").textContent = data.current_price.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        });

        // Day change badge styling
        const changeBadge = document.getElementById("infoChangeBadge");
        const isBullish = data.day_change >= 0;
        const sign = isBullish ? "+" : "";
        changeBadge.className = `badge font-monospace px-2 py-1 ${isBullish ? 'bg-bullish-subtle' : 'bg-bearish-subtle'}`;
        changeBadge.innerHTML = `<i class="bi bi-caret-${isBullish ? 'up' : 'down'}-fill me-1"></i>${sign}${data.day_change.toFixed(2)} (${sign}${data.day_change_pct.toFixed(2)}%)`;

        // Secondary metrics
        document.getElementById("infoPrevClose").textContent = `${currSymbol}${data.previous_close.toFixed(2)}`;
        document.getElementById("infoMarketCap").textContent = formatLargeNumber(data.market_cap);
        document.getElementById("infoSector").textContent = data.sector;
        document.getElementById("infoSector").title = data.sector;

        // 52-Week Range Bar calculation
        if (data.fifty_two_low && data.fifty_two_high && data.fifty_two_high > data.fifty_two_low) {
            document.getElementById("info52Low").textContent = `${currSymbol}${data.fifty_two_low.toFixed(2)}`;
            document.getElementById("info52High").textContent = `${currSymbol}${data.fifty_two_high.toFixed(2)}`;
            const pct = ((data.current_price - data.fifty_two_low) / (data.fifty_two_high - data.fifty_two_low)) * 100;
            const clamped = Math.max(0, Math.min(100, pct));
            const pBar = document.getElementById("range52ProgressBar");
            pBar.style.width = `${clamped}%`;
            pBar.setAttribute("aria-valuenow", clamped.toFixed(0));
        }

    } catch (err) {
        showToast(`Network error: ${err.message}`, true);
    } finally {
        if (skeleton) skeleton.classList.add("d-none");
        if (content) content.classList.remove("opacity-50");
    }
}

/**
 * Fetches historical OHLCV + technical indicators and draws charts.
 */
async function loadHistoricalData(period = "1y") {
    const overlay = document.getElementById("chartLoadingOverlay");
    if (overlay) overlay.classList.remove("d-none");

    try {
        const res = await fetch(`/api/history/${encodeURIComponent(activeTicker)}?period=${period}`);
        const json = await res.json();

        if (!json.success || !json.data) {
            showToast(json.error || "Failed to load history data", true);
            return;
        }

        const records = json.data;
        renderHistoricalChart(records, currentCurrency);
        renderRsiChart(records);
        renderMacdChart(records);

    } catch (err) {
        showToast(`Error loading charts: ${err.message}`, true);
    } finally {
        if (overlay) overlay.classList.add("d-none");
    }
}

/**
 * Checks if a trained LSTM model exists for this ticker.
 */
async function checkModelStatus() {
    const badge = document.getElementById("modelStatusBadge");
    try {
        const res = await fetch(`/api/model-status/${encodeURIComponent(activeTicker)}`);
        const json = await res.json();

        if (json.success && json.is_trained && json.metadata) {
            badge.className = "badge badge-pulse-green";
            badge.innerHTML = `<i class="bi bi-check-circle me-1"></i> Model Ready`;
            populateScorecard(json.metadata);
        } else {
            badge.className = "badge badge-pulse-yellow";
            badge.innerHTML = `<i class="bi bi-exclamation-triangle me-1"></i> Needs Training`;
            resetScorecard();
        }
    } catch (err) {
        badge.className = "badge bg-secondary text-light";
        badge.textContent = "Status Unknown";
    }
}

function populateScorecard(meta) {
    const metrics = meta.test_metrics || {};
    const currSymbol = currentCurrency === "INR" ? "₹" : "$";

    document.getElementById("metricMAE").textContent = metrics.mae !== undefined ? `${currSymbol}${metrics.mae}` : "--";
    document.getElementById("metricRMSE").textContent = metrics.rmse !== undefined ? `${currSymbol}${metrics.rmse}` : "--";
    document.getElementById("metricMAPE").textContent = metrics.mape !== undefined ? `${metrics.mape}%` : "--%";
    document.getElementById("metricDirectional").textContent = metrics.directional_accuracy !== undefined ? `${metrics.directional_accuracy}%` : "--%";
    document.getElementById("metricR2").textContent = metrics.r2_score !== undefined ? `${metrics.r2_score}` : "--";
    document.getElementById("metricSamples").textContent = meta.test_samples || "--";

    document.getElementById("metaTrainedDate").textContent = meta.trained_at ? meta.trained_at.substring(0, 16).replace("T", " ") : "Recently";
    document.getElementById("metaSeqLen").textContent = `${meta.sequence_length || 60} Days`;
    document.getElementById("metaDateRange").textContent = `${meta.data_start || ''} to ${meta.data_end || ''}`;
}

function resetScorecard() {
    document.getElementById("metricMAE").textContent = "--";
    document.getElementById("metricRMSE").textContent = "--";
    document.getElementById("metricMAPE").textContent = "--%";
    document.getElementById("metricDirectional").textContent = "--%";
    document.getElementById("metricR2").textContent = "--";
    document.getElementById("metricSamples").textContent = "--";
    document.getElementById("metaTrainedDate").textContent = "Not trained yet";
    document.getElementById("metaDateRange").textContent = "--";
}

/**
 * Triggers multi-step forecasting request.
 */
async function triggerPrediction() {
    const btn = document.getElementById("btnRunPrediction");
    const overlay = document.getElementById("forecastLoadingOverlay");
    const days = parseInt(document.querySelector('input[name="forecastDays"]:checked')?.value || "7");

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Predicting...`;
    if (overlay) overlay.classList.remove("d-none");

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: activeTicker, days: days }),
        });

        const json = await res.json();

        if (!json.success) {
            if (json.needs_training) {
                showToast("This stock needs an initial LSTM model trained. Opening training dialog...", false);
                const modal = new bootstrap.Modal(document.getElementById("trainModelModal"));
                modal.show();
            } else {
                showToast(json.error || "Prediction request failed", true);
            }
            return;
        }

        displayPredictionResults(json.data, days);
        showToast(`Generated ${days}-day forecast for ${activeTicker}`);

    } catch (err) {
        showToast(`Prediction error: ${err.message}`, true);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="bi bi-magic me-2"></i><span>Generate Forecast</span>`;
        if (overlay) overlay.classList.add("d-none");
    }
}

/**
 * Renders prediction summary cards and forecast comparison chart.
 */
function displayPredictionResults(data, days) {
    document.getElementById("predictionEmptyPrompt").classList.add("d-none");
    const hero = document.getElementById("predictionHeroResult");
    hero.classList.remove("d-none");

    const currSymbol = currentCurrency === "INR" ? "₹" : "$";
    const lastPred = data.predictions[data.predictions.length - 1];

    document.getElementById("predTargetDate").textContent = lastPred.date;
    document.getElementById("predHorizonBadge").textContent = `${days} Days Ahead`;
    document.getElementById("predFinalPrice").textContent = `${currSymbol}${lastPred.predicted_price.toFixed(2)}`;

    const isPositive = data.overall_change >= 0;
    const sign = isPositive ? "+" : "";
    const changeBadge = document.getElementById("predChangeBadge");
    changeBadge.className = `badge font-monospace px-2 py-1 ${isPositive ? 'bg-bullish-subtle' : 'bg-bearish-subtle'}`;
    changeBadge.innerHTML = `${sign}${currSymbol}${data.overall_change.toFixed(2)} (${sign}${data.overall_percent_change.toFixed(2)}%)`;

    document.getElementById("predConfidenceRange").textContent =
        `${currSymbol}${lastPred.lower_bound.toFixed(2)} — ${currSymbol}${lastPred.upper_bound.toFixed(2)}`;

    // Populate steps table
    const tbody = document.getElementById("forecastTableBody");
    tbody.innerHTML = "";
    data.predictions.forEach(p => {
        const rowSign = p.change_from_latest >= 0 ? "+" : "";
        const rowColor = p.change_from_latest >= 0 ? "text-success" : "text-danger";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>+${p.step}d</td>
            <td>${p.date}</td>
            <td class="fw-bold">${currSymbol}${p.predicted_price.toFixed(2)}</td>
            <td class="${rowColor}">${rowSign}${p.percent_change.toFixed(2)}%</td>
            <td class="text-secondary">${currSymbol}${p.lower_bound.toFixed(2)} - ${currSymbol}${p.upper_bound.toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
    });

    // Render interactive chart
    renderForecastComparisonChart(data.historical_tail, data.predictions, currentCurrency);
}

/**
 * Triggers LSTM model training from the modal dialog.
 */
async function triggerModelTraining() {
    const btn = document.getElementById("btnStartTraining");
    const progressBox = document.getElementById("trainingProgressBox");
    const epochs = parseInt(document.getElementById("inputEpochs")?.value || "20");

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Training in progress...`;
    progressBox.classList.remove("d-none");

    try {
        const res = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: activeTicker, epochs: epochs }),
        });

        const json = await res.json();

        if (!json.success) {
            showToast(json.error || "Model training failed", true);
            return;
        }

        showToast(`Model successfully trained for ${activeTicker}!`);

        // Close modal
        const modalEl = document.getElementById("trainModelModal");
        const modal = bootstrap.Modal.getInstance(modalEl);
        if (modal) modal.hide();

        // Refresh model status and scorecard
        await checkModelStatus();

        // Immediately trigger prediction for seamless UX
        await triggerPrediction();

    } catch (err) {
        showToast(`Training error: ${err.message}`, true);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="bi bi-play-fill me-1"></i><span>Start Training</span>`;
        progressBox.classList.add("d-none");
    }
}
