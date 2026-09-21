/**
 * Global application utilities and search normalization.
 */

// Known Indian symbols mapped for frontend auto-normalization
const INDIAN_SYMBOLS = new Set([
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "ITC", "BHARTIARTL", "HINDUNILVR", "LT",
    "KOTAKBANK", "AXISBANK", "TATAMOTORS", "MARUTI",
    "BAJFINANCE", "WIPRO", "ADANIENT", "SUNPHARMA", "TITAN", "ULTRACEMCO"
]);

function normalizeInputTicker(raw) {
    if (!raw) return "";
    let clean = raw.trim().toUpperCase();
    if (INDIAN_SYMBOLS.has(clean)) {
        return clean + ".NS";
    }
    return clean;
}

function showToast(message, isError = false) {
    const toastEl = document.getElementById("liveToast");
    const toastMsgEl = document.getElementById("toastMessage");
    if (!toastEl || !toastMsgEl) return;

    toastMsgEl.innerHTML = `
        <i class="bi ${isError ? 'bi-exclamation-octagon text-danger' : 'bi-check-circle text-accent'} fs-5"></i>
        <span>${message}</span>
    `;

    const toast = bootstrap.Toast.getOrCreateInstance(toastEl, { delay: 4000 });
    toast.show();
}

function formatCurrency(amount, currency = "USD") {
    if (amount === null || amount === undefined || isNaN(amount)) return "--";
    const symbol = currency === "INR" ? "₹" : "$";
    return symbol + parseFloat(amount).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function formatLargeNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return "--";
    const abs = Math.abs(num);
    if (abs >= 1e12) return (num / 1e12).toFixed(2) + "T";
    if (abs >= 1e9) return (num / 1e9).toFixed(2) + "B";
    if (abs >= 1e6) return (num / 1e6).toFixed(2) + "M";
    if (abs >= 1e3) return (num / 1e3).toFixed(2) + "K";
    return num.toLocaleString();
}

// Hook search forms to normalize Indian tickers automatically
document.addEventListener("DOMContentLoaded", () => {
    const heroForm = document.getElementById("heroSearchForm");
    if (heroForm) {
        heroForm.addEventListener("submit", (e) => {
            const input = document.getElementById("heroTickerInput");
            if (input) {
                input.value = normalizeInputTicker(input.value);
            }
        });
    }

    const navForms = document.querySelectorAll(".nav-search-form");
    navForms.forEach(form => {
        form.addEventListener("submit", (e) => {
            const input = form.querySelector("input[name='ticker']");
            if (input) {
                input.value = normalizeInputTicker(input.value);
            }
        });
    });

    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
});
