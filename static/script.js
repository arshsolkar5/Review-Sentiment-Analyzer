/**
 * SentimentScope — script.js
 * Frontend interactions for the Flask-backed sentiment predictor.
 */
let evaluationMetrics = null;
let activeModel = "lr";
let metricsChart = null;
let confidenceChart = null;

const EXAMPLES = {
  positive: `Absolutely love this product! It exceeded all my expectations. The build quality is fantastic and it arrived well ahead of the estimated delivery date. Setup was effortless and it works perfectly out of the box. Highly recommend to anyone looking for reliable, great value for money.`,
  negative: `Extremely disappointed with this purchase. The product stopped working after just two days and the material feels incredibly cheap — nothing like what was shown in the photos. Customer support was unhelpful and dismissive. Total waste of money. Would give zero stars if I could.`
};

function chartBaseOptions(maxY) {
  return {
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: "#e6edf3"
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: "#e6edf3"
        },
        grid: {
          color: "rgba(255,255,255,0.06)"
        }
      },
      y: {
        beginAtZero: true,
        max: maxY,
        ticks: {
          color: "#8b949e"
        },
        grid: {
          color: "rgba(255,255,255,0.08)"
        }
      }
    }
  };
}

function barGradient(context, startColor, endColor) {
  const chart = context.chart;
  const { ctx, chartArea } = chart;

  if (!chartArea) {
    return startColor;
  }

  const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
  gradient.addColorStop(0, startColor);
  gradient.addColorStop(1, endColor);
  return gradient;
}

function fillExample(type) {
  const textarea = document.getElementById("review-input");
  textarea.value = EXAMPLES[type];
  textarea.focus();
  textarea.style.transition = "box-shadow 0.3s";
  textarea.style.boxShadow = `0 0 0 3px ${type === "positive"
    ? "rgba(46,204,113,0.25)"
    : "rgba(231,76,60,0.25)"}`;
  setTimeout(() => { textarea.style.boxShadow = ""; }, 1000);
}

function renderCard(cardId, sentimentId, barId, pctId, data) {
  const card = document.getElementById(cardId);
  const label = document.getElementById(sentimentId);
  const bar = document.getElementById(barId);
  const pct = document.getElementById(pctId);
  const cls = data.sentiment.toLowerCase();

  card.className = `model-card is-${cls}`;
  label.textContent = data.sentiment;
  label.className = `model-sentiment sentiment-${cls}`;
  bar.className = `conf-bar bar-${cls}`;

  requestAnimationFrame(() => {
    bar.style.width = `${data.confidence}%`;
  });

  pct.textContent = `${data.confidence}%`;
}

function updateComparison(prediction, elapsedMs) {
  const confidenceDiff = Math.abs(prediction.lr.confidence - prediction.nb.confidence).toFixed(2);
  const samePrediction = prediction.lr.sentiment === prediction.nb.sentiment;
  const higherConfidenceModel =
    prediction.lr.confidence === prediction.nb.confidence
      ? "Tie"
      : prediction.lr.confidence > prediction.nb.confidence
        ? "Logistic Regression"
        : "Naive Bayes";

  const preferredModel = evaluationMetrics ? evaluationMetrics.preferred_model : null;
  const winnerContext = preferredModel && preferredModel !== "Tie"
    ? `${preferredModel} is also the stronger model overall on the held-out evaluation split.`
    : "Overall evaluation results show both models performing at a similar level.";

  const summary = samePrediction
    ? `Both models classify this review as ${prediction.lr.sentiment}. ${higherConfidenceModel === "Tie" ? "Their confidence scores are aligned for this input." : `${higherConfidenceModel} is more confident by ${confidenceDiff}%.`} ${winnerContext}`
    : `The models disagree on this review: Logistic Regression predicts ${prediction.lr.sentiment}, while Naive Bayes predicts ${prediction.nb.sentiment}. ${higherConfidenceModel === "Tie" ? "Their confidence scores are tied, so this input is comparatively ambiguous." : `${higherConfidenceModel} has the higher confidence by ${confidenceDiff}%, which makes it the stronger choice for this specific input.`} ${winnerContext}`;

  document.getElementById("confidence-diff").textContent = `${confidenceDiff}%`;
  document.getElementById("analysis-time").textContent = `${elapsedMs} ms`;
  document.getElementById("model-agreement").textContent = samePrediction ? "Yes" : "No";
  document.getElementById("higher-confidence").textContent = higherConfidenceModel;
  document.getElementById("prediction-summary").textContent = summary;
}

function updateEvaluationSummary(modelKey) {
  const metrics = evaluationMetrics[modelKey];
  document.getElementById("correct-count").textContent = metrics.correct;
  document.getElementById("incorrect-count").textContent = metrics.incorrect;
  document.getElementById("total-count").textContent = metrics.total;
  document.getElementById("model-conclusion").textContent = evaluationMetrics.conclusion;
}

function renderConfusionMatrix(modelKey) {
  const metrics = evaluationMetrics[modelKey];
  const labels = metrics.labels;
  const matrix = metrics.confusion_matrix;

  const head = document.getElementById("confusion-matrix-head");
  const body = document.getElementById("confusion-matrix-body");

  head.innerHTML = `
    <tr>
      <th>Actual \\ Predicted</th>
      ${labels.map((label) => `<th>${label}</th>`).join("")}
    </tr>
  `;

  body.innerHTML = labels.map((label, rowIndex) => `
    <tr>
      <th>${label}</th>
      ${matrix[rowIndex].map((value) => `<td>${value}</td>`).join("")}
    </tr>
  `).join("");
}

function renderMetricsChart() {
  if (metricsChart) {
    metricsChart.destroy();
  }

  const metrics = evaluationMetrics[activeModel];
  const isLogisticRegression = activeModel === "lr";

  metricsChart = new Chart(document.getElementById("metricsChart"), {
    type: "bar",
    data: {
      labels: ["Accuracy", "Precision", "Recall", "F1 Score"],
      datasets: [{
        label: isLogisticRegression ? "LR Performance" : "NB Performance",
        data: [
          metrics.accuracy,
          metrics.precision,
          metrics.recall,
          metrics.f1
        ],
        backgroundColor: isLogisticRegression
          ? ["#58d9c2", "#2ecc71", "#f0c040", "#e67e22"]
          : ["#3498db", "#9b59b6", "#f39c12", "#e74c3c"],
        borderWidth: 0,
        borderRadius: 0
      }]
    },
    options: chartBaseOptions(100)
  });
}

function renderConfidenceChart(prediction) {
  if (confidenceChart) {
    confidenceChart.destroy();
  }

  confidenceChart = new Chart(document.getElementById("confidenceChart"), {
    type: "bar",
    data: {
      labels: ["Logistic Regression", "Naive Bayes"],
      datasets: [{
        label: "Confidence %",
        data: [prediction.lr.confidence, prediction.nb.confidence],
        backgroundColor: ["#58d9c2", "#2ecc71"],
        borderWidth: 0,
        borderRadius: 0
      }]
    },
    options: chartBaseOptions(100)
  });
}

function showMatrix(modelKey) {
  activeModel = modelKey;
  document.getElementById("lr-toggle").classList.toggle("active", modelKey === "lr");
  document.getElementById("nb-toggle").classList.toggle("active", modelKey === "nb");
  renderConfusionMatrix(modelKey);
  updateEvaluationSummary(modelKey);
  renderMetricsChart();
}

function predictSentiment() {
  const startTime = performance.now();
  const text = document.getElementById("review-input").value.trim();

  if (!text) {
    shakeTextarea();
    return;
  }

  const btn = document.querySelector(".predict-btn");
  btn.classList.add("loading");
  btn.disabled = true;

  Promise.all([
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ review: text })
    }).then((response) => response.json()),
    evaluationMetrics
      ? Promise.resolve(evaluationMetrics)
      : fetch("/metrics").then((response) => response.json())
  ])
    .then(([prediction, metrics]) => {
      if (prediction.error) {
        throw new Error(prediction.error);
      }
      if (metrics.error) {
        throw new Error(metrics.error);
      }

      evaluationMetrics = metrics;

      const section = document.getElementById("results");
      section.style.display = "block";

      renderCard("lr-card", "lr-sentiment", "lr-bar", "lr-pct", prediction.lr);
      renderCard("nb-card", "nb-sentiment", "nb-bar", "nb-pct", prediction.nb);

      const endTime = performance.now();
      const totalTime = (endTime - startTime).toFixed(2);

      updateComparison(prediction, totalTime);
      showMatrix(activeModel);
      renderConfidenceChart(prediction);

      section.scrollIntoView({ behavior: "smooth", block: "start" });
      btn.classList.remove("loading");
      btn.disabled = false;
    })
    .catch((error) => {
      btn.classList.remove("loading");
      btn.disabled = false;
      alert(`Prediction failed: ${error.message}`);
    });
}

function shakeTextarea() {
  const ta = document.getElementById("review-input");
  ta.style.transition = "transform 0.07s, border-color 0.2s";
  ta.style.borderColor = "#e74c3c";
  const shakes = [6, -6, 5, -5, 3, 0];
  shakes.forEach((x, i) => {
    setTimeout(() => {
      ta.style.transform = `translateX(${x}px)`;
      if (i === shakes.length - 1) {
        ta.style.borderColor = "";
      }
    }, i * 60);
  });
}

document.getElementById("review-input").addEventListener("keydown", function (e) {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    predictSentiment();
  }
});
