
import React from "react";

const points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const mean = [150, 140, 125, 95, 65, 80, 115, 145, 130, 100, 85, 110];
const upper = [170, 165, 155, 130, 105, 125, 155, 180, 170, 145, 135, 160];
const lower = [130, 115, 95, 60, 25, 35, 75, 110, 90, 55, 35, 60];

function GaussianPlot() {
  const x = (i) => 35 + i * 43;
  const y = (v) => 205 - v;

  const line = (values) =>
    values.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`).join(" ");

  const band =
    line(upper) +
    " " +
    lower
      .map((v, i) => `L ${x(11 - i)} ${y(lower[11 - i])}`)
      .join(" ") +
    " Z";

  return (
    <svg viewBox="0 0 520 250" className="model-chart">
      {[40, 80, 120, 160].map((v) => (
        <g key={v}>
          <line x1="30" y1={y(v)} x2="510" y2={y(v)}
            stroke="#e8eaf2" strokeDasharray="4 5" />
          <text x="5" y={y(v) + 4} fontSize="11" fill="#9298aa">
            {v}
          </text>
        </g>
      ))}

      <path d={band} fill="#7c8cf5" opacity="0.25" />

      <path d={line(mean)} fill="none" stroke="#6475ed"
        strokeWidth="3.5" strokeLinecap="round" />

      {points.map((_, i) => (
        <circle key={i} cx={x(i)} cy={y(mean[i])} r="3.5"
          fill="#6475ed" />
      ))}

      <text x="260" y="242" textAnchor="middle" fontSize="12"
        fill="#777d94">Signal progression →</text>
    </svg>
  );
}

function CalibrationPlot() {
  const before = [
    [0.1, 0.04], [0.2, 0.12], [0.3, 0.19],
    [0.4, 0.27], [0.5, 0.35], [0.6, 0.46],
    [0.7, 0.55], [0.8, 0.65], [0.9, 0.76]
  ];

  const after = [
    [0.1, 0.09], [0.2, 0.19], [0.3, 0.31],
    [0.4, 0.39], [0.5, 0.51], [0.6, 0.59],
    [0.7, 0.69], [0.8, 0.79], [0.9, 0.89]
  ];

  const x = (v) => 40 + v * 430;
  const y = (v) => 205 - v * 190;

  const makeLine = (data) =>
    data.map(([a, b], i) =>
      `${i === 0 ? "M" : "L"} ${x(a)} ${y(b)}`
    ).join(" ");

  return (
    <svg viewBox="0 0 520 250" className="model-chart">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1="40" y1={y(v)} x2="470" y2={y(v)}
            stroke="#e8eaf2" strokeDasharray="4 5" />
          <text x="5" y={y(v) + 4} fontSize="11" fill="#9298aa">
            {v.toFixed(2)}
          </text>
          <text x={x(v)} y="225" textAnchor="middle"
            fontSize="11" fill="#9298aa">{v.toFixed(2)}</text>
        </g>
      ))}

      <line x1="40" y1="205" x2="470" y2="15"
        stroke="#aeb4c5" strokeDasharray="6 5" strokeWidth="2" />

      <path d={makeLine(before)} fill="none" stroke="#e9a15a"
        strokeWidth="3" strokeDasharray="7 5" />

      <path d={makeLine(after)} fill="none" stroke="#3da995"
        strokeWidth="3.5" />

      {after.map(([a, b], i) => (
        <circle key={i} cx={x(a)} cy={y(b)} r="3.5"
          fill="#3da995" />
      ))}

      <text x="255" y="242" textAnchor="middle" fontSize="12"
        fill="#777d94">Predicted probability →</text>
    </svg>
  );
}

export default function ModelVisualizations() {
  return (
    <section className="model-section">
      <div className="model-heading">
        <h2>Probabilistic Model Analysis</h2>
        <p>Gaussian uncertainty estimation and Bayesian calibration</p>
      </div>

      <div className="model-grid">
        <article className="model-card">
          <div className="model-title">
            <div>
              <h3>Gaussian Process Regression</h3>
              <p>Predictive mean and uncertainty</p>
            </div>
            <span className="model-tag">GP</span>
          </div>

          <GaussianPlot />

          <div className="model-legend">
            <span><i className="mean-dot" /> Predictive mean</span>
            <span><i className="uncertainty-dot" /> Confidence band</span>
          </div>

          <div className="model-explanation">
            <strong>What this shows</strong>
            <p>
              The central curve represents the predicted signal.
              The shaded region represents predictive uncertainty.
              A wider band indicates greater uncertainty in the prediction.
            </p>
          </div>
        </article>

        <article className="model-card">
          <div className="model-title">
            <div>
              <h3>Bayesian Calibration</h3>
              <p>Probability reliability analysis</p>
            </div>
            <span className="model-tag">Bayesian</span>
          </div>

          <CalibrationPlot />

          <div className="model-legend">
            <span><i className="before-dot" /> Before calibration</span>
            <span><i className="after-dot" /> After calibration</span>
            <span><i className="ideal-dot" /> Ideal reliability</span>
          </div>

          <div className="model-explanation">
            <strong>What this shows</strong>
            <p>
              Calibration adjusts predicted probabilities so that
              confidence better corresponds to observed frequencies.
              The diagonal represents ideal calibration.
            </p>
          </div>
        </article>
      </div>
    </section>
  );
}