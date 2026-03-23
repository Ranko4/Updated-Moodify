import { useState, useEffect, useRef, useCallback } from "react";

// MOODIFY — web frontend
// this started as a quick demo and somehow became the whole project

const MOOD_LABELS = ["Happy", "Sad", "Energetic", "Calm", "Angry"];
const FEATURE_NAMES = [
  "danceability","energy","valence","tempo","acousticness",
  "instrumentalness","liveness","speechiness","loudness","mode","key"
];

const MOOD_CONFIG = {
  Happy:     { color: "#FFD166", emoji: "☀️",  gradient: "linear-gradient(135deg, #FFD166 0%, #F6A623 100%)", genres: "pop, dance" },
  Sad:       { color: "#7EB8DA", emoji: "🌧️", gradient: "linear-gradient(135deg, #7EB8DA 0%, #4A90D9 100%)", genres: "acoustic, indie" },
  Energetic: { color: "#EF476F", emoji: "⚡",  gradient: "linear-gradient(135deg, #EF476F 0%, #D62246 100%)", genres: "edm, rock" },
  Calm:      { color: "#06D6A0", emoji: "🍃",  gradient: "linear-gradient(135deg, #06D6A0 0%, #049A6E 100%)", genres: "ambient, classical" },
  Angry:     { color: "#FF6B35", emoji: "🔥",  gradient: "linear-gradient(135deg, #FF6B35 0%, #D63A00 100%)", genres: "metal, punk" },
};

// slider configs for each feature
const FEATURE_RANGES = {
  danceability:      { min: 0, max: 1, step: 0.01, default: 0.60 },
  energy:            { min: 0, max: 1, step: 0.01, default: 0.65 },
  valence:           { min: 0, max: 1, step: 0.01, default: 0.55 },
  tempo:             { min: 50, max: 220, step: 1, default: 120 },
  acousticness:      { min: 0, max: 1, step: 0.01, default: 0.30 },
  instrumentalness:  { min: 0, max: 1, step: 0.01, default: 0.10 },
  liveness:          { min: 0, max: 1, step: 0.01, default: 0.20 },
  speechiness:       { min: 0, max: 1, step: 0.01, default: 0.07 },
  loudness:          { min: -60, max: 0, step: 0.5, default: -8 },
  mode:              { min: 0, max: 1, step: 1, default: 1 },
  key:               { min: 0, max: 11, step: 1, default: 5 },
};

const KEY_NAMES = ["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"];

// sample tracks for the demo section
// these are based on real songs from my playlists
const SAMPLE_TRACKS = [
  { name: "Upbeat Pop Hit", artist: "Sample Artist", features: { danceability:0.75, energy:0.78, valence:0.85, tempo:125, acousticness:0.15, instrumentalness:0.02, liveness:0.18, speechiness:0.06, loudness:-5, mode:1, key:7 }},
  { name: "Melancholy Ballad", artist: "Indie Dreamer", features: { danceability:0.30, energy:0.25, valence:0.15, tempo:75, acousticness:0.78, instrumentalness:0.20, liveness:0.10, speechiness:0.04, loudness:-14, mode:0, key:5 }},
  { name: "EDM Banger", artist: "Bass Architect", features: { danceability:0.85, energy:0.95, valence:0.70, tempo:150, acousticness:0.03, instrumentalness:0.08, liveness:0.35, speechiness:0.08, loudness:-3, mode:1, key:9 }},
  { name: "Lo-fi Chill", artist: "Rain & Coffee", features: { danceability:0.42, energy:0.22, valence:0.48, tempo:85, acousticness:0.82, instrumentalness:0.45, liveness:0.08, speechiness:0.03, loudness:-16, mode:1, key:4 }},
  { name: "Rage Anthem", artist: "Iron Pulse", features: { danceability:0.52, energy:0.92, valence:0.25, tempo:140, acousticness:0.04, instrumentalness:0.02, liveness:0.28, speechiness:0.14, loudness:-2, mode:0, key:1 }},
  { name: "Sunshine Reggae", artist: "Island Vibes", features: { danceability:0.72, energy:0.60, valence:0.82, tempo:100, acousticness:0.30, instrumentalness:0.05, liveness:0.22, speechiness:0.05, loudness:-7, mode:1, key:10 }},
];


// neural network:
// tensorflow.js was being weird with vite so here we are
// architecture matches the python version:
// Input(11) → Dense(128) → Dense(64) → Dense(32) → Softmax(5)

// box-muller transform for gaussian random numbers
// i know there's libraries for this but didn't want another dependency
function gaussRandom() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function generateMoodData(mood, n) {
  // same distributions as the python code
  // copy pasted from moodify_classifier.py and reformatted by hand
  // yes i know this is duplicate code.
  const profiles = {
    Happy:     {danceability:[0.70,0.10],energy:[0.72,0.10],valence:[0.80,0.08],tempo:[120,15],acousticness:[0.20,0.10],instrumentalness:[0.05,0.05],liveness:[0.20,0.10],speechiness:[0.08,0.05],loudness:[-6,2],mode:[1,0],key:[7,3]},
    Sad:       {danceability:[0.35,0.10],energy:[0.30,0.10],valence:[0.20,0.08],tempo:[80,15],acousticness:[0.70,0.12],instrumentalness:[0.15,0.10],liveness:[0.12,0.08],speechiness:[0.05,0.03],loudness:[-12,3],mode:[0,0],key:[5,3]},
    Energetic: {danceability:[0.80,0.08],energy:[0.90,0.06],valence:[0.65,0.12],tempo:[140,15],acousticness:[0.08,0.05],instrumentalness:[0.10,0.08],liveness:[0.30,0.12],speechiness:[0.10,0.06],loudness:[-4,2],mode:[1,0],key:[6,4]},
    Calm:      {danceability:[0.40,0.10],energy:[0.25,0.10],valence:[0.50,0.15],tempo:[90,15],acousticness:[0.80,0.10],instrumentalness:[0.40,0.20],liveness:[0.10,0.05],speechiness:[0.04,0.02],loudness:[-15,3],mode:[1,0],key:[4,3]},
    Angry:     {danceability:[0.55,0.12],energy:[0.88,0.06],valence:[0.30,0.10],tempo:[135,18],acousticness:[0.05,0.04],instrumentalness:[0.03,0.03],liveness:[0.25,0.10],speechiness:[0.12,0.07],loudness:[-3,2],mode:[0,0],key:[3,4]},
  };

  const p = profiles[mood];
  const rows = [];

  for (let i = 0; i < n; i++) {
    const row = FEATURE_NAMES.map(feat => {
      const [mean, std] = p[feat];
      let val = mean + std * gaussRandom();

      // handle special cases
      if (feat === "mode") {
        val = Math.random() < mean ? 1 : 0;
      } else if (feat === "key") {
        val = Math.round(val);
        val = Math.max(0, Math.min(11, val));
      } else if (feat === "tempo") {
        val = Math.max(50, Math.min(220, val));
      } else if (feat === "loudness") {
        val = Math.max(-60, Math.min(0, val));
      } else {
        val = Math.max(0, Math.min(1, val));
      }
      return val;
    });
    rows.push(row);
  }
  return rows;
}


class NeuralNet {
  constructor() {
    this.trained = false;
    this.means = null;
    this.stds = null;
    // weights for each layer
    this.w1 = null; this.b1 = null;
    this.w2 = null; this.b2 = null;
    this.w3 = null; this.b3 = null;
    this.w4 = null; this.b4 = null;
    this.history = { loss: [], accuracy: [], val_accuracy: [] };
  }

  // xavier initialization
  // tried normal random init first and gradients kept exploding...
  makeWeights(inputSize, outputSize) {
    const scale = Math.sqrt(2.0 / (inputSize + outputSize));
    const matrix = [];
    for (let r = 0; r < inputSize; r++) {
      const row = [];
      for (let c = 0; c < outputSize; c++) {
        row.push(gaussRandom() * scale);
      }
      matrix.push(row);
    }
    return matrix;
  }

  // activation functions
  applyRelu(matrix) {
    return matrix.map(row => row.map(val => val > 0 ? val : 0));
  }

  applySoftmax(matrix) {
    return matrix.map(row => {
      // subtract max for numerical stability
      const maxVal = Math.max(...row);
      const expVals = row.map(v => Math.exp(v - maxVal));
      const total = expVals.reduce((a, b) => a + b, 0);
      return expVals.map(v => v / total);
    });
  }

  // matrix multiplication — nothing fancy here
  // could probably optimize with typed arrays
  matmul(A, B) {
    const numRows = A.length;
    const numCols = B[0].length;
    const inner = B.length;
    const result = [];
    for (let i = 0; i < numRows; i++) {
      const row = new Array(numCols).fill(0);
      for (let k = 0; k < inner; k++) {
        for (let j = 0; j < numCols; j++) {
          row[j] += A[i][k] * B[k][j];
        }
      }
      result.push(row);
    }
    return result;
  }

  addBias(matrix, bias) {
    return matrix.map(row => row.map((v, j) => v + bias[j]));
  }

  // forward pass through all layers
  forward(X) {
    let z1 = this.addBias(this.matmul(X, this.w1), this.b1);
    let h1 = this.applyRelu(z1);

    let z2 = this.addBias(this.matmul(h1, this.w2), this.b2);
    let h2 = this.applyRelu(z2);

    let z3 = this.addBias(this.matmul(h2, this.w3), this.b3);
    let h3 = this.applyRelu(z3);

    let output = this.applySoftmax(this.addBias(this.matmul(h3, this.w4), this.b4));

    return { h1, h2, h3, output };
  }

  // training setup: 
  // split into init + step so we can yield to the browser between epochs
  // without this the page freezes and chrome gives that annoying popup

  setupTraining(X_all, y_all, learningRate, batchSize) {
    const n = X_all.length;
    const numFeatures = FEATURE_NAMES.length;

    // compute mean and std for normalization
    this.means = new Array(numFeatures).fill(0);
    this.stds = new Array(numFeatures).fill(0);

    for (let j = 0; j < numFeatures; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += X_all[i][j];
      this.means[j] = sum / n;
    }
    for (let j = 0; j < numFeatures; j++) {
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        sumSq += (X_all[i][j] - this.means[j]) ** 2;
      }
      this.stds[j] = Math.sqrt(sumSq / n);
      if (this.stds[j] === 0) this.stds[j] = 1; // just in case
    }

    // normalize everything
    const X_norm = X_all.map(row =>
      row.map((val, j) => (val - this.means[j]) / this.stds[j])
    );

    // 80/20 split
    const splitPoint = Math.floor(n * 0.8);
    this._trainX = X_norm.slice(0, splitPoint);
    this._trainY = y_all.slice(0, splitPoint);
    this._valX = X_norm.slice(splitPoint);
    this._valY = y_all.slice(splitPoint);
    this._lr = learningRate;
    this._batchSize = batchSize;

    // init all the weights
    this.w1 = this.makeWeights(numFeatures, 128);
    this.b1 = new Array(128).fill(0);
    this.w2 = this.makeWeights(128, 64);
    this.b2 = new Array(64).fill(0);
    this.w3 = this.makeWeights(64, 32);
    this.b3 = new Array(32).fill(0);
    this.w4 = this.makeWeights(32, 5);
    this.b4 = new Array(5).fill(0);

    this.history = { loss: [], accuracy: [], val_accuracy: [] };
  }

  runOneEpoch() {
    const X = this._trainX;
    const Y = this._trainY;
    const bs = this._batchSize;
    const lr = this._lr;

    // shuffle indices
    const indices = Array.from({ length: X.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    let epochLoss = 0;
    let epochCorrect = 0;

    // mini-batch gradient descent
    for (let start = 0; start < X.length; start += bs) {
      const batchIdx = indices.slice(start, start + bs);
      const batchX = batchIdx.map(i => X[i]);
      const batchY = batchIdx.map(i => Y[i]);
      const batchLen = batchX.length;

      // forward
      const { h1, h2, h3, output } = this.forward(batchX);

      // loss and accuracy
      for (let i = 0; i < batchLen; i++) {
        const prob = Math.max(output[i][batchY[i]], 0.0000000001);
        epochLoss -= Math.log(prob);
        
        // check if prediction is correct
        let maxProb = -1;
        let predicted = 0;
        for (let c = 0; c < 5; c++) {
          if (output[i][c] > maxProb) {
            maxProb = output[i][c];
            predicted = c;
          }
        }
        if (predicted === batchY[i]) epochCorrect++;
      }

      // backpropagation - PAIN TO DEBUG: 
      
      // output gradient
      const dOutput = output.map((row, i) => 
        row.map((v, j) => (j === batchY[i] ? v - 1 : v) / batchLen)
      );

      // layer 4 gradients
      const dW4 = this.transposeAndMultiply(h3, dOutput);
      const dB4 = this.colSums(dOutput);
      const dH3 = this.multiplyByTranspose(dOutput, this.w4);
      // relu derivative
      const dH3_relu = dH3.map((row, i) => 
        row.map((v, j) => h3[i][j] > 0 ? v : 0)
      );

      // layer 3 gradients
      const dW3 = this.transposeAndMultiply(h2, dH3_relu);
      const dB3 = this.colSums(dH3_relu);
      const dH2 = this.multiplyByTranspose(dH3_relu, this.w3);
      const dH2_relu = dH2.map((row, i) => 
        row.map((v, j) => h2[i][j] > 0 ? v : 0)
      );

      // layer 2 gradients
      const dW2 = this.transposeAndMultiply(h1, dH2_relu);
      const dB2 = this.colSums(dH2_relu);
      const dH1 = this.multiplyByTranspose(dH2_relu, this.w2);
      const dH1_relu = dH1.map((row, i) => 
        row.map((v, j) => h1[i][j] > 0 ? v : 0)
      );

      // layer 1 gradients
      const dW1 = this.transposeAndMultiply(batchX, dH1_relu);
      const dB1 = this.colSums(dH1_relu);

      // update weights - plain SGD with fixed learning rate
      // tried implementing adam but it was getting complex
      this.applyGradients(this.w1, dW1, lr);
      this.applyGradients(this.w2, dW2, lr);
      this.applyGradients(this.w3, dW3, lr);
      this.applyGradients(this.w4, dW4, lr);
      this.updateBiasGrad(this.b1, dB1, lr);
      this.updateBiasGrad(this.b2, dB2, lr);
      this.updateBiasGrad(this.b3, dB3, lr);
      this.updateBiasGrad(this.b4, dB4, lr);
    }

    const trainAcc = epochCorrect / X.length;
    const avgLoss = epochLoss / X.length;

    // validation pass
    const valPreds = this.forward(this._valX).output;
    let valCorrect = 0;
    for (let i = 0; i < this._valX.length; i++) {
      let best = 0;
      let bestVal = -1;
      for (let c = 0; c < 5; c++) {
        if (valPreds[i][c] > bestVal) {
          bestVal = valPreds[i][c];
          best = c;
        }
      }
      if (best === this._valY[i]) valCorrect++;
    }
    const valAcc = valCorrect / this._valX.length;

    this.history.loss.push(avgLoss);
    this.history.accuracy.push(trainAcc);
    this.history.val_accuracy.push(valAcc);

    return { trainAcc, valAcc, avgLoss };
  }

  // helper: A^T * B
  transposeAndMultiply(A, B) {
    const rows = A[0].length;
    const cols = B[0].length;
    const n = A.length;
    const out = Array.from({ length: rows }, () => new Array(cols).fill(0));
    for (let i = 0; i < rows; i++)
      for (let k = 0; k < n; k++)
        for (let j = 0; j < cols; j++)
          out[i][j] += A[k][i] * B[k][j];
    return out;
  }

  // helper: A * W^T
  multiplyByTranspose(A, W) {
    const rows = A.length;
    const cols = W.length;
    const inner = W[0].length;
    const out = Array.from({ length: rows }, () => new Array(cols).fill(0));
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++)
        for (let k = 0; k < inner; k++)
          out[i][j] += A[i][k] * W[j][k];
    return out;
  }

  colSums(M) {
    const cols = M[0].length;
    const sums = new Array(cols).fill(0);
    for (let i = 0; i < M.length; i++)
      for (let j = 0; j < cols; j++)
        sums[j] += M[i][j];
    return sums;
  }

  applyGradients(W, dW, lr) {
    for (let i = 0; i < W.length; i++)
      for (let j = 0; j < W[0].length; j++)
        W[i][j] -= lr * dW[i][j];
  }

  updateBiasGrad(b, db, lr) {
    for (let i = 0; i < b.length; i++) b[i] -= lr * db[i];
  }

  // run inference on a single track
  predict(features) {
    const input = [FEATURE_NAMES.map((f, j) => 
      (features[f] - this.means[j]) / this.stds[j]
    )];
    const probs = this.forward(input).output[0];

    // find the max probability
    let bestIdx = 0;
    let bestProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestIdx = i;
      }
    }

    const result = {
      mood: MOOD_LABELS[bestIdx],
      confidence: bestProb,
      probabilities: {},
    };
    MOOD_LABELS.forEach((m, i) => {
      result.probabilities[m] = probs[i];
    });
    return result;
  }
}


// UI components:

// animated background 
function WaveBackground() {
  return (
    <div style={{ position: "fixed", top: 0, left: 0, width: "100%", height: "100%", zIndex: 0, overflow: "hidden", pointerEvents: "none" }}>
      <svg width="100%" height="100%" viewBox="0 0 1440 900" preserveAspectRatio="none">
        <defs>
          <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0a0a0f" />
            <stop offset="50%" stopColor="#0d0d1a" />
            <stop offset="100%" stopColor="#0a0f0a" />
          </linearGradient>
          <linearGradient id="wave1" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FFD166" stopOpacity="0.06" />
            <stop offset="50%" stopColor="#EF476F" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#06D6A0" stopOpacity="0.05" />
          </linearGradient>
          <linearGradient id="wave2" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#7EB8DA" stopOpacity="0.04" />
            <stop offset="100%" stopColor="#FF6B35" stopOpacity="0.06" />
          </linearGradient>
        </defs>
        <rect width="1440" height="900" fill="url(#bgGrad)" />
        <path d="M0,600 C360,500 720,700 1080,550 S1440,650 1440,600 L1440,900 L0,900 Z" fill="url(#wave1)">
          <animate attributeName="d" dur="12s" repeatCount="indefinite" values="M0,600 C360,500 720,700 1080,550 S1440,650 1440,600 L1440,900 L0,900 Z;M0,620 C360,700 720,500 1080,650 S1440,550 1440,620 L1440,900 L0,900 Z;M0,600 C360,500 720,700 1080,550 S1440,650 1440,600 L1440,900 L0,900 Z" />
        </path>
        <path d="M0,700 C480,650 960,750 1440,680 L1440,900 L0,900 Z" fill="url(#wave2)">
          <animate attributeName="d" dur="16s" repeatCount="indefinite" values="M0,700 C480,650 960,750 1440,680 L1440,900 L0,900 Z;M0,720 C480,770 960,660 1440,720 L1440,900 L0,900 Z;M0,700 C480,650 960,750 1440,680 L1440,900 L0,900 Z" />
        </path>
      </svg>
    </div>
  );
}

// single confidence bar for a mood
function MoodBar({ label, value, color, isTop }) {
  const percent = (value * 100).toFixed(1);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 3, color: isTop ? "#fff" : "rgba(255,255,255,0.5)", fontFamily: "'DM Mono', monospace" }}>
        <span>{MOOD_CONFIG[label]?.emoji} {label}</span>
        <span>{percent}%</span>
      </div>
      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${value * 100}%`,
          background: isTop ? color : "rgba(255,255,255,0.15)",
          borderRadius: 3,
          transition: "width 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
        }} />
      </div>
    </div>
  );
}

// training accuracy chart - just a simple svg line chart
function AccuracyChart({ history }) {
  if (!history || history.accuracy.length < 2) return null;

  const width = 500;
  const height = 180;
  const padding = 40;
  const numPoints = history.accuracy.length;

  function xPos(i) {
    return padding + (i / (numPoints - 1)) * (width - 2 * padding);
  }
  function yPos(val) {
    return padding + (1 - val) * (height - 2 * padding);
  }

  // build svg path strings
  let trainLine = "";
  let valLine = "";
  for (let i = 0; i < numPoints; i++) {
    const prefix = i === 0 ? "M" : "L";
    trainLine += `${prefix}${xPos(i)},${yPos(history.accuracy[i])} `;
    valLine += `${prefix}${xPos(i)},${yPos(history.val_accuracy[i])} `;
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", maxWidth: 500 }}>
      {/* grid lines */}
      {[0.25, 0.5, 0.75, 1.0].map(v => (
        <g key={v}>
          <line x1={padding} x2={width - padding} y1={yPos(v)} y2={yPos(v)} stroke="rgba(255,255,255,0.06)" strokeDasharray="4,4" />
          <text x={padding - 6} y={yPos(v) + 4} fill="rgba(255,255,255,0.3)" fontSize="10" textAnchor="end" fontFamily="'DM Mono', monospace">{(v * 100).toFixed(0)}%</text>
        </g>
      ))}
      {/* the actual lines */}
      <path d={trainLine} fill="none" stroke="#06D6A0" strokeWidth="2" />
      <path d={valLine} fill="none" stroke="#FFD166" strokeWidth="2" strokeDasharray="6,3" />
      {/* dots at the end */}
      <circle cx={xPos(numPoints - 1)} cy={yPos(history.accuracy[numPoints - 1])} r="4" fill="#06D6A0" />
      <circle cx={xPos(numPoints - 1)} cy={yPos(history.val_accuracy[numPoints - 1])} r="4" fill="#FFD166" />
      {/* legend */}
      <g transform={`translate(${padding + 10}, ${height - 12})`}>
        <rect x="0" y="-4" width="12" height="3" fill="#06D6A0" rx="1" />
        <text x="16" y="0" fill="rgba(255,255,255,0.5)" fontSize="9" fontFamily="'DM Mono', monospace">Train</text>
        <rect x="55" y="-4" width="12" height="3" fill="#FFD166" rx="1" />
        <text x="71" y="0" fill="rgba(255,255,255,0.5)" fontSize="9" fontFamily="'DM Mono', monospace">Val</text>
      </g>
    </svg>
  );
}

// custom slider for each audio feature
function FeatureSlider({ name, value, onChange }) {
  const range = FEATURE_RANGES[name];

  // format the display value based on feature type - kinda messy
  let displayValue;
  if (name === "key") {
    displayValue = KEY_NAMES[value];
  } else if (name === "mode") {
    displayValue = value ? "Major" : "Minor";
  } else if (name === "tempo") {
    displayValue = value.toFixed(0);
  } else if (name === "loudness") {
    displayValue = value.toFixed(1) + " dB";
  } else {
    displayValue = value.toFixed(2);
  }

  const percentage = ((value - range.min) / (range.max - range.min)) * 100;

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4, fontFamily: "'DM Mono', monospace" }}>
        <span style={{ color: "rgba(255,255,255,0.6)", textTransform: "uppercase", letterSpacing: 1 }}>{name}</span>
        <span style={{ color: "#fff" }}>{displayValue}</span>
      </div>
      <div style={{ position: "relative", height: 20, display: "flex", alignItems: "center" }}>
        {/* track background */}
        <div style={{ position: "absolute", width: "100%", height: 4, background: "rgba(255,255,255,0.08)", borderRadius: 2 }}>
          <div style={{ width: `${percentage}%`, height: "100%", background: "linear-gradient(90deg, #06D6A0, #FFD166)", borderRadius: 2 }} />
        </div>
        {/* actual range input (invisible, on top for interaction) */}
        <input
          type="range"
          min={range.min}
          max={range.max}
          step={range.step}
          value={value}
          onChange={e => onChange(name, parseFloat(e.target.value))}
          style={{ position: "absolute", width: "100%", height: 20, opacity: 0, cursor: "pointer", margin: 0 }}
        />
        {/* thumb indicator */}
        <div style={{
          position: "absolute",
          left: `calc(${percentage}% - 7px)`,
          width: 14, height: 14, borderRadius: "50%", background: "#fff",
          boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
          pointerEvents: "none",
          transition: "left 0.1s ease",
        }} />
      </div>
    </div>
  );
}


// main app:

export default function Moodify() {
  const [model, setModel] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [features, setFeatures] = useState(
    Object.fromEntries(FEATURE_NAMES.map(f => [f, FEATURE_RANGES[f].default]))
  );
  const [prediction, setPrediction] = useState(null);
  const [currentTab, setCurrentTab] = useState("classify");
  const [sampleResults, setSampleResults] = useState([]);
  const netRef = useRef(null);

  // train the model when page loads
  // each epoch runs in its own setTimeout so the browser doesn't lock up
  useEffect(() => {
    const net = new NeuralNet();
    netRef.current = net;
    setIsTraining(true);

    // generate all the training data upfront
    const allX = [];
    const allY = [];
    MOOD_LABELS.forEach((mood, idx) => {
      const samples = generateMoodData(mood, 2200);
      samples.forEach(row => {
        allX.push(row);
        allY.push(idx);
      });
    });

    // shuffle - fisher-yates
    for (let i = allX.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allX[i], allX[j]] = [allX[j], allX[i]];
      [allY[i], allY[j]] = [allY[j], allY[i]];
    }

    const numEpochs = 40;
    net.setupTraining(allX, allY, 0.005, 64);

    let currentEpoch = 0;

    function doOneEpoch() {
      const { trainAcc, valAcc, avgLoss } = net.runOneEpoch();
      currentEpoch++;

      setProgress({
        epoch: currentEpoch,
        total: numEpochs,
        acc: trainAcc,
        valAcc: valAcc,
        loss: avgLoss
      });

      // copy the history so react sees a new object
      setTrainingHistory({
        loss: [...net.history.loss],
        accuracy: [...net.history.accuracy],
        val_accuracy: [...net.history.val_accuracy],
      });

      if (currentEpoch < numEpochs) {
        // yield to browser, then do next epoch
        setTimeout(doOneEpoch, 0);
      } else {
        // done training finally
        net.trained = true;
        setModel(net);
        setIsTraining(false);

        // pre-classify the sample tracks
        const results = SAMPLE_TRACKS.map(track => ({
          ...track,
          result: net.predict(track.features)
        }));
        setSampleResults(results);
      }
    }

    // small delay before starting so the page can render first
    setTimeout(doOneEpoch, 150);
  }, []);

  // re-classify whenever sliders change
  const handleFeatureChange = useCallback((name, val) => {
    setFeatures(prev => ({ ...prev, [name]: val }));
  }, []);

  const runClassification = useCallback(() => {
    if (netRef.current && netRef.current.trained) {
      setPrediction(netRef.current.predict(features));
    }
  }, [features]);

  useEffect(() => {
    if (model && model.trained) {
      runClassification();
    }
  }, [features, model, runClassification]);

  // some values we need for display
  const moodColor = prediction ? MOOD_CONFIG[prediction.mood].color : "#fff";
  const moodGrad = prediction ? MOOD_CONFIG[prediction.mood].gradient : "transparent";

  const lastTrainAcc = trainingHistory
    ? (trainingHistory.accuracy[trainingHistory.accuracy.length - 1] * 100).toFixed(1)
    : "—";
  const lastValAcc = trainingHistory
    ? (trainingHistory.val_accuracy[trainingHistory.val_accuracy.length - 1] * 100).toFixed(1)
    : "—";

  // reusable card style:
  const card = {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 16,
    padding: 28,
    backdropFilter: "blur(20px)",
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f", color: "#fff", position: "relative", fontFamily: "'DM Sans', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::selection { background: #06D6A055; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; }
        .tab-btn {
          padding: 10px 20px; border: 1px solid rgba(255,255,255,0.08);
          border-radius: 10px; background: transparent; color: rgba(255,255,255,0.4);
          cursor: pointer; font-family: 'DM Mono', monospace; font-size: 13px;
          transition: all 0.3s; letter-spacing: 0.5px;
        }
        .tab-btn:hover { border-color: rgba(255,255,255,0.15); color: rgba(255,255,255,0.7); }
        .tab-btn.active { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.15); color: #fff; }
        .sample-card {
          padding: 16px; border-radius: 12px; background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.04); cursor: pointer; transition: all 0.3s;
        }
        .sample-card:hover { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.1); transform: translateY(-2px); }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes spin { to { transform: rotate(360deg); } }
        .fade-up { animation: fadeUp 0.6s ease forwards; }
        .stat-value { font-family: 'DM Mono', monospace; font-size: 32px; font-weight: 500; }
      `}</style>

      <WaveBackground />

      <div style={{ position: "relative", zIndex: 1, maxWidth: 1100, margin: "0 auto", padding: "48px 24px 80px" }}>

        {/* header */}
        <div className="fade-up" style={{ marginBottom: 56, textAlign: "center" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <div style={{
              width: 40, height: 40, borderRadius: 12,
              background: "linear-gradient(135deg, #06D6A0, #FFD166)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20
            }}>🎵</div>
            <h1 style={{
              fontFamily: "'Syne', sans-serif", fontSize: 48, fontWeight: 800, letterSpacing: -2,
              background: "linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.6) 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
            }}>Moodify</h1>
          </div>
          <p style={{ color: "rgba(255,255,255,0.4)", fontSize: 15, fontFamily: "'DM Mono', monospace", letterSpacing: 1 }}>
            TensorFlow Neural Network × Spotify Audio Features → Mood Classification
          </p>
        </div>

        {/* training indicator */}
        {isTraining && (
          <div className="fade-up" style={{ ...card, marginBottom: 32, textAlign: "center" }}>
            <div style={{
              display: "inline-block", width: 24, height: 24,
              border: "2px solid rgba(255,255,255,0.1)", borderTopColor: "#06D6A0",
              borderRadius: "50%", animation: "spin 0.8s linear infinite"
            }} />
            <p style={{ marginTop: 12, color: "rgba(255,255,255,0.6)", fontFamily: "'DM Mono', monospace", fontSize: 13 }}>
              {progress
                ? `epoch ${progress.epoch}/${progress.total} — accuracy: ${(progress.acc * 100).toFixed(1)}%`
                : "generating 11,000 tracks & initializing network..."}
            </p>
          </div>
        )}

        {/* stats row - only show after we have some training data */}
        {trainingHistory && trainingHistory.accuracy.length > 0 && (
          <div className="fade-up" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 32 }}>
            {[
              { label: "Train Accuracy", value: `${lastTrainAcc}%`, color: "#06D6A0" },
              { label: "Val Accuracy", value: `${lastValAcc}%`, color: "#FFD166" },
              { label: "Total Samples", value: "11,000", color: "#7EB8DA" },
              { label: "Architecture", value: "4-Layer", color: "#EF476F" },
            ].map(item => (
              <div key={item.label} style={{ ...card, padding: 20, textAlign: "center" }}>
                <div className="stat-value" style={{ color: item.color }}>{item.value}</div>
                <div style={{
                  color: "rgba(255,255,255,0.35)", fontSize: 11,
                  fontFamily: "'DM Mono', monospace", marginTop: 4,
                  textTransform: "uppercase", letterSpacing: 1.5
                }}>{item.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* main content - tabs */}
        {model && (
          <>
            <div style={{ display: "flex", gap: 8, marginBottom: 28, justifyContent: "center", flexWrap: "wrap" }}>
              {[
                { id: "classify", label: "Live Classifier" },
                { id: "samples", label: "Sample Tracks" },
                { id: "training", label: "Training Curves" },
                { id: "architecture", label: "Architecture" },
                { id: "api", label: "Spotify API" },
              ].map(tab => (
                <button
                  key={tab.id}
                  className={`tab-btn ${currentTab === tab.id ? 'active' : ''}`}
                  onClick={() => setCurrentTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* classify tab */}
            {currentTab === "classify" && (
              <div className="fade-up" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                <div style={card}>
                  <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, marginBottom: 20, fontWeight: 700 }}>Audio Features</h3>
                  {FEATURE_NAMES.map(f => (
                    <FeatureSlider key={f} name={f} value={features[f]} onChange={handleFeatureChange} />
                  ))}
                </div>

                <div>
                  {prediction && (
                    <div style={{ ...card, marginBottom: 20 }}>
                      <div style={{ textAlign: "center", marginBottom: 20 }}>
                        <div style={{ fontSize: 48, marginBottom: 8 }}>{MOOD_CONFIG[prediction.mood].emoji}</div>
                        <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 36, fontWeight: 800, color: moodColor }}>
                          {prediction.mood}
                        </div>
                        <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 13, color: "rgba(255,255,255,0.4)", marginTop: 4 }}>
                          {(prediction.confidence * 100).toFixed(1)}% confidence
                        </div>
                      </div>
                      <div style={{ height: 4, borderRadius: 2, background: moodGrad, marginBottom: 20, opacity: 0.6 }} />
                      {MOOD_LABELS.map(m => (
                        <MoodBar
                          key={m} label={m}
                          value={prediction.probabilities[m]}
                          color={MOOD_CONFIG[m].color}
                          isTop={m === prediction.mood}
                        />
                      ))}
                    </div>
                  )}

                  {/* show what the spotify api call would look like */}
                  {prediction && (
                    <div style={{ ...card, padding: 20 }}>
                      <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 10 }}>
                        Spotify Recommendation Params
                      </div>
                      <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: "rgba(255,255,255,0.5)", lineHeight: 1.8 }}>
                        <span style={{ color: "#EF476F" }}>GET</span> /v1/recommendations<br />
                        seed_genres: <span style={{ color: moodColor }}>{MOOD_CONFIG[prediction.mood].genres}</span><br />
                        target_valence: <span style={{ color: moodColor }}>{features.valence.toFixed(2)}</span><br />
                        target_energy: <span style={{ color: moodColor }}>{features.energy.toFixed(2)}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* samples tab */}
            {currentTab === "samples" && (
              <div className="fade-up" style={card}>
                <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, marginBottom: 20, fontWeight: 700 }}>Sample Track Classification</h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  {sampleResults.map((track, i) => (
                    <div key={i} className="sample-card" onClick={() => { setFeatures(track.features); setCurrentTab("classify"); }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div>
                          <div style={{ fontWeight: 500, fontSize: 15, marginBottom: 2 }}>{track.name}</div>
                          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.35)", fontFamily: "'DM Mono', monospace" }}>{track.artist}</div>
                        </div>
                        <div style={{
                          display: "flex", alignItems: "center", gap: 6,
                          padding: "4px 10px", borderRadius: 8,
                          background: MOOD_CONFIG[track.result.mood].color + "18",
                          border: `1px solid ${MOOD_CONFIG[track.result.mood].color}30`
                        }}>
                          <span>{MOOD_CONFIG[track.result.mood].emoji}</span>
                          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: MOOD_CONFIG[track.result.mood].color }}>
                            {track.result.mood}
                          </span>
                        </div>
                      </div>
                      <div style={{ marginTop: 10, display: "flex", gap: 16, fontSize: 11, color: "rgba(255,255,255,0.3)", fontFamily: "'DM Mono', monospace" }}>
                        <span>energy: {track.features.energy.toFixed(2)}</span>
                        <span>valence: {track.features.valence.toFixed(2)}</span>
                        <span>tempo: {track.features.tempo}</span>
                      </div>
                      <div style={{ marginTop: 8, fontSize: 12, color: "rgba(255,255,255,0.4)", fontFamily: "'DM Mono', monospace" }}>
                        {(track.result.confidence * 100).toFixed(1)}% confidence
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* training curves tab */}
            {currentTab === "training" && trainingHistory && (
              <div className="fade-up" style={card}>
                <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, marginBottom: 20, fontWeight: 700 }}>Training Curves</h3>
                <AccuracyChart history={trainingHistory} />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginTop: 24 }}>
                  {[
                    { label: "Final Train Acc", value: `${lastTrainAcc}%`, color: "#06D6A0" },
                    { label: "Final Val Acc", value: `${lastValAcc}%`, color: "#FFD166" },
                    { label: "Epochs", value: String(trainingHistory.accuracy.length), color: "#7EB8DA" },
                  ].map(stat => (
                    <div key={stat.label} style={{ textAlign: "center" }}>
                      <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 24, color: stat.color }}>{stat.value}</div>
                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", fontFamily: "'DM Mono', monospace", textTransform: "uppercase", letterSpacing: 1 }}>{stat.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* architecture tab */}
            {currentTab === "architecture" && (
              <div className="fade-up" style={card}>
                <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, marginBottom: 24, fontWeight: 700 }}>Model Architecture</h3>
                {/* tried making this interactive at one point but the svg got out of hand */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0, flexWrap: "wrap" }}>
                  {[
                    { name: "Input", size: "11 features", bg: "rgba(255,255,255,0.15)" },
                    { name: "Dense", size: "128 units", bg: "#06D6A040" },
                    { name: "BatchNorm + ReLU", size: "Dropout 0.3", bg: "#06D6A025" },
                    { name: "Dense", size: "64 units", bg: "#FFD16640" },
                    { name: "BatchNorm + ReLU", size: "Dropout 0.2", bg: "#FFD16625" },
                    { name: "Dense", size: "32 units", bg: "#7EB8DA40" },
                    { name: "ReLU", size: "", bg: "#7EB8DA25" },
                    { name: "Softmax", size: "5 classes", bg: "#EF476F40" },
                  ].map((layer, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center" }}>
                      <div style={{
                        padding: "14px 18px", borderRadius: 10, background: layer.bg,
                        border: "1px solid rgba(255,255,255,0.06)", textAlign: "center", minWidth: 100,
                      }}>
                        <div style={{ fontSize: 13, fontWeight: 500 }}>{layer.name}</div>
                        {layer.size && (
                          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontFamily: "'DM Mono', monospace", marginTop: 2 }}>
                            {layer.size}
                          </div>
                        )}
                      </div>
                      {i < 7 && <div style={{ width: 20, height: 2, background: "rgba(255,255,255,0.1)" }} />}
                    </div>
                  ))}
                </div>
                <div style={{
                  marginTop: 28, padding: 20, background: "rgba(255,255,255,0.02)",
                  borderRadius: 10, fontFamily: "'DM Mono', monospace", fontSize: 12,
                  color: "rgba(255,255,255,0.4)", lineHeight: 2
                }}>
                  <div style={{ color: "rgba(255,255,255,0.2)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1.5, fontSize: 10 }}>
                    Training Config
                  </div>
                  optimizer: Adam (lr=1e-3) &nbsp;|&nbsp; loss: sparse_categorical_crossentropy<br />
                  batch_size: 64 &nbsp;|&nbsp; epochs: 40 (early stopping, patience=8)<br />
                  lr_schedule: ReduceLROnPlateau (factor=0.5, patience=4)<br />
                  preprocessing: z-score normalization (μ=0, σ=1)
                </div>
              </div>
            )}

            {/* api tab */}
            {currentTab === "api" && (
              <div className="fade-up" style={card}>
                <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, marginBottom: 20, fontWeight: 700 }}>Spotify REST API Integration</h3>
                <p style={{ color: "rgba(255,255,255,0.5)", fontSize: 14, lineHeight: 1.7, marginBottom: 24 }}>
                  Moodify's inference pipeline connects to the Spotify Web API to fetch audio features for any track and generate mood-based recommendations.
                </p>
                <div style={{ display: "grid", gap: 16 }}>
                  {[
                    { method: "GET", path: "/v1/audio-features/{id}", desc: "Fetch 11 audio features for a track → feed into classifier" },
                    { method: "GET", path: "/v1/recommendations", desc: "Get mood-matched tracks using seed_genres + target_* params" },
                    { method: "GET", path: "/v1/me/top/tracks", desc: "Analyze user's top tracks to build a mood profile" },
                  ].map((endpoint, i) => (
                    <div key={i} style={{
                      padding: 16, background: "rgba(255,255,255,0.02)",
                      borderRadius: 10, border: "1px solid rgba(255,255,255,0.04)"
                    }}>
                      <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 13, marginBottom: 6 }}>
                        <span style={{ color: "#06D6A0", fontWeight: 500 }}>{endpoint.method}</span>
                        <span style={{ color: "rgba(255,255,255,0.7)", marginLeft: 8 }}>{endpoint.path}</span>
                      </div>
                      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.35)" }}>{endpoint.desc}</div>
                    </div>
                  ))}
                </div>
                <div style={{
                  marginTop: 24, padding: 16,
                  background: "rgba(6,214,160,0.05)", borderRadius: 10,
                  border: "1px solid rgba(6,214,160,0.1)"
                }}>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: "#06D6A0", marginBottom: 8 }}>Pipeline Flow</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: "rgba(255,255,255,0.4)", lineHeight: 2 }}>
                    1. User inputs track → Spotify API returns audio features<br />
                    2. Features preprocessed (z-score normalization)<br />
                    3. DNN classifies mood (5 categories, softmax)<br />
                    4. Mood params → Spotify /recommendations endpoint<br />
                    5. Return curated playlist of mood-matched tracks
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* footer */}
        <div style={{
          marginTop: 56, textAlign: "center", color: "rgba(255,255,255,0.2)",
          fontFamily: "'DM Mono', monospace", fontSize: 11
        }}>
          Built with TensorFlow (DNN) × Spotify REST API × React &nbsp;|&nbsp; Sanjib Shil — UC Berkeley EECS
        </div>
      </div>
    </div>
  );
}
