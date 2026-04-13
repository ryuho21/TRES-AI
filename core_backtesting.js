/**
 * core_backtesting.js — Rigorous chronological backtesting engine v5.2
 *
 * FIXES (v5.2):
 * - FIX-F: L1 Markov uses log geometric mean (numerical stability)
 * - FIX-G: trainRatio exposed as parameter (80/90/95 options)
 * - Fixed pooled SE formula: uses (1/n1 + 1/n2) not (2/n1) when n1 ≠ n2
 * - scoreAll_current aligned with scoring engine (log Markov, correct mlR)
 *
 * PRESERVED from v5.1:
 * - BUG-B: Model caches built ONCE from training data only (no leakage)
 * - BUG-D: Correct erf / normalCDF / upper-tail p-value
 * - Bonferroni correction (α = 0.05 / 4 = 0.0125)
 * - Wilson score confidence intervals
 * - Strict chronological split — NO shuffling
 */

import { twoPropZTest, bonferroniAlpha, wilsonCI } from './core_statistics.js';

const N_MODELS   = 4;
const FAMILY_ALPHA = 0.05;
const BONF_ALPHA   = bonferroniAlpha(FAMILY_ALPHA, N_MODELS); // 0.0125

/**
 * Main entry point.
 *
 * @param {Array}  allDraws   - Full chronological draw list (base + user)
 * @param {number} trainRatio - Fraction for training (0.80 / 0.90 / 0.95)
 * @returns {BacktestReport}
 */
function runBacktest(allDraws, trainRatio = 0.95) {
  if (!allDraws || allDraws.length < 20) {
    return { error: `Insufficient data (need ≥ 20 draws, have ${allDraws?.length ?? 0})` };
  }

  // 1. Strict chronological split — no shuffling
  const splitIdx  = Math.floor(allDraws.length * trainRatio);
  const trainDraws = allDraws.slice(0, splitIdx);
  const testDraws  = allDraws.slice(splitIdx);

  if (testDraws.length === 0) {
    return { error: 'Test set is empty. Reduce trainRatio.' };
  }

  // 2. Build model caches from training data ONLY (BUG-B preserved)
  const trainCache = buildTrainCache(trainDraws);

  // 3. Evaluate all models over test set
  const modelNames = ['Current System', 'Markov Only', 'Frequency Only', 'Random Baseline'];
  const modelFns = {
    'Current System':  makeCurrentModel(trainCache),
    'Markov Only':     makeMarkovModel(trainCache),
    'Frequency Only':  makeFrequencyModel(trainCache),
    'Random Baseline': makeRandomModel(),
  };

  const results = {};
  for (const name of modelNames) {
    results[name] = evaluateModel(modelFns[name], testDraws, trainDraws);
  }

  // 4. Statistical test (Bonferroni-corrected)
  const cur = results['Current System'];
  const rnd = results['Random Baseline'];
  const sigTest = twoPropZTest(
    cur.exactMatches, cur.n,
    rnd.exactMatches, rnd.n,
    BONF_ALPHA
  );

  // 5. Verdict
  const verdict = determineVerdict(results, sigTest);

  return {
    metadata: {
      timestamp:  new Date().toISOString(),
      totalDraws: allDraws.length,
      trainSize:  trainDraws.length,
      testSize:   testDraws.length,
      trainRatio,
      bonferroniAlpha: BONF_ALPHA,
    },
    split: {
      trainStart: formatDraw(trainDraws[0]),
      trainEnd:   formatDraw(trainDraws[trainDraws.length - 1]),
      testStart:  formatDraw(testDraws[0]),
      testEnd:    formatDraw(testDraws[testDraws.length - 1]),
    },
    results,
    significance: sigTest,
    verdict,
    interpretation: getInterpretation(verdict, sigTest, results),
    recommendation: getRecommendation(verdict),
  };
}

/**
 * Pre-build all caches from training data — called ONCE.
 * No access to global draws inside model scoring functions.
 */
function buildTrainCache(trainDraws) {
  const trans    = Array.from({ length: 3 }, () =>
    Array.from({ length: 10 }, () => new Array(10).fill(0))
  );
  const posFreq  = Array.from({ length: 3 }, () => new Array(10).fill(0));
  const comboFreq = {};

  trainDraws.forEach(d => {
    comboFreq[d.result] = (comboFreq[d.result] || 0) + 1;
    for (let p = 0; p < 3; p++) posFreq[p][+d.result[p]]++;
  });

  const results = trainDraws.map(d => d.result);
  for (let i = 1; i < results.length; i++) {
    const prev = results[i - 1], curr = results[i];
    for (let p = 0; p < 3; p++) {
      trans[p][+prev[p]][+curr[p]]++;
    }
  }

  const transRowSums = trans.map(posMatrix =>
    posMatrix.map(row => row.reduce((a, b) => a + b, 0))
  );

  const lastSeen = {};
  trainDraws.forEach((d, idx) => { lastSeen[d.result] = idx; });

  const totalDraws    = trainDraws.length;
  const maxComboFreq  = Math.max(...Object.values(comboFreq), 1);

  return { trans, transRowSums, posFreq, comboFreq, lastSeen, totalDraws, maxComboFreq };
}

/**
 * Current multi-layer system — mirrors scoring engine logic.
 * FIX-F: Log geometric mean for Markov (numerical stability).
 */
function makeCurrentModel(cache) {
  const { trans, transRowSums, posFreq, comboFreq, lastSeen, totalDraws, maxComboFreq } = cache;

  return function scoreAll(lastResult) {
    return Array.from({ length: 1000 }, (_, n) => {
      const num = n.toString().padStart(3, '0');

      // L1: log geometric mean of per-position Markov probabilities
      let logM = 0;
      for (let p = 0; p < 3; p++) {
        const from = +lastResult[p], to = +num[p];
        const rowSum = transRowSums[p][from] || 1;
        logM += Math.log((trans[p][from][to] + 1) / (rowSum + 10));
      }
      const mScore = Math.exp(logM / 3);

      // L2: Positional digit frequency (Laplace-smoothed)
      let posScore = 1;
      for (let p = 0; p < 3; p++) {
        const total = posFreq[p].reduce((a, b) => a + b, 0);
        posScore *= (posFreq[p][+num[p]] + 1) / (total + 10);
      }

      // L3: Historical combo frequency
      const freqScore = ((comboFreq[num] || 0) + 1) / (maxComboFreq + 1);

      // L4: Recency — exponential decay (half-life 200 draws)
      const lastIdx = lastSeen[num] !== undefined ? lastSeen[num] : 0;
      const gap     = totalDraws - 1 - lastIdx;
      const recency = Math.exp(-gap / 200);

      // L5: Overdue boost
      const expectedGap = totalDraws / Math.max(comboFreq[num] || 0, 1);
      const overdue     = Math.min(2.0, gap / Math.max(expectedGap, 1));

      const score = 0.30 * mScore + 0.25 * posScore + 0.20 * freqScore +
                    0.15 * recency + 0.10 * overdue;

      return { combo: num, score };
    }).sort((a, b) => b.score - a.score);
  };
}

/**
 * Markov-only baseline. FIX-F: log geometric mean.
 */
function makeMarkovModel(cache) {
  const { trans, transRowSums } = cache;
  return function scoreAll(lastResult) {
    return Array.from({ length: 1000 }, (_, n) => {
      const num = n.toString().padStart(3, '0');
      let logM = 0;
      for (let p = 0; p < 3; p++) {
        const from = +lastResult[p], to = +num[p];
        const rowSum = transRowSums[p][from] || 1;
        logM += Math.log((trans[p][from][to] + 1) / (rowSum + 10));
      }
      return { combo: num, score: Math.exp(logM / 3) };
    }).sort((a, b) => b.score - a.score);
  };
}

function makeFrequencyModel(cache) {
  const { comboFreq } = cache;
  return function scoreAll() {
    return Array.from({ length: 1000 }, (_, n) => {
      const num = n.toString().padStart(3, '0');
      return { combo: num, score: comboFreq[num] || 0 };
    }).sort((a, b) => b.score - a.score);
  };
}

/**
 * Random baseline using seeded LCG for reproducibility.
 * Same seed per draw index → same rankings across re-runs.
 */
function makeRandomModel() {
  return function scoreAll(lastResult, drawIdx) {
    return Array.from({ length: 1000 }, (_, n) => ({
      combo: n.toString().padStart(3, '0'),
      score: lcgRand(drawIdx * 1000 + n),
    })).sort((a, b) => b.score - a.score);
  };
}

/** Linear congruential generator — reproducible random baseline */
function lcgRand(seed) {
  const a = 1664525, c = 1013904223, m = 2 ** 32;
  return ((a * seed + c) % m) / m;
}

/** Evaluate one model over the test set, rolling context forward */
function evaluateModel(scoreFn, testDraws, trainDraws) {
  let lastResult = trainDraws[trainDraws.length - 1].result;

  const metrics = {
    exact: 0,
    predictions: 0,
    ranks: [],
    topKCounts: { 1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0 },
  };

  testDraws.forEach((testDraw, idx) => {
    metrics.predictions++;
    const actual = String(testDraw.result).padStart(3, '0');

    const ranked = scoreFn(lastResult, idx);
    if (!Array.isArray(ranked) || ranked.length === 0) return;

    const rank = ranked.findIndex(p => p.combo === actual);
    if (rank >= 0) {
      metrics.ranks.push(rank + 1);
      if (rank === 0) metrics.exact++;
      for (const k of [1, 5, 10, 20, 50, 100]) {
        if (rank < k) metrics.topKCounts[k]++;
      }
    }

    lastResult = actual;
  });

  return computeFinalMetrics(metrics);
}

function computeFinalMetrics(metrics) {
  const n     = metrics.predictions;
  const exact = metrics.exact;
  const exactRate = n > 0 ? exact / n : 0;

  const ci = wilsonCI(exact, n, BONF_ALPHA);

  const topK   = {};
  const topKCI = {};
  for (const k of [1, 5, 10, 20, 50, 100]) {
    const kExact = metrics.topKCounts[k] || 0;
    topK[k]   = n > 0 ? kExact / n : 0;
    topKCI[k] = wilsonCI(kExact, n, BONF_ALPHA);
  }

  const sortedRanks = [...metrics.ranks].sort((a, b) => a - b);
  const medianRank  = sortedRanks.length > 0
    ? sortedRanks[Math.floor(sortedRanks.length / 2)]
    : 500;
  const meanRank = sortedRanks.length > 0
    ? sortedRanks.reduce((a, b) => a + b, 0) / sortedRanks.length
    : 500;

  return { n, exactMatches: exact, exactRate, exactCI: ci, topK, topKCI, meanRank: +meanRank.toFixed(1), medianRank };
}

function determineVerdict(results, sigTest) {
  const cur = results['Current System'];
  const rnd = results['Random Baseline'];
  if (!cur || !rnd) return 'UNKNOWN';

  const beatsByExact = cur.exactRate - rnd.exactRate > 0.005;
  const beatsByTop20 = cur.topK[20]  - rnd.topK[20]  > 0.03;
  const statSig      = sigTest.significant;

  if (beatsByExact && beatsByTop20 && statSig) return 'REAL SIGNAL';
  if ((beatsByExact || beatsByTop20) && sigTest.pValue < 0.05) return 'WEAK SIGNAL';
  return 'NO SIGNAL';
}

function getInterpretation(verdict, sig, results) {
  const cur    = results['Current System'];
  const rnd    = results['Random Baseline'];
  const effect = (sig.effectSize * 100).toFixed(3);
  const top20diff = (cur && rnd)
    ? ((cur.topK[20] - rnd.topK[20]) * 100).toFixed(2)
    : '?';

  const base = `Effect: ${effect}% exact-match difference. Top-20 improvement: ${top20diff}%. P-value: ${sig.pValue.toFixed(4)} (Bonferroni α=${BONF_ALPHA}).`;

  if (verdict === 'REAL SIGNAL')  return `Statistically significant predictive signal detected. ${base} Proceed to probabilistic upgrade (Option C).`;
  if (verdict === 'WEAK SIGNAL')  return `Marginal signal detected but below Bonferroni threshold. ${base} Effect size too small for practical use.`;
  return `No statistically significant predictive signal. ${base} Recommend analytics-only mode (Option A).`;
}

function getRecommendation(verdict) {
  if (verdict === 'REAL SIGNAL') return {
    path: 'OPTION C — Probabilistic Upgrade',
    actions: [
      'Add Wilson score confidence intervals per prediction',
      'Implement calibrated Bayesian model (Logistic Regression or XGBoost)',
      'Calibrate: verify predicted probability matches actual hit rate',
      'Monitor with rolling backtests as new draws arrive',
      'Mark all outputs EXPERIMENTAL with uncertainty bands',
    ],
  };

  return {
    path: 'OPTION A — Analytics Explorer',
    actions: [
      'Rebrand as "Lottery Pattern Analytics Dashboard" (not a predictor)',
      'Expose chi-square bias detection, entropy analysis, Markov heatmaps',
      'Show gap/streak/coverage analytics with clear disclaimers',
      'Disable or clearly disclaim the "Predict Next" feature',
      'Use for pattern curiosity and historical exploration only',
    ],
  };
}

function formatDraw(d) {
  if (!d) return '?';
  return `${d.month_name || ''} ${d.day || ''}, ${d.year || ''} (${d.draw_time || ''})`;
}

export { runBacktest, buildTrainCache, BONF_ALPHA };
