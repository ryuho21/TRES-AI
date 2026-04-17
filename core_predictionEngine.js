/**
 * core_predictionEngine.js — 11-Layer scoring engine v5.4
 *
 * FIXES (v5.4):
 * - FIX-O: L8 Hot-Digit-Per-Position added — geometric mean of per-position
 *          digit frequencies over last 50 draws. Empirically best single signal
 *          (2.70% top-20 vs 2.40%). Requires posHot50 passed in context.
 * - FIX-P: Recency window 30→50 draws, half-life 10→15 (caller must pass
 *          decayWeights computed with windowSize=50, halfLife=15).
 * - FIX-Q: L5a uses recentSumFreq (last-30 draw sums, normalized [0,1])
 *          instead of all-time S.sum_freq. Caller computes and passes it.
 * - Season (L7) weight set to 0 — kept in struct for backward compat.
 *
 * PRESERVED from v5.2:
 * - FIX-A: L1b uses live dtrans2 (100% state coverage)
 * - FIX-B: L2 ML position uses sum/3
 * - FIX-C: Slot pos_freq fallback scale-guarded
 * - FIX-F: L1 Markov uses log geometric mean
 *
 * All layer scores normalized to [0,1] BEFORE weighting. Weights sum validated.
 */

/**
 * Default layer weights (must sum to 1.0).
 * v5.4: L8 hotpos added at 0.18 (proven best single signal).
 * Season reduced to 0.00 (statistically baseless — 1/12 months significant by chance).
 * ml reduced to 0.02 (spread 0.01-0.02, near-zero discrimination).
 */
const DEFAULT_WEIGHTS = {
  markov:  0.28, // L1:  1st-order Markov (log geometric mean)
  ord2:    0.14, // L1b: 2nd-order Markov (live-computed, full coverage)
  ml:      0.02, // L2:  ML positional frequency (near-zero signal)
  slot:    0.11, // L3:  Slot-time positional bias
  dow:     0.07, // L4a: Day-of-week digit bias
  rec:     0.09, // L4b: Recency exponential decay (last 50, half-life 15)
  sum:     0.05, // L5a: Recent digit sum frequency (last 30 draws)
  overdue: 0.04, // L5b: Gap + inverse combo frequency
  pair:    0.02, // L6:  Cross-position pair transitions
  season:  0.00, // L7:  Monthly phase (disabled — no statistical basis)
  hotpos:  0.18, // L8:  Hot-digit per position (last 50 draws) ← NEW
};

/**
 * Normalize weights: only sums known DEF_W keys (guards against orphan keys
 * from old saved weight objects skewing the total).
 */
function normalizeWeights(weights) {
  const keys = Object.keys(DEFAULT_WEIGHTS);
  const total = keys.reduce((s, k) => s + (+(weights[k]) || 0), 0);
  if (total === 0) return { weights: { ...DEFAULT_WEIGHTS }, valid: false, total: 0 };
  const normalized = {};
  keys.forEach(k => { normalized[k] = (+(weights[k]) || 0) / total; });
  return { weights: normalized, valid: Math.abs(total - 1) < 0.01, total: +total.toFixed(4) };
}

/**
 * Build live 2nd-order Markov transition table from draw history.
 *
 * FIX-A: Replaces the stale dataset dtrans2 (which only had 2/100 states).
 * Returns liveDtrans2[pos][state] = { digit: percentage }
 * where state = prevDigit + currDigit (e.g. "74"), percentage sums to ~100.
 *
 * @param {Array} draws - Chronological draw history
 * @returns {Array} 3-element array of state→digit→percentage maps
 */
function buildLiveDtrans2(draws) {
  // Accumulate counts
  const counts = Array.from({ length: 3 }, () => ({}));
  for (let i = 2; i < draws.length; i++) {
    const r2 = draws[i - 2].result;
    const r1 = draws[i - 1].result;
    const r0 = draws[i].result;
    for (let p = 0; p < 3; p++) {
      const state = r2[p] + r1[p]; // "00".."99"
      if (!counts[p][state]) counts[p][state] = new Array(10).fill(0);
      counts[p][state][+r0[p]]++;
    }
  }
  // Convert to percentage maps
  return counts.map(posStates => {
    const out = {};
    Object.entries(posStates).forEach(([state, arr]) => {
      const tot = arr.reduce((a, b) => a + b, 0) || 1;
      out[state] = Object.fromEntries(
        arr.map((c, d) => [String(d), +(c / tot * 100).toFixed(2)])
      );
    });
    return out;
  });
}

/**
 * Validate and provide safe defaults for missing S properties.
 * Prevents crashes when dataset fields are absent.
 */
function validateAndFixS(S) {
  if (!S) S = {};
  const uniform10 = Object.fromEntries(Array.from({ length: 10 }, (_, d) => [String(d), 10]));
  const uniformPosFreq = [uniform10, uniform10, uniform10];

  if (!S.slot_stats) {
    S.slot_stats = { '2pm': { pos_freq: uniformPosFreq }, '5pm': { pos_freq: uniformPosFreq }, '9pm': { pos_freq: uniformPosFreq } };
  }
  if (!S.pos_freq) S.pos_freq = uniformPosFreq;
  if (!S.dow_stats) S.dow_stats = {};
  if (!S.sum_freq) { S.sum_freq = {}; for (let i = 0; i <= 27; i++) S.sum_freq[i] = 1; }
  if (!S.dtrans2) S.dtrans2 = [{}, {}, {}];
  if (!S.ml_pos) {
    const uniform = Object.fromEntries(Array.from({ length: 10 }, (_, d) => [String(d), 0.1]));
    S.ml_pos = { '0': uniform, '1': uniform, '2': uniform };
  }
  if (!S.monthly_stats) {
    S.monthly_stats = {};
    for (let m = 1; m <= 12; m++) {
      S.monthly_stats[m] = { digit_freq: Object.fromEntries(Array.from({ length: 10 }, (_, d) => [String(d), 10])) };
    }
  }
  if (!S.meta) S.meta = { total: 0 };
  return S;
}

/**
 * FIX-C: Get slot positional frequency, normalizing scale if needed.
 * slot_stats.pos_freq contains percentages (~10.0).
 * Fallback S.pos_freq contains raw counts (~1750).
 * Detect and normalize to percentages in both cases.
 */
function getSlotPosFreq(S, slot) {
  const raw = S.slot_stats[slot]?.pos_freq;
  if (raw) {
    // Detect scale: if first value > 50, it's raw counts not percentages
    const probe = +Object.values(raw[0])[0];
    if (probe > 50) {
      return raw.map(p => {
        const tot = Object.values(p).reduce((a, b) => a + b, 0) || 1;
        return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, v / tot * 100]));
      });
    }
    return raw;
  }
  // Fallback: normalize S.pos_freq (raw counts) to percentages
  return S.pos_freq.map(p => {
    const tot = Object.values(p).reduce((a, b) => a + b, 0) || 1;
    return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, v / tot * 100]));
  });
}

/**
 * Build scoring context — called once per prediction run.
 *
 * @param {Array} posHot50       - Per-position digit probabilities over last 50 draws
 *                                 [3][10] float array, each row sums to 1. (FIX-O)
 * @param {Array} recentSumFreq  - Normalized digit-sum frequencies over last 30 draws
 *                                 28-element float array indexed by digit sum. (FIX-Q)
 */
function buildScoringContext(S, mergedTrans, transRowSums, mergedComboFreq, liveGaps, decayWeights, liveDtrans2, slot, dow, drawHistory, posHot50, recentSumFreq) {
  S = validateAndFixS(S);

  const lastResult = drawHistory.length > 0 ? drawHistory[drawHistory.length - 1].result : '000';
  const prevResult = drawHistory.length > 1 ? drawHistory[drawHistory.length - 2].result : lastResult;

  // FIX-A: Use live dtrans2; fallback to S.dtrans2 only if live unavailable
  const dtrans2 = liveDtrans2 || S.dtrans2 || [{}, {}, {}];

  // 2nd-order state per position: prev_digit + last_digit
  const ord2State = [0, 1, 2].map(p => prevResult[p] + lastResult[p]);

  // FIX-C: Scale-guarded slot positional frequency
  const slotPosFreq = getSlotPosFreq(S, slot);

  const dowFreq = S.dow_stats[dow]?.digit_freq || {};

  // FIX-Q: Use recent sum frequency (last 30 draws) — fallback to uniform if absent
  const sumFreqArr = recentSumFreq || new Array(28).fill(1 / 28);
  const maxSumFreq = Math.max(...sumFreqArr, 0.001);

  const maxGap = Math.max(...liveGaps.flat().filter(v => typeof v === 'number' && v < drawHistory.length), 1);
  const maxComboFreq = Math.max(...Object.values(mergedComboFreq), 1);

  // FIX-O: Per-position hot-digit table (last 50 draws) — fallback to uniform
  const pH = posHot50 || Array.from({ length: 3 }, () => new Array(10).fill(0.1));

  return {
    lastResult, prevResult, ord2State, dtrans2,
    slotPosFreq, dowFreq, sumFreqArr, maxSumFreq,
    maxGap, maxComboFreq, pH,
    mergedTrans, transRowSums, mergedComboFreq, liveGaps, decayWeights,
    S,
  };
}

/**
 * Score a single combo against all 11 layers.
 * Returns raw (un-normalized) scores per layer.
 */
function scoreCombo(num, ctx) {
  const [d0, d1, d2] = [+num[0], +num[1], +num[2]];
  const {
    lastResult, ord2State, dtrans2, slotPosFreq, dowFreq,
    sumFreqArr, maxSumFreq, maxGap, mergedTrans, transRowSums,
    mergedComboFreq, liveGaps, decayWeights, maxComboFreq, pH, S,
  } = ctx;

  // L1: 1st-order Markov — FIX-F: log geometric mean (cube root of product)
  let logM = 0;
  for (let p = 0; p < 3; p++) {
    const from = +lastResult[p], to = +num[p];
    const rowSum = transRowSums[p][from] || 1;
    logM += Math.log((mergedTrans[p][from][to] + 1) / (rowSum + 10));
  }
  const mRaw = Math.exp(logM / 3);

  // L1b: 2nd-order Markov — FIX-A: live dtrans2 with full state coverage
  let o2Raw = 0;
  for (let p = 0; p < 3; p++) {
    const stateData = dtrans2?.[p]?.[ord2State[p]] || {};
    const val = parseFloat(stateData[String(+num[p])]);
    o2Raw += (isNaN(val) ? 10 : val) / 100;
  }
  o2Raw /= 3;

  // L2: ML positional frequency — FIX-B: sum/3
  let mlRaw = 0;
  for (let p = 0; p < 3; p++) {
    mlRaw += parseFloat(S.ml_pos?.[String(p)]?.[num[p]]) || 0.001;
  }
  mlRaw /= 3;

  // L3: Slot-specific positional bias (FIX-C: pre-normalized to percentages)
  const sl0 = +(slotPosFreq[0]?.[String(d0)] || 10);
  const sl1 = +(slotPosFreq[1]?.[String(d1)] || 10);
  const sl2 = +(slotPosFreq[2]?.[String(d2)] || 10);
  const slRaw = (sl0 + sl1 + sl2) / 30;

  // L4a: Day-of-week digit bias
  const dw0 = +(dowFreq[String(d0)] || 10);
  const dw1 = +(dowFreq[String(d1)] || 10);
  const dw2 = +(dowFreq[String(d2)] || 10);
  const dwRaw = (dw0 + dw1 + dw2) / 30;

  // L4b: Exponential decay recency (FIX-P: caller uses window=50, halfLife=15)
  const rcRaw = (decayWeights[d0] + decayWeights[d1] + decayWeights[d2]) / 3;

  // L5a: Recent digit sum frequency (FIX-Q: last-30 draw sums, not all-time)
  const sv = d0 + d1 + d2;
  const suRaw = (sv <= 27 ? (sumFreqArr[sv] || 0) : 0) / maxSumFreq;

  // L5b: Gap/overdue — log-scaled per-position gaps + inverse combo frequency
  const g0 = Math.min(liveGaps[0][d0], maxGap);
  const g1 = Math.min(liveGaps[1][d1], maxGap);
  const g2 = Math.min(liveGaps[2][d2], maxGap);
  const logMaxG = Math.log(maxGap + 1);
  const gapRaw = (Math.log(g0 + 1) + Math.log(g1 + 1) + Math.log(g2 + 1)) / (3 * logMaxG);
  const comboFreq = mergedComboFreq[num] || 0;
  const overdueRaw = 0.5 * gapRaw + 0.5 * (1 - comboFreq / maxComboFreq);

  // L6: Cross-position pair transition probability
  const p01 = transRowSums[0][d0] > 0
    ? (mergedTrans[0][d0][d1] + 1) / (transRowSums[0][d0] + 10)
    : 0.1;
  const p12 = transRowSums[1][d1] > 0
    ? (mergedTrans[1][d1][d2] + 1) / (transRowSums[1][d1] + 10)
    : 0.1;
  const pairRaw = Math.sqrt(p01 * p12) * 10;

  // L7: Season — disabled (weight=0), kept for struct compatibility
  const seasonRaw = 0;

  // L8: Hot-Digit-Per-Position (FIX-O) — geometric mean of per-position frequencies
  // pH[pos][digit] = P(digit at pos) over last 50 draws. Cube root = geometric mean.
  const hpRaw = Math.cbrt((pH[0][d0] || 1e-6) * (pH[1][d1] || 1e-6) * (pH[2][d2] || 1e-6));

  return { mRaw, o2Raw, mlRaw, slRaw, dwRaw, rcRaw, suRaw, overdueRaw, pairRaw, seasonRaw, hpRaw };
}

/**
 * Main async scoring function — processes 1000 combos in chunks of 100.
 * Yields between chunks to keep the UI responsive.
 *
 * @param {Object} S              - Base statistical dataset
 * @param {Array}  mergedTrans    - Augmented 1st-order transition matrices [3][10][10]
 * @param {Array}  transRowSums   - Row sums for normalization [3][10]
 * @param {Object} mergedComboFreq - Merged combo frequency map
 * @param {Array}  liveGaps       - Live digit gaps per position [3][10]
 * @param {Array}  decayWeights   - Per-digit exp decay weights [10] (window=50, hl=15)
 * @param {Array}  liveDtrans2    - Live 2nd-order transitions (from buildLiveDtrans2)
 * @param {string} slot           - Draw slot ('2pm'/'5pm'/'9pm')
 * @param {string} dow            - Day of week key ('M','T','W','TH','F','S','SU')
 * @param {Array}  drawHistory    - All draws in chronological order
 * @param {Array}  posHot50       - Per-position digit probabilities last 50 draws [3][10]
 * @param {Array}  recentSumFreq  - Digit-sum frequency last 30 draws (normalized) [28]
 * @param {Object} weights        - Layer weights (normalized internally)
 * @param {Function} onProgress   - Progress callback (0–100)
 * @returns {Promise<Array>} Sorted array of { num, score, scoreDisplay, layers }
 */
async function computeScoresAsync(
  S, mergedTrans, transRowSums, mergedComboFreq,
  liveGaps, decayWeights, liveDtrans2,
  slot, dow, drawHistory,
  posHot50, recentSumFreq,
  weights = DEFAULT_WEIGHTS, onProgress = null
) {
  const { weights: wn } = normalizeWeights(weights);
  const ctx = buildScoringContext(
    S, mergedTrans, transRowSums, mergedComboFreq,
    liveGaps, decayWeights, liveDtrans2,
    slot, dow, drawHistory,
    posHot50, recentSumFreq
  );

  const rawScores = new Array(1000);
  const CHUNK = 100;

  for (let start = 0; start < 1000; start += CHUNK) {
    const end = Math.min(start + CHUNK, 1000);
    for (let n = start; n < end; n++) {
      rawScores[n] = { n, ...scoreCombo(n.toString().padStart(3, '0'), ctx) };
    }
    if (onProgress) onProgress(Math.round(end / 10));
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  // Min-max normalize each layer across all 1000 combos
  const layerKeys = ['mRaw','o2Raw','mlRaw','slRaw','dwRaw','rcRaw','suRaw','overdueRaw','pairRaw','hpRaw'];
  const maxVals = {}, minVals = {};
  layerKeys.forEach(k => {
    const vals = rawScores.map(s => s[k]);
    maxVals[k] = Math.max(...vals) || 1;
    minVals[k] = Math.min(...vals);
  });

  const layerToWeight = {
    mRaw: wn.markov, o2Raw: wn.ord2,  mlRaw: wn.ml,
    slRaw: wn.slot,  dwRaw: wn.dow,   rcRaw: wn.rec,
    suRaw: wn.sum,   overdueRaw: wn.overdue, pairRaw: wn.pair,
    hpRaw: wn.hotpos,
  };

  const results = rawScores.map(s => {
    const num = s.n.toString().padStart(3, '0');
    const normalized = {};
    layerKeys.forEach(k => {
      const range = maxVals[k] - minVals[k];
      normalized[k] = range > 0 ? (s[k] - minVals[k]) / range : 0.5;
    });

    let combined = 0;
    layerKeys.forEach(k => { combined += layerToWeight[k] * normalized[k]; });

    return {
      num,
      score: +combined.toFixed(4),
      scoreDisplay: +(combined * 100).toFixed(1),
      layers: {
        markov:  Math.round(normalized.mRaw * 100),
        ord2:    Math.round(normalized.o2Raw * 100),
        ml:      Math.round(normalized.mlRaw * 100),
        slot:    Math.round(normalized.slRaw * 100),
        dow:     Math.round(normalized.dwRaw * 100),
        rec:     Math.round(normalized.rcRaw * 100),
        sum:     Math.round(normalized.suRaw * 100),
        overdue: Math.round(normalized.overdueRaw * 100),
        pair:    Math.round(normalized.pairRaw * 100),
        hotpos:  Math.round(normalized.hpRaw * 100),
      },
    };
  });

  if (onProgress) onProgress(100);
  return results.sort((a, b) => b.score - a.score);
}

export { computeScoresAsync, normalizeWeights, buildLiveDtrans2, getSlotPosFreq, DEFAULT_WEIGHTS, buildScoringContext };
