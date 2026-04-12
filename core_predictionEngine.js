/**
 * predictionEngine.js — 6-layer scoring engine
 *
 * FIXES:
 * - BUG-E: computeScores() uses chunked async processing
 * - All layer scores normalized to [0,1] BEFORE weighting
 * - No reference to synthetic base data; uses precomputed S aggregates
 * - Weight validation ensures sum ≈ 1.0
 */

/**
 * Default layer weights (sum = 1.0)
 * These can be overridden by user slider settings.
 */
const DEFAULT_WEIGHTS = {
  markov:  0.28,  // L1: 1st-order Markov
  ord2:    0.08,  // L1b: 2nd-order Markov
  ml:      0.22,  // L2: ML positional frequency
  slot:    0.12,  // L3: Slot-time bias
  dow:     0.09,  // L4a: Day-of-week
  rec:     0.07,  // L4b: Live recency (decay)
  sum:     0.05,  // L5a: Digit sum distribution
  overdue: 0.05,  // L5b: Gap / overdue
  pair:    0.02,  // L6: Positional pair transitions
  season:  0.02,  // L7: Monthly phase
};

/**
 * Validate and normalize weights to sum = 1.0.
 * Returns normalized weights + validation report.
 */
function normalizeWeights(weights) {
  const keys = Object.keys(DEFAULT_WEIGHTS);
  // Only sum and normalize known layer keys — guards against stale/orphan keys
  // that may persist from old saved weight objects.
  const total = keys.reduce((s, k) => s + (+(weights[k]) || 0), 0);
  if (total === 0) return { weights: { ...DEFAULT_WEIGHTS }, valid: false, total: 0 };
  const normalized = {};
  keys.forEach(k => { normalized[k] = (+(weights[k]) || 0) / total; });
  return { weights: normalized, valid: Math.abs(total - 1) < 0.01, total: +total.toFixed(4) };
}

/**
 * Validate S structure and provide defaults for missing properties.
 * Ensures graceful degradation if S is incomplete.
 */
function validateAndFixS(S) {
  if (!S) S = {};
  
  // Ensure slot_stats exists
  if (!S.slot_stats) {
    S.slot_stats = {
      '2pm': { pos_freq: [{'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}] },
      '5pm': { pos_freq: [{'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}] },
      '9pm': { pos_freq: [{'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}] }
    };
  }
  
  // Ensure pos_freq exists as fallback
  if (!S.pos_freq) {
    S.pos_freq = [{'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}, {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10}];
  }
  
  // Ensure dow_stats exists
  if (!S.dow_stats) {
    S.dow_stats = {};
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].forEach(dow => {
      S.dow_stats[dow] = { digit_freq: {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10} };
    });
  }
  
  // Ensure sum_freq exists
  if (!S.sum_freq) {
    S.sum_freq = {};
    for (let i = 0; i <= 27; i++) S.sum_freq[i] = 1;
  }
  
  // Ensure dtrans2 exists (2nd-order Markov)
  if (!S.dtrans2) {
    S.dtrans2 = [{}, {}, {}]; // 3 positions, each with empty state data
  }
  
  // Ensure ml_pos exists (ML positional weights)
  if (!S.ml_pos) {
    S.ml_pos = {
      '0': {'0':0.1,'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.1,'7':0.1,'8':0.1,'9':0.1},
      '1': {'0':0.1,'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.1,'7':0.1,'8':0.1,'9':0.1},
      '2': {'0':0.1,'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.1,'7':0.1,'8':0.1,'9':0.1}
    };
  }
  
  // Ensure monthly_stats exists
  if (!S.monthly_stats) {
    S.monthly_stats = {};
    for (let m = 1; m <= 12; m++) {
      S.monthly_stats[m] = { digit_freq: {'0':10,'1':10,'2':10,'3':10,'4':10,'5':10,'6':10,'7':10,'8':10,'9':10} };
    }
  }
  
  // Ensure meta exists
  if (!S.meta) S.meta = { total: 0 };
  
  return S;
}

/**
 * Build scoring context from current state.
 * Called once per prediction run, not per combo.
 */
function buildScoringContext(S, mergedTrans, transRowSums, mergedComboFreq, liveGaps, decayWeights, slot, dow, drawHistory) {
  // Validate S and add missing properties
  S = validateAndFixS(S);
  
  const lastResult = drawHistory.length > 0
    ? drawHistory[drawHistory.length - 1].result
    : '000';
  const prevResult = drawHistory.length > 1
    ? drawHistory[drawHistory.length - 2].result
    : lastResult;

  // Slot-specific positional frequencies
  const slotPosFreq = S.slot_stats[slot]?.pos_freq || S.pos_freq;

  // Day-of-week digit frequencies
  const dowFreq = S.dow_stats[dow]?.digit_freq || {};

  // Sum frequency (from S)
  const sumFreq = S.sum_freq || {};
  const maxSumFreq = Math.max(...Object.values(sumFreq), 1);

  // Max gap for normalization
  const maxGap = Math.max(...liveGaps.flat().map(v => v === drawHistory.length ? 0 : v), 1);

  // Max combo frequency for normalization
  const maxComboFreq = Math.max(...Object.values(mergedComboFreq), 1);

  // Per-digit overdue from combo frequency map
  const totalDraws = drawHistory.length + (S.meta?.total || 0);

  // Current month and day phase for seasonal scoring
  const now = new Date();
  const curMonth = now.getMonth() + 1;
  const curDay = now.getDate();
  const dayPhase = curDay <= 10 ? 0 : curDay <= 20 ? 1 : 2;

  // 2nd-order state string: prevDigit → lastDigit per position
  const ord2State = [0, 1, 2].map(p => prevResult[p] + lastResult[p]);

  return {
    lastResult, prevResult, ord2State,
    slotPosFreq, dowFreq, sumFreq, maxSumFreq,
    maxGap, maxComboFreq, totalDraws,
    curMonth, dayPhase,
    mergedTrans, transRowSums, mergedComboFreq, liveGaps, decayWeights,
    S,
  };
}

/**
 * Score a single combo against all 7 layers.
 * Returns normalized [0,1] scores per layer.
 */
function scoreCombo(num, ctx) {
  const [d0, d1, d2] = [+num[0], +num[1], +num[2]];
  const { lastResult, ord2State, slotPosFreq, dowFreq, sumFreq, maxSumFreq,
          maxGap, mergedTrans, transRowSums, mergedComboFreq, liveGaps,
          decayWeights, maxComboFreq, S } = ctx;

  // L1: Markov
  let mRaw = 1.0;
  for (let p = 0; p < 3; p++) {
    const from = +lastResult[p], to = +num[p];
    const rowSum = transRowSums[p][from] || 1;
    mRaw *= (mergedTrans[p][from][to] + 1) / (rowSum + 10);
  }

  // L1b: 2nd-order Markov  (dtrans2 values are percentages summing ≈100 per state)
  let o2Raw = 1.0;
  for (let p = 0; p < 3; p++) {
    const stateData = (S.dtrans2?.[p]?.[ord2State[p]]) || {};
    const digitStr = String(+num[p]);
    const val = parseFloat(stateData[digitStr]);
    o2Raw *= (isNaN(val) ? 10 : val) / 100;
  }

  // L2: ML positional frequency (from S.ml_pos)
  let mlRaw = 1.0;
  for (let p = 0; p < 3; p++) {
    mlRaw *= S.ml_pos?.[String(p)]?.[num[p]] || 0.001;
  }

  // L3: Slot-specific position bias (normalized to 10%)
  const sl0 = +(slotPosFreq[0]?.[String(d0)] || 10);
  const sl1 = +(slotPosFreq[1]?.[String(d1)] || 10);
  const sl2 = +(slotPosFreq[2]?.[String(d2)] || 10);
  const slRaw = (sl0 + sl1 + sl2) / 30; // normalize: each ~10%, so avg ~10%, /30 → ~0.33 range

  // L4a: Day-of-week bias
  const dw0 = +(dowFreq[String(d0)] || 10);
  const dw1 = +(dowFreq[String(d1)] || 10);
  const dw2 = +(dowFreq[String(d2)] || 10);
  const dwRaw = (dw0 + dw1 + dw2) / 30;

  // L4b: Live recency (exponential decay)
  const rcRaw = (decayWeights[d0] + decayWeights[d1] + decayWeights[d2]) / 3;

  // L5a: Sum distribution
  const sv = d0 + d1 + d2;
  const suRaw = (sumFreq[sv] || 1) / maxSumFreq;

  // L5b: Gap / overdue (log-scaled for stability)
  const g0 = liveGaps[0][d0], g1 = liveGaps[1][d1], g2 = liveGaps[2][d2];
  const safe = v => Math.min(v, maxGap);
  const gapRaw = (Math.log(safe(g0) + 1) + Math.log(safe(g1) + 1) + Math.log(safe(g2) + 1)) /
                 (3 * Math.log(maxGap + 1));

  // Combo-level overdue
  const comboFreq = mergedComboFreq[num] || 0;
  const ovRaw = comboFreq > 0
    ? Math.min(1.0, (comboFreq / maxComboFreq))
    : 0.5;

  const overdueRaw = 0.5 * gapRaw + 0.5 * (1 - ovRaw); // inverse: rare combos score higher

  // L6: Positional pair transition score
  const pairRaw = computePairScore(num, mergedTrans, transRowSums);

  // L7: Seasonal phase — how common is this sum in the current month phase?
  const monthData = S.monthly_stats?.[ctx.curMonth];
  const monthDigitFreq = monthData?.digit_freq || {};
  const seasonRaw = [d0, d1, d2].reduce((s, d) => s + +(monthDigitFreq[String(d)] || 10), 0) / 30;

  return { mRaw, o2Raw, mlRaw, slRaw, dwRaw, rcRaw, suRaw, overdueRaw, pairRaw, seasonRaw };
}

/**
 * Positional pair transition: P(d0→d1) × P(d1→d2) from position transitions.
 * Uses positions 0→1 and 1→2 transitions within same draw (co-occurrence).
 */
function computePairScore(num, trans, rowSums) {
  const [d0, d1, d2] = [+num[0], +num[1], +num[2]];
  // P(d1 | d0 appears in pos0 → pos1 transition data)
  // We use pos0→pos1 cross-position data from trans[1] conditioned on trans[0] last
  // Simplified: just use raw joint probability approximation
  const p01 = rowSums[0][d0] > 0 ? (trans[0][d0][d1] + 1) / (rowSums[0][d0] + 10) : 0.1;
  const p12 = rowSums[1][d1] > 0 ? (trans[1][d1][d2] + 1) / (rowSums[1][d1] + 10) : 0.1;
  return Math.sqrt(p01 * p12) * 10; // scale to ~0-1 range (normalized downstream)
}

/**
 * Main async scoring function — processes 1000 combos in chunks.
 * Yields control between chunks to keep UI responsive (BUG-E fix).
 *
 * @param {Object} S - Base statistical dataset
 * @param {Array} mergedTrans - Augmented transition matrices
 * @param {Array} transRowSums - Row sums for normalization
 * @param {Object} mergedComboFreq - Merged combo frequency map
 * @param {Array} liveGaps - Live digit gaps per position
 * @param {Array} decayWeights - Per-digit decay weights
 * @param {string} slot - Draw slot (2pm/5pm/9pm)
 * @param {string} dow - Day of week
 * @param {Array} drawHistory - All draws in chronological order
 * @param {Object} weights - Layer weights (will be normalized)
 * @param {Function} onProgress - Progress callback (0-100)
 * @returns {Promise<Array>} Sorted array of { num, score, layers... }
 */
async function computeScoresAsync(
  S, mergedTrans, transRowSums, mergedComboFreq,
  liveGaps, decayWeights, slot, dow, drawHistory,
  weights = DEFAULT_WEIGHTS, onProgress = null
) {
  const { weights: wn } = normalizeWeights(weights);
  const ctx = buildScoringContext(
    S, mergedTrans, transRowSums, mergedComboFreq,
    liveGaps, decayWeights, slot, dow, drawHistory
  );

  const rawScores = new Array(1000);

  // Process in chunks of 100 to yield control
  const CHUNK = 100;
  for (let start = 0; start < 1000; start += CHUNK) {
    const end = Math.min(start + CHUNK, 1000);
    for (let n = start; n < end; n++) {
      rawScores[n] = { n, ...scoreCombo(n.toString().padStart(3, '0'), ctx) };
    }
    if (onProgress) onProgress(Math.round(end / 10));
    // Yield to event loop between chunks
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  // Normalize each layer across all 1000 combos
  const layerKeys = ['mRaw', 'o2Raw', 'mlRaw', 'slRaw', 'dwRaw', 'rcRaw', 'suRaw', 'overdueRaw', 'pairRaw', 'seasonRaw'];
  const maxVals = {};
  const minVals = {};
  layerKeys.forEach(k => {
    const vals = rawScores.map(s => s[k]);
    maxVals[k] = Math.max(...vals) || 1;
    minVals[k] = Math.min(...vals);
  });

  const layerToWeight = {
    mRaw: wn.markov, o2Raw: wn.ord2, mlRaw: wn.ml,
    slRaw: wn.slot, dwRaw: wn.dow, rcRaw: wn.rec,
    suRaw: wn.sum, overdueRaw: wn.overdue, pairRaw: wn.pair, seasonRaw: wn.season,
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
        markov:   Math.round(normalized.mRaw * 100),
        ord2:     Math.round(normalized.o2Raw * 100),
        ml:       Math.round(normalized.mlRaw * 100),
        slot:     Math.round(normalized.slRaw * 100),
        dow:      Math.round(normalized.dwRaw * 100),
        rec:      Math.round(normalized.rcRaw * 100),
        sum:      Math.round(normalized.suRaw * 100),
        overdue:  Math.round(normalized.overdueRaw * 100),
        pair:     Math.round(normalized.pairRaw * 100),
        season:   Math.round(normalized.seasonRaw * 100),
      },
    };
  });

  if (onProgress) onProgress(100);
  return results.sort((a, b) => b.score - a.score);
}

export { computeScoresAsync, normalizeWeights, DEFAULT_WEIGHTS };
