/**
 * dataStore.js — Merges base statistical dataset (S) with user-added draws
 *
 * KEY DESIGN DECISION:
 * The base statistical data (S) is embedded as precomputed aggregates
 * (frequency tables, Markov matrices, etc.) — NOT raw historical draws.
 * We never generate synthetic draws. User-added draws extend the live dataset.
 *
 * This eliminates BUG-A (synthetic data pollution) entirely.
 */

/**
 * Build the combined draw list: base recent draws + user draws.
 * Base draws (S.recent_500) are treated as verified seed data.
 * User draws are appended and deduplicated.
 *
 * @param {Object} S - The embedded statistical dataset
 * @param {Array}  userDraws - User-persisted draws from localStorage
 * @returns {{ draws: Array, userStartIdx: number }}
 */
function buildDrawList(S, userDraws) {
  const baseDraws = (S.recent_500 || []).map(d => ({ ...d, _source: 'base' }));

  // Deduplicate user draws against base
  const baseKeys = new Set(baseDraws.map(drawKey));
  const newUserDraws = userDraws
    .map(d => ({ ...d, _source: 'user' }))
    .filter(d => !baseKeys.has(drawKey(d)));

  const draws = [...baseDraws, ...newUserDraws];
  return { draws, userStartIdx: baseDraws.length };
}

/** Stable key for deduplication */
function drawKey(d) {
  return `${d.year}-${d.month}-${d.day}-${d.draw_time}`;
}

/**
 * Compute live digit gaps from a draw list.
 * Returns gaps[position][digit] = draws since last appearance (0 = last draw)
 */
function computeLiveGaps(draws) {
  // NOTE: Must return Array-of-Arrays (not Array-of-Objects) so that
  // liveGaps.flat() in predictionEngine yields numeric values, not Objects.
  const gaps = Array.from({ length: 3 }, () => new Array(10).fill(null));

  for (let i = draws.length - 1; i >= 0; i--) {
    const r = draws[i].result;
    let allFilled = true;
    for (let pos = 0; pos < 3; pos++) {
      const digit = +r[pos];
      if (gaps[pos][digit] === null) {
        gaps[pos][digit] = draws.length - 1 - i;
      }
      if (gaps[pos][digit] === null) allFilled = false;
    }
    if (allFilled) break;
  }

  // Replace nulls with sentinel (never seen in available draws)
  for (let pos = 0; pos < 3; pos++) {
    for (let d = 0; d < 10; d++) {
      if (gaps[pos][d] === null) gaps[pos][d] = draws.length;
    }
  }

  return gaps;
}

/**
 * Compute digit frequency in last N draws (live, reflects user additions).
 * Returns normalized percentages [0–100] indexed by digit.
 */
function computeRecentFreq(draws, windowSize = 90) {
  const slice = draws.slice(-windowSize);
  const counts = new Array(10).fill(0);
  slice.forEach(r => r.result.split('').forEach(d => counts[+d]++));
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  return counts.map(c => +(c / total * 100).toFixed(2));
}

/**
 * Compute exponential-decay recency weights.
 * Returns per-digit weight [0-1], normalized so sum = 1.
 */
function computeDecayWeights(draws, windowSize = 30, decayHalfLife = 10) {
  const slice = draws.slice(-windowSize);
  const weights = new Array(10).fill(0);

  slice.forEach((r, idx) => {
    const age = slice.length - 1 - idx;         // 0 = most recent
    const w = Math.pow(0.5, age / decayHalfLife); // half-life decay
    r.result.split('').forEach(d => { weights[+d] += w; });
  });

  const total = weights.reduce((a, b) => a + b, 0) || 1;
  return weights.map(w => w / total);
}

/**
 * Build live combo frequency map from user draws only.
 * Base combo frequencies are pre-computed in S.base_freq.
 */
function buildLiveComboFreq(draws) {
  const freq = {};
  draws.forEach(d => {
    freq[d.result] = (freq[d.result] || 0) + 1;
  });
  return freq;
}

/**
 * Merge base combo frequency with live user draw frequencies.
 * base_freq values are draw counts from S.meta.total draws.
 */
function mergeComboFreq(baseFreq, liveFreq) {
  const merged = { ...baseFreq };
  Object.entries(liveFreq).forEach(([combo, count]) => {
    merged[combo] = (merged[combo] || 0) + count;
  });
  return merged;
}

/**
 * Build augmented 1st-order Markov transition matrices from user draws.
 * Returns trans[pos][fromDigit][toDigit] = count (integers only).
 * Base matrices (S.trans_full) are already pre-computed.
 */
function buildLiveTransitions(draws) {
  const trans = Array.from({ length: 3 }, () =>
    Array.from({ length: 10 }, () => new Array(10).fill(0))
  );

  for (let i = 1; i < draws.length; i++) {
    const prev = draws[i - 1].result;
    const curr = draws[i].result;
    for (let pos = 0; pos < 3; pos++) {
      trans[pos][+prev[pos]][+curr[pos]]++;
    }
  }

  return trans;
}

/**
 * Merge pre-computed base transition matrices with live augmentations.
 * baseTrans = S.trans_full (arrays of counts)
 * liveTrans = output of buildLiveTransitions
 */
function mergeTransitions(baseTrans, liveTrans) {
  return baseTrans.map((posMatrix, pos) =>
    posMatrix.map((fromRow, from) =>
      fromRow.map((count, to) => count + liveTrans[pos][from][to])
    )
  );
}

export {
  buildDrawList,
  drawKey,
  computeLiveGaps,
  computeRecentFreq,
  computeDecayWeights,
  buildLiveComboFreq,
  mergeComboFreq,
  buildLiveTransitions,
  mergeTransitions,
};
