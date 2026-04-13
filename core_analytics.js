/**
 * core_analytics.js — Non-predictive pattern exploration v5.2
 *
 * NEW (v5.2):
 * - analyzeMarkov(): 1st-order transition heatmaps + notable transitions per position
 * - analyzeMarkov2ndOrder(): 2nd-order Markov analysis using live dtrans2
 * - All functions now accept optional liveDtrans2 parameter
 *
 * PRESERVED:
 * - All functions are PURE: describe history, not the future
 * - Every output includes a disclaimer flag
 * - BUG-D: correct chi-square and entropy via core_statistics
 */

import { chiSquareTest, shannonEntropy } from './core_statistics.js';

const DISCLAIMER = 'Describes historical patterns only. No predictive power confirmed by backtesting.';

/**
 * Chi-square bias detection across digits, positions, and time windows.
 */
function analyzeDigitBias(draws) {
  if (!draws || draws.length === 0) {
    return { error: 'No draws available', disclaimer: DISCLAIMER };
  }

  const overallCounts = new Array(10).fill(0);
  draws.forEach(d => d.result.split('').forEach(x => overallCounts[+x]++));
  const overallTest = chiSquareTest(overallCounts);

  const positionTests = [0, 1, 2].map(pos => {
    const counts = new Array(10).fill(0);
    draws.forEach(d => counts[+d.result[pos]]++);
    return { position: pos + 1, counts, ...chiSquareTest(counts) };
  });

  // Rolling 30-draw windows
  const windowSize = 30;
  const windowTests = [];
  for (let i = windowSize; i <= draws.length; i += Math.max(1, Math.floor(windowSize / 2))) {
    const window = draws.slice(i - windowSize, i);
    const counts = new Array(10).fill(0);
    window.forEach(d => d.result.split('').forEach(x => counts[+x]++));
    windowTests.push({
      windowEnd: i,
      result: window[window.length - 1]?.result || '',
      ...chiSquareTest(counts),
    });
  }

  return {
    overall:       { counts: overallCounts, ...overallTest },
    byPosition:    positionTests,
    rollingWindows: windowTests,
    disclaimer:    DISCLAIMER,
  };
}

/**
 * Shannon entropy analysis.
 */
function analyzeEntropy(draws) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  const counts = new Array(10).fill(0);
  draws.forEach(d => d.result.split('').forEach(x => counts[+x]++));
  const digitEntropy = shannonEntropy(counts);

  const positionEntropy = [0, 1, 2].map(pos => {
    const c = new Array(10).fill(0);
    draws.forEach(d => c[+d.result[pos]]++);
    return { position: pos + 1, ...shannonEntropy(c) };
  });

  const comboCounts = {};
  draws.forEach(d => { comboCounts[d.result] = (comboCounts[d.result] || 0) + 1; });
  for (let n = 0; n < 1000; n++) {
    const num = n.toString().padStart(3, '0');
    if (!comboCounts[num]) comboCounts[num] = 0;
  }
  const comboEntropy = shannonEntropy(Object.values(comboCounts));

  const digitNorm = digitEntropy.normalizedEntropy;
  const interpretation = digitNorm > 0.99
    ? 'Near-maximum entropy: digits are distributed very uniformly.'
    : digitNorm > 0.97
    ? 'Slight structure in digit distribution (very small effect).'
    : 'Notable deviation from maximum entropy — potential bias.';

  return { digitEntropy, positionEntropy, comboEntropy, interpretation, disclaimer: DISCLAIMER };
}

/**
 * Gap analysis: draws since each digit/combo last appeared.
 */
function analyzeGaps(draws) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  const posGaps = Array.from({ length: 3 }, () => new Array(10).fill(null));
  for (let i = draws.length - 1; i >= 0; i--) {
    const r = draws[i].result;
    for (let pos = 0; pos < 3; pos++) {
      const d = +r[pos];
      if (posGaps[pos][d] === null) posGaps[pos][d] = draws.length - 1 - i;
    }
    if (posGaps.every(pg => pg.every(v => v !== null))) break;
  }
  posGaps.forEach((pg, pos) => pg.forEach((v, d) => {
    if (v === null) posGaps[pos][d] = draws.length;
  }));

  const comboLastSeen = {};
  draws.forEach((d, idx) => { comboLastSeen[d.result] = idx; });
  const comboGaps = {};
  Object.entries(comboLastSeen).forEach(([combo, lastIdx]) => {
    comboGaps[combo] = draws.length - 1 - lastIdx;
  });

  const neverSeen = [];
  for (let n = 0; n < 1000; n++) {
    const num = n.toString().padStart(3, '0');
    if (!Object.prototype.hasOwnProperty.call(comboLastSeen, num)) neverSeen.push(num);
  }

  const overdueRanked = Object.entries(comboGaps)
    .map(([combo, gap]) => {
      const freq   = draws.filter(d => d.result === combo).length;
      const expGap = freq > 0 ? draws.length / freq : draws.length;
      return { combo, gap, expectedGap: +expGap.toFixed(1), overdueFactor: +(gap / expGap).toFixed(2) };
    })
    .sort((a, b) => b.overdueFactor - a.overdueFactor)
    .slice(0, 20);

  return {
    positionGaps: posGaps,
    overdueRanked,
    neverSeen: { count: neverSeen.length, combos: neverSeen.slice(0, 20) },
    expectedGapUniform: +(draws.length / Math.max(Object.keys(comboLastSeen).length, 1)).toFixed(1),
    disclaimer: DISCLAIMER + " 'Overdue' does NOT mean 'due soon' — this is gambler's fallacy.",
  };
}

/**
 * Streak analysis: hot/cold numbers, run-length distributions.
 */
function analyzeStreaks(draws, windowSize = 30) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  const slice    = draws.slice(-windowSize);
  const counts   = new Array(10).fill(0);
  slice.forEach(d => d.result.split('').forEach(x => counts[+x]++));

  const expected      = (slice.length * 3) / 10;
  const digitsRanked  = counts.map((c, d) => ({
    digit:    d,
    count:    c,
    expected: +expected.toFixed(1),
    zScore:   +(((c - expected) / Math.sqrt(expected)) || 0).toFixed(2),
  })).sort((a, b) => b.count - a.count);

  const hot  = digitsRanked.slice(0, 3);
  const cold = digitsRanked.slice(-3).reverse();

  const runLengths = [[], [], []];
  [0, 1, 2].forEach(pos => {
    let run = 1;
    for (let i = 1; i < draws.length; i++) {
      if (draws[i].result[pos] === draws[i - 1].result[pos]) {
        run++;
      } else {
        if (run > 1) runLengths[pos].push(run);
        run = 1;
      }
    }
  });

  const recentCombos = draws.slice(-100);
  const comboRuns    = {};
  recentCombos.forEach(d => { comboRuns[d.result] = (comboRuns[d.result] || 0) + 1; });
  const repeaters = Object.entries(comboRuns)
    .filter(([, c]) => c > 1)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  return { window: windowSize, hot, cold, digitsRanked, runLengths, repeaters, disclaimer: DISCLAIMER };
}

/**
 * Combination coverage analysis.
 */
function analyzeCoverage(draws) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  const seen      = new Set(draws.map(d => d.result));
  const seenCount = seen.size;

  const byPattern = { triple: 0, double: 0, straight: 0 };
  seen.forEach(c => {
    const [a, b, cc] = [+c[0], +c[1], +c[2]];
    if (a === b && b === cc) byPattern.triple++;
    else if (a === b || b === cc || a === cc) byPattern.double++;
    else byPattern.straight++;
  });

  const expectedToSeeAll = 1000 *
    Array.from({ length: 1000 }, (_, i) => 1 / (1000 - i)).reduce((a, b) => a + b, 0);

  return {
    seenCount,
    unseenCount: 1000 - seenCount,
    coverage: +(seenCount / 10).toFixed(2),
    byPattern,
    expectedDrawsToSeeAll: Math.round(expectedToSeeAll),
    disclaimer: DISCLAIMER,
  };
}

/**
 * Structural pattern distribution (triples, doubles, straights).
 */
function analyzeStructure(draws) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  const counts = { triple: 0, double: 0, palindrome: 0, straight: 0 };
  draws.forEach(d => {
    const [a, b, c] = [+d.result[0], +d.result[1], +d.result[2]];
    if (a === b && b === c) counts.triple++;
    else if (a === b || b === c || a === c) counts.double++;
    else counts.straight++;
    if (a === c && a !== b) counts.palindrome++;
  });

  const n = draws.length;
  return {
    counts,
    percentages: Object.fromEntries(
      Object.entries(counts).map(([k, v]) => [k, +(v / n * 100).toFixed(2)])
    ),
    expected: {
      triple:   +(10  / 1000 * 100).toFixed(2),
      double:   +(270 / 1000 * 100).toFixed(2),
      straight: +(720 / 1000 * 100).toFixed(2),
    },
    disclaimer: DISCLAIMER,
  };
}

/**
 * NEW: 1st-order Markov transition analysis.
 * Builds per-position 10×10 transition probability matrices and identifies
 * notably over/under-represented transitions (|deviation from 10%| > threshold).
 *
 * @param {Array}  mergedTrans    - 3×10×10 merged transition count matrix
 * @param {Array}  transRowSums   - Pre-computed row sums for normalization
 * @param {number} deviationThreshold - Min % deviation to flag as notable (default 3)
 * @param {number} minCount       - Min raw count to include in notable list (default 5)
 * @returns {MarkovReport}
 */
function analyzeMarkov(mergedTrans, transRowSums, deviationThreshold = 3, minCount = 5) {
  if (!mergedTrans || !transRowSums) {
    return { error: 'Transition matrices not available', disclaimer: DISCLAIMER };
  }

  const positions = [0, 1, 2].map(pos => {
    // Build 10×10 probability matrix for this position
    const matrix = Array.from({ length: 10 }, (_, from) => {
      const rowSum = transRowSums[pos][from] || 1;
      return Array.from({ length: 10 }, (_, to) => {
        const pct = (mergedTrans[pos][from][to] / rowSum) * 100;
        return +pct.toFixed(2);
      });
    });

    // Notable transitions: |deviation from 10% uniform| > threshold
    const notable = [];
    for (let from = 0; from < 10; from++) {
      const rowSum = transRowSums[pos][from] || 1;
      for (let to = 0; to < 10; to++) {
        const count = mergedTrans[pos][from][to];
        const pct   = (count / rowSum) * 100;
        const dev   = pct - 10;
        if (Math.abs(dev) > deviationThreshold && count >= minCount) {
          notable.push({ from, to, pct: +pct.toFixed(2), count, deviation: +dev.toFixed(2) });
        }
      }
    }
    notable.sort((a, b) => Math.abs(b.deviation) - Math.abs(a.deviation));

    // Entropy of each row — uniformly random = log2(10) ≈ 3.32 bits
    const rowEntropies = Array.from({ length: 10 }, (_, from) => {
      const rowSum = transRowSums[pos][from] || 1;
      const probs  = Array.from({ length: 10 }, (_, to) =>
        mergedTrans[pos][from][to] / rowSum
      );
      let h = 0;
      probs.forEach(p => { if (p > 0) h -= p * Math.log2(p); });
      return { from, entropy: +h.toFixed(3), normalized: +(h / Math.log2(10)).toFixed(3), rowSum };
    });

    return { position: pos + 1, matrix, notable: notable.slice(0, 10), rowEntropies };
  });

  return {
    positions,
    totalTransitions: mergedTrans[0].flat().reduce((a, b) => a + b, 0),
    disclaimer: DISCLAIMER,
  };
}

/**
 * NEW: 2nd-order Markov analysis using live-computed dtrans2.
 * Shows which (prev,curr) → next digit transitions are strongest per position.
 *
 * @param {Array}  liveDtrans2 - Output of buildLiveDtrans2()
 * @param {number} deviationThreshold - Min % deviation from 10% to flag as notable
 * @returns {Dtrans2Report}
 */
function analyzeMarkov2ndOrder(liveDtrans2, deviationThreshold = 5) {
  if (!liveDtrans2) {
    return { error: '2nd-order transitions not available', disclaimer: DISCLAIMER };
  }

  const stateCount = liveDtrans2.reduce((total, posMap) => total + Object.keys(posMap).length, 0);
  const totalPossible = 3 * 100; // 3 positions × 100 two-digit states

  const positions = [0, 1, 2].map(pos => {
    const posMap  = liveDtrans2[pos] || {};
    const states  = Object.keys(posMap);
    const notable = [];

    states.forEach(state => {
      const dist = posMap[state];
      Object.entries(dist).forEach(([digit, pct]) => {
        const dev = +pct - 10;
        if (Math.abs(dev) > deviationThreshold) {
          notable.push({ state, digit: +digit, pct: +pct, deviation: +dev.toFixed(2) });
        }
      });
    });
    notable.sort((a, b) => Math.abs(b.deviation) - Math.abs(a.deviation));

    // Which states have the highest max deviation (strongest conditional signal)?
    const stateSignal = states.map(state => {
      const dist = posMap[state];
      const maxDev = Math.max(...Object.values(dist).map(p => Math.abs(+p - 10)));
      return { state, maxDeviation: +maxDev.toFixed(2) };
    }).sort((a, b) => b.maxDeviation - a.maxDeviation);

    return { position: pos + 1, statesCovered: states.length, notable: notable.slice(0, 8), stateSignal: stateSignal.slice(0, 5) };
  });

  return {
    positions,
    stateCoverage: { covered: stateCount, total: totalPossible, pct: +(stateCount / totalPossible * 100).toFixed(1) },
    disclaimer: DISCLAIMER,
  };
}

export {
  analyzeDigitBias,
  analyzeEntropy,
  analyzeGaps,
  analyzeStreaks,
  analyzeCoverage,
  analyzeStructure,
  analyzeMarkov,
  analyzeMarkov2ndOrder,
  DISCLAIMER,
};
