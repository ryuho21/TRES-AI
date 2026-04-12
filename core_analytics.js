/**
 * analytics.js — Non-predictive pattern exploration
 *
 * All functions are PURE: they describe what happened, not what will happen.
 * Every output includes a disclaimer flag.
 */

import { chiSquareTest, shannonEntropy } from './core_statistics.js';

const DISCLAIMER = 'Describes historical patterns only. No predictive power confirmed by backtesting.';

/**
 * Chi-square bias detection across digits, positions, and time windows.
 *
 * @param {Array} draws - Full draw list
 * @returns {ChiSquareReport}
 */
function analyzeDigitBias(draws) {
  if (!draws || draws.length === 0) {
    return { error: 'No draws available', disclaimer: DISCLAIMER };
  }

  // Overall digit distribution
  const overallCounts = new Array(10).fill(0);
  draws.forEach(d => d.result.split('').forEach(x => overallCounts[+x]++));
  const overallTest = chiSquareTest(overallCounts);

  // Per-position distribution
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
    const test = chiSquareTest(counts);
    windowTests.push({
      windowEnd: i,
      result: window[window.length - 1]?.result || '',
      ...test,
    });
  }

  return {
    overall: { counts: overallCounts, ...overallTest },
    byPosition: positionTests,
    rollingWindows: windowTests,
    disclaimer: DISCLAIMER,
  };
}

/**
 * Shannon entropy analysis.
 * Measures how "random" the draw distribution is.
 */
function analyzeEntropy(draws) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  // Overall digit entropy
  const counts = new Array(10).fill(0);
  draws.forEach(d => d.result.split('').forEach(x => counts[+x]++));
  const digitEntropy = shannonEntropy(counts);

  // Per-position
  const positionEntropy = [0, 1, 2].map(pos => {
    const c = new Array(10).fill(0);
    draws.forEach(d => c[+d.result[pos]]++);
    return { position: pos + 1, ...shannonEntropy(c) };
  });

  // Combo-level entropy (over 1000 possible combos)
  const comboCounts = {};
  draws.forEach(d => { comboCounts[d.result] = (comboCounts[d.result] || 0) + 1; });
  // Pad unseen combos with 0
  for (let n = 0; n < 1000; n++) {
    const num = n.toString().padStart(3, '0');
    if (!comboCounts[num]) comboCounts[num] = 0;
  }
  const comboCountsArr = Object.values(comboCounts);
  const comboEntropy = shannonEntropy(comboCountsArr);

  // Interpretation
  const digitNorm = digitEntropy.normalizedEntropy;
  const interpretation = digitNorm > 0.99
    ? 'Near-maximum entropy: digits are distributed very uniformly.'
    : digitNorm > 0.97
    ? 'Slight structure in digit distribution (very small effect).'
    : 'Notable deviation from maximum entropy — potential bias.';

  return {
    digitEntropy,
    positionEntropy,
    comboEntropy,
    interpretation,
    disclaimer: DISCLAIMER,
  };
}

/**
 * Gap analysis: how many draws since each digit/combo last appeared.
 *
 * @param {Array} draws - Chronological draw list
 * @returns {GapReport}
 */
function analyzeGaps(draws) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  // Per-position digit gaps
  const posGaps = Array.from({ length: 3 }, () => new Array(10).fill(null));
  for (let i = draws.length - 1; i >= 0; i--) {
    const r = draws[i].result;
    for (let pos = 0; pos < 3; pos++) {
      const d = +r[pos];
      if (posGaps[pos][d] === null) posGaps[pos][d] = draws.length - 1 - i;
    }
    if (posGaps.every(pg => pg.every(v => v !== null))) break;
  }

  // Replace nulls with "never seen" sentinel
  posGaps.forEach((pg, pos) => pg.forEach((v, d) => {
    if (v === null) posGaps[pos][d] = draws.length;
  }));

  // Combo gaps (last N draws)
  const comboLastSeen = {};
  draws.forEach((d, idx) => { comboLastSeen[d.result] = idx; });
  const comboGaps = {};
  Object.entries(comboLastSeen).forEach(([combo, lastIdx]) => {
    comboGaps[combo] = draws.length - 1 - lastIdx;
  });

  // Never-seen combos
  const neverSeen = [];
  for (let n = 0; n < 1000; n++) {
    const num = n.toString().padStart(3, '0');
    if (!comboLastSeen.hasOwnProperty(num)) neverSeen.push(num);
  }

  // Expected gap under uniform distribution
  const uniqueCombos = Object.keys(comboLastSeen).length;
  const expectedGap = draws.length / Math.max(uniqueCombos, 1);

  // Most overdue (relative to expected frequency)
  const overdueRanked = Object.entries(comboGaps)
    .map(([combo, gap]) => {
      const freq = draws.filter(d => d.result === combo).length;
      const expGap = freq > 0 ? draws.length / freq : draws.length;
      return { combo, gap, expectedGap: +expGap.toFixed(1), overdueFactor: +(gap / expGap).toFixed(2) };
    })
    .sort((a, b) => b.overdueFactor - a.overdueFactor)
    .slice(0, 20);

  return {
    positionGaps: posGaps,
    overdueRanked,
    neverSeen: { count: neverSeen.length, combos: neverSeen.slice(0, 20) },
    expectedGapUniform: +expectedGap.toFixed(1),
    disclaimer: DISCLAIMER + ' "Overdue" does NOT mean "due soon" — this is gambler\'s fallacy.',
  };
}

/**
 * Streak analysis: hot/cold numbers, run-length distributions.
 */
function analyzeStreaks(draws, windowSize = 30) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  const slice = draws.slice(-windowSize);
  const counts = new Array(10).fill(0);
  slice.forEach(d => d.result.split('').forEach(x => counts[+x]++));

  const expected = (slice.length * 3) / 10;
  const digitsRanked = counts.map((c, d) => ({
    digit: d,
    count: c,
    expected: +expected.toFixed(1),
    zScore: +(((c - expected) / Math.sqrt(expected)) || 0).toFixed(2),
  })).sort((a, b) => b.count - a.count);

  const hot  = digitsRanked.slice(0, 3);
  const cold = digitsRanked.slice(-3).reverse();

  // Run-length: consecutive appearances in same position
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

  // Combo repetition in last 100 draws
  const recentCombos = draws.slice(-100);
  const comboRuns = {};
  recentCombos.forEach(d => { comboRuns[d.result] = (comboRuns[d.result] || 0) + 1; });
  const repeaters = Object.entries(comboRuns)
    .filter(([, c]) => c > 1)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  return {
    window: windowSize,
    hot, cold,
    digitsRanked,
    runLengths,
    repeaters,
    disclaimer: DISCLAIMER,
  };
}

/**
 * Combination coverage: how many of 1000 possible combos have appeared.
 */
function analyzeCoverage(draws) {
  if (!draws || draws.length === 0) return { error: 'No draws', disclaimer: DISCLAIMER };

  const seen = new Set(draws.map(d => d.result));
  const seenCount = seen.size;
  const total = 1000;
  const coverage = seenCount / total;

  // Coverage by pattern type
  const byPattern = { triple: 0, double: 0, straight: 0 };
  seen.forEach(c => {
    const [a, b, cc] = [+c[0], +c[1], +c[2]];
    if (a === b && b === cc) byPattern.triple++;
    else if (a === b || b === cc || a === cc) byPattern.double++;
    else byPattern.straight++;
  });

  // Expected combos under uniform distribution
  // Coupon collector problem: E[draws to see k combos] ≈ 1000 * H(1000) / H(1000-k)
  const expectedToSeeAll = 1000 * Array.from({ length: 1000 }, (_, i) => 1 / (1000 - i)).reduce((a, b) => a + b, 0);

  return {
    seenCount,
    unseenCount: total - seenCount,
    coverage: +(coverage * 100).toFixed(2),
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
      triple: +(10 / 1000 * 100).toFixed(2),
      double: +(270 / 1000 * 100).toFixed(2),
      straight: +(720 / 1000 * 100).toFixed(2),
    },
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
  DISCLAIMER,
};
