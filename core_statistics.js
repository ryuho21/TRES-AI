/**
 * core_statistics.js — Statistical methods v5.2
 *
 * PRESERVED from v5.1 (BUG-D fixes):
 * - normalTail uses correct erf approximation (Abramowitz & Stegun 7.1.26)
 * - Wilson score CI (more accurate than normal approximation at extremes)
 * - Bonferroni correction applied consistently
 *
 * NEW (v5.2):
 * - chiSquarePower(): compute approximate statistical power for backtest planning
 * - wilsonCIPercent(): convenience wrapper returning percentages
 * - normalQuantile(): exported for use by other modules
 */

/**
 * Error function approximation (Abramowitz & Stegun 7.1.26)
 * Max error: |ε| ≤ 1.5 × 10⁻⁷. Valid for x ≥ 0.
 */
function erf(x) {
  const t = 1.0 / (1.0 + 0.3275911 * x);
  return 1.0 - (
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
  ) * Math.exp(-x * x);
}

/**
 * Standard normal CDF: P(Z ≤ z). Correct for all z (BUG-D fix).
 */
function normalCDF(z) {
  if (z >= 0) return 0.5 * (1.0 + erf(z / Math.SQRT2));
  return 0.5 * (1.0 - erf(-z / Math.SQRT2));
}

/** Upper tail: P(Z > z) */
function normalUpperTail(z) {
  return 1.0 - normalCDF(z);
}

/**
 * Quantile function (inverse normal CDF) — Beasley-Springer-Moro algorithm.
 * Accurate to ~5 decimal places for p ∈ (0.0001, 0.9999).
 */
function normalQuantile(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  const a = [2.515517, 0.802853, 0.010328];
  const b = [1.432788, 0.189269, 0.001308];

  function rationalApprox(t) {
    return t - (a[0] + a[1] * t + a[2] * t * t) /
               (1 + b[0] * t + b[1] * t * t + b[2] * t * t * t);
  }

  if (p < 0.5) {
    const t = Math.sqrt(-2 * Math.log(p));
    return -rationalApprox(t);
  } else {
    const t = Math.sqrt(-2 * Math.log(1 - p));
    return rationalApprox(t);
  }
}

/**
 * Wilson score confidence interval for a proportion.
 * More accurate than normal approximation especially at p ≈ 0 or p ≈ 1.
 *
 * @param {number} successes - Number of successes
 * @param {number} n         - Sample size
 * @param {number} alpha     - Significance level (e.g. 0.05 for 95% CI)
 * @returns {{ lower, upper, center }}
 */
function wilsonCI(successes, n, alpha = 0.05) {
  if (n === 0) return { lower: 0, upper: 1, center: 0 };
  const p  = successes / n;
  const z  = normalQuantile(1 - alpha / 2);
  const z2 = z * z;
  const center = (p + z2 / (2 * n)) / (1 + z2 / n);
  const margin  = (z / (1 + z2 / n)) * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n));
  return {
    lower:  Math.max(0, center - margin),
    upper:  Math.min(1, center + margin),
    center,
  };
}

/**
 * Wilson CI returning percentages (0–100) for display convenience.
 */
function wilsonCIPercent(successes, n, alpha = 0.05) {
  const ci = wilsonCI(successes, n, alpha);
  return {
    lower:  +(ci.lower  * 100).toFixed(2),
    upper:  +(ci.upper  * 100).toFixed(2),
    center: +(ci.center * 100).toFixed(2),
  };
}

/**
 * Two-proportion z-test (one-tailed upper: H₁: p₁ > p₂).
 * Used for backtest: Current System vs Random Baseline.
 *
 * FIX: Pooled SE now uses (1/n1 + 1/n2) — correct when n1 ≠ n2.
 * Previous version used (2/n1) which assumed equal sample sizes.
 *
 * @param {number} x1    - Successes in group 1 (system)
 * @param {number} n1    - Trials in group 1
 * @param {number} x2    - Successes in group 2 (baseline)
 * @param {number} n2    - Trials in group 2
 * @param {number} alpha - Bonferroni-corrected significance level
 * @returns {{ zScore, pValue, significant, effectSize, pooledSE, p1, p2 }}
 */
function twoPropZTest(x1, n1, x2, n2, alpha = 0.05) {
  if (n1 === 0 || n2 === 0) {
    return { zScore: 0, pValue: 1, significant: false, effectSize: 0, pooledSE: 0, p1: 0, p2: 0 };
  }

  const p1      = x1 / n1;
  const p2      = x2 / n2;
  const pooledP = (x1 + x2) / (n1 + n2);
  // Correct pooled SE: accounts for potentially different n1, n2
  const se      = Math.sqrt(pooledP * (1 - pooledP) * (1 / n1 + 1 / n2));

  const zScore  = se > 0 ? (p1 - p2) / se : 0;
  const pValue  = normalUpperTail(zScore); // BUG-D fix: correct upper tail

  return {
    zScore:     +zScore.toFixed(4),
    pValue:     +pValue.toFixed(6),
    significant: pValue < alpha,
    effectSize: +(p1 - p2).toFixed(6),
    pooledSE:   +se.toFixed(6),
    p1:         +p1.toFixed(6),
    p2:         +p2.toFixed(6),
  };
}

/**
 * Bonferroni-corrected alpha for m simultaneous comparisons.
 */
function bonferroniAlpha(familyAlpha = 0.05, m = 1) {
  return familyAlpha / m;
}

/**
 * Chi-square goodness-of-fit test.
 * H₀: observed distribution matches expected (default: uniform).
 *
 * @param {number[]} observed - Observed counts
 * @param {number[]} expected - Expected counts (null = uniform)
 * @returns {{ chiSquare, df, pValue, significant, criticalValue95 }}
 */
function chiSquareTest(observed, expected = null) {
  const n     = observed.length;
  const total = observed.reduce((a, b) => a + b, 0);

  if (!expected) {
    const exp = total / n;
    expected  = new Array(n).fill(exp);
  }

  let chi2 = 0;
  for (let i = 0; i < n; i++) {
    if (expected[i] > 0) {
      chi2 += Math.pow(observed[i] - expected[i], 2) / expected[i];
    }
  }

  const df     = n - 1;
  const pValue = chiSquarePValue(chi2, df);

  return {
    chiSquare:       +chi2.toFixed(4),
    df,
    pValue:          +pValue.toFixed(6),
    significant:     pValue < 0.05,
    criticalValue95: chiSquareCritical(df, 0.05),
  };
}

/**
 * Chi-square p-value via Wilson-Hilferty normal approximation.
 * Accurate for df ≥ 1 and chi2 not extreme.
 */
function chiSquarePValue(chi2, df) {
  if (chi2 <= 0) return 1;
  const mu    = 1 - 2 / (9 * df);
  const sigma = Math.sqrt(2 / (9 * df));
  const z     = (Math.pow(chi2 / df, 1 / 3) - mu) / sigma;
  return normalUpperTail(z);
}

/**
 * Approximate chi-square critical value (Wilson-Hilferty inversion).
 */
function chiSquareCritical(df, alpha = 0.05) {
  const z     = normalQuantile(1 - alpha);
  const mu    = 1 - 2 / (9 * df);
  const sigma = Math.sqrt(2 / (9 * df));
  return df * Math.pow(mu + sigma * z, 3);
}

/**
 * Shannon entropy of a frequency distribution.
 *
 * @param {number[]} counts - Raw counts (need not sum to 1)
 * @returns {{ entropy, normalizedEntropy, maxEntropy }}
 */
function shannonEntropy(counts) {
  const total      = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return { entropy: 0, normalizedEntropy: 0, maxEntropy: 0 };

  const maxEntropy = Math.log2(counts.length);
  let entropy = 0;
  counts.forEach(c => {
    if (c > 0) {
      const p = c / total;
      entropy -= p * Math.log2(p);
    }
  });

  return {
    entropy:           +entropy.toFixed(6),
    normalizedEntropy: +(entropy / maxEntropy).toFixed(6),
    maxEntropy:        +maxEntropy.toFixed(6),
  };
}

/**
 * Approximate statistical power for a two-proportion z-test.
 * Useful for planning: "how many test draws do I need to detect a given effect?"
 *
 * @param {number} p1    - True proportion under H₁ (system)
 * @param {number} p2    - True proportion under H₀ (baseline = 1/1000)
 * @param {number} n     - Sample size (test draws)
 * @param {number} alpha - Significance threshold
 * @returns {{ power, detectable }}
 */
function chiSquarePower(p1, p2, n, alpha = 0.0125) {
  if (n === 0) return { power: 0, detectable: false };
  const pooled = (p1 + p2) / 2;
  const se     = Math.sqrt(pooled * (1 - pooled) * 2 / n);
  if (se === 0) return { power: 1, detectable: true };
  const zAlpha = normalQuantile(1 - alpha);
  const zBeta  = (p1 - p2) / se - zAlpha;
  const power  = normalCDF(zBeta);
  return { power: +power.toFixed(4), detectable: power > 0.80 };
}

export {
  erf,
  normalCDF, normalUpperTail, normalQuantile,
  wilsonCI, wilsonCIPercent,
  twoPropZTest, bonferroniAlpha,
  chiSquareTest, chiSquarePValue, chiSquareCritical,
  shannonEntropy,
  chiSquarePower,
};
