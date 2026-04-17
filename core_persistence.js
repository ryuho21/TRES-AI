/**
 * core_persistence.js — localStorage layer with quota management v5.4
 *
 * FIXES (v5.4):
 * - Added KEYS.LAST_SCORES ('sw5_last_scores') for prediction retention (FIX-K)
 *
 * PRESERVED from v5.2:
 * - FIX-E: KEYS.VALIDATION_CACHE constant (consistent key across all paths)
 * - Prediction log trimmed to 500 entries on every save
 */

const KEYS = {
  USER_DRAWS:       'sw5_user_draws',
  PREDICTIONS:      'sw5_predictions',
  WEIGHTS:          'sw5_weights',
  VALIDATION_CACHE: 'sw5_validation',
  LAST_SCORES:      'sw5_last_scores',  // FIX-K: prediction retention across reloads
};

/** Read JSON from localStorage safely */
function readStore(key, fallback = null) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

/** Write JSON to localStorage with quota guard */
function writeStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return { ok: true };
  } catch (e) {
    if (e.name === 'QuotaExceededError') {
      return { ok: false, error: 'quota' };
    }
    return { ok: false, error: e.message };
  }
}

/** Append to an array-valued key, with automatic pruning if quota exceeded */
function appendToArray(key, item, maxItems = 2000) {
  let arr = readStore(key, []);
  arr.push(item);
  if (arr.length > maxItems) arr = arr.slice(-maxItems);
  const result = writeStore(key, arr);
  if (result.error === 'quota') {
    arr = arr.slice(-Math.floor(maxItems * 0.5));
    writeStore(key, arr);
  }
  return arr;
}

/** Load user-added draw results */
function loadUserDraws() {
  return readStore(KEYS.USER_DRAWS, []);
}

/** Persist user-added draw results */
function saveUserDraws(draws) {
  return writeStore(KEYS.USER_DRAWS, draws);
}

/** Add one draw result; guards against duplicates via O(1) Set lookup */
function addUserDraw(draw, existingDraws, existingKeys) {
  const key = `${draw.year}-${draw.month}-${draw.day}-${draw.draw_time}`;
  if (existingKeys && existingKeys.has(key)) {
    return { ok: false, error: 'duplicate' };
  }
  // Fallback linear scan if no Set provided
  if (!existingKeys) {
    const isDupe = existingDraws.some(
      d => d.year === draw.year && d.month === draw.month &&
           d.day  === draw.day  && d.draw_time === draw.draw_time
    );
    if (isDupe) return { ok: false, error: 'duplicate' };
  }

  const updated = [...existingDraws, draw];
  const result  = writeStore(KEYS.USER_DRAWS, updated);
  return result.ok ? { ok: true, draws: updated } : result;
}

/** Load prediction log */
function loadPredictions() {
  return readStore(KEYS.PREDICTIONS, []);
}

/**
 * Save prediction log.
 * FIX: Always trims to 500 entries (not only on quota error).
 * BUG-B preserved: prevents unbounded growth.
 */
function savePredictions(preds) {
  const trimmed = preds.length > 500 ? preds.slice(-500) : preds;
  return writeStore(KEYS.PREDICTIONS, trimmed);
}

/** Load layer weights */
function loadWeights() {
  return readStore(KEYS.WEIGHTS, null);
}

/** Save layer weights */
function saveWeights(weights) {
  return writeStore(KEYS.WEIGHTS, weights);
}

/**
 * Invalidate validation cache.
 * FIX-E: Uses KEYS.VALIDATION_CACHE constant — previously two different
 * key strings ('sw5_validation' and 'sw5_validation_cache') were used
 * in different code paths, causing stale cache to persist after new draws.
 */
function clearValidationCache() {
  localStorage.removeItem(KEYS.VALIDATION_CACHE);
}

/** Load cached validation report */
function loadValidationCache() {
  const cache = readStore(KEYS.VALIDATION_CACHE, null);
  // Guard against stale error objects or incomplete reports
  if (!cache || cache.error || !cache.results || !cache.significance) {
    clearValidationCache();
    return null;
  }
  return cache;
}

/** Save validation report */
function saveValidationCache(report) {
  // Only cache valid, complete reports
  if (!report || report.error || !report.results) return { ok: false, error: 'invalid report' };
  return writeStore(KEYS.VALIDATION_CACHE, report);
}

/** Load last computed prediction scores (retention across reloads) */
function loadLastScores() {
  const cache = readStore(KEYS.LAST_SCORES, null);
  // FIX-S: reject stale pre-v5.4 caches that lack the hotpos layer
  if (!cache || !cache.scores?.length || cache.scores[0]?.layers?.hotpos === undefined) {
    return null;
  }
  return cache;
}

/** Save last computed prediction scores (top-20 + metadata) */
function saveLastScores(scores, slot, dow) {
  if (!scores?.length) return { ok: false, error: 'empty scores' };
  return writeStore(KEYS.LAST_SCORES, {
    scores: scores.slice(0, 20),
    slot,
    dow,
    ts: new Date().toISOString(),
  });
}

/** Clear all app data (full reset) */
function clearAllData() {
  Object.values(KEYS).forEach(k => localStorage.removeItem(k));
}

export {
  KEYS,
  readStore, writeStore, appendToArray,
  loadUserDraws, saveUserDraws, addUserDraw,
  loadPredictions, savePredictions,
  loadWeights, saveWeights,
  loadLastScores, saveLastScores,
  clearValidationCache, loadValidationCache, saveValidationCache,
  clearAllData,
};
