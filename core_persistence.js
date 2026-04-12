/**
 * persistence.js — Clean localStorage layer with quota management
 * Pure functions: no side effects except storage reads/writes
 */

const KEYS = {
  USER_DRAWS: 'sw5_user_draws',
  PREDICTIONS: 'sw5_predictions',
  WEIGHTS: 'sw5_weights',
  VALIDATION_CACHE: 'sw5_validation_cache',
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
  // Trim to maxItems before writing
  if (arr.length > maxItems) arr = arr.slice(-maxItems);
  const result = writeStore(key, arr);
  if (result.error === 'quota') {
    arr = arr.slice(-Math.floor(maxItems * 0.5));
    writeStore(key, arr);
  }
  return arr;
}

/** Load user-added draw results (NOT the base statistical dataset) */
function loadUserDraws() {
  return readStore(KEYS.USER_DRAWS, []);
}

/** Persist user-added draw results */
function saveUserDraws(draws) {
  return writeStore(KEYS.USER_DRAWS, draws);
}

/** Add one draw result; guards against duplicates */
function addUserDraw(draw, existingDraws) {
  const isDupe = existingDraws.some(
    d => d.year === draw.year && d.month === draw.month &&
         d.day === draw.day && d.draw_time === draw.draw_time
  );
  if (isDupe) return { ok: false, error: 'duplicate' };
  const updated = [...existingDraws, draw];
  const result = writeStore(KEYS.USER_DRAWS, updated);
  return result.ok ? { ok: true, draws: updated } : result;
}

/** Load prediction log */
function loadPredictions() {
  return readStore(KEYS.PREDICTIONS, []);
}

/** Save prediction log with size guard */
function savePredictions(preds) {
  // Keep only last 500 to prevent unbounded growth
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

/** Invalidate validation cache (call when new draws are added) */
function clearValidationCache() {
  localStorage.removeItem(KEYS.VALIDATION_CACHE);
}

/** Load cached validation report */
function loadValidationCache() {
  return readStore(KEYS.VALIDATION_CACHE, null);
}

/** Save validation report */
function saveValidationCache(report) {
  return writeStore(KEYS.VALIDATION_CACHE, report);
}

export {
  KEYS,
  loadUserDraws, saveUserDraws, addUserDraw,
  loadPredictions, savePredictions,
  loadWeights, saveWeights,
  clearValidationCache, loadValidationCache, saveValidationCache,
};
