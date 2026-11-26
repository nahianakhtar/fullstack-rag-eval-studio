import { randomBytes } from "node:crypto";
import { createServer } from "node:http";
import { performance } from "node:perf_hooks";

const SECRET_PATTERNS = [
  { re: /Bearer\s+[A-Za-z0-9._\-+/=]{8,}/gi, token: "Bearer [REDACTED_TOKEN]" },
  { re: /\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{12,}\b/g, token: "[REDACTED_API_KEY]" },
  { re: /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, token: "[REDACTED_GH_TOKEN]" },
  { re: /\bAKIA[0-9A-Z]{16}\b/g, token: "[REDACTED_AWS_KEY]" },
  { re: /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g, token: "[REDACTED_EMAIL]" },
  { re: /\b\d{3}-\d{2}-\d{4}\b/g, token: "[REDACTED_SSN]" }
];

const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
  "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
  "to", "using", "via", "was", "were", "with"
]);

/**
 * Strict recursive JSON-Schema validator paired with deterministic PII/key redaction.
 */
export class SchemaToolValidator {
  redact(value) {
    let redactionCount = 0;
    const scrub = (node) => {
      if (typeof node === "string") {
        let out = node;
        for (const { re, token } of SECRET_PATTERNS) {
          out = out.replace(re, () => {
            redactionCount += 1;
            return token;
          });
        }
        return out;
      }
      if (Array.isArray(node)) {
        return node.map((item) => scrub(item));
      }
      if (node !== null && typeof node === "object") {
        const copy = {};
        for (const [k, v] of Object.entries(node)) {
          copy[k] = scrub(v);
        }
        return copy;
      }
      return node;
    };
    return { sanitized: scrub(value), redactionCount };
  }

  validate(value, schema, path = "$") {
    const errors = [];
    if (!schema || typeof schema !== "object") {
      return errors;
    }

    if (schema.type) {
      const okType = this.#matchesType(value, schema.type);
      if (!okType) {
        errors.push(`${path}: expected type '${schema.type}'`);
        return errors;
      }
    }

    if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
      errors.push(`${path}: value not in allowed enum [${schema.enum.join(", ")}]`);
    }

    if (typeof value === "string") {
      if (typeof schema.minLength === "number" && value.length < schema.minLength) {
        errors.push(`${path}: string length ${value.length} < minLength ${schema.minLength}`);
      }
      if (typeof schema.maxLength === "number" && value.length > schema.maxLength) {
        errors.push(`${path}: string length ${value.length} > maxLength ${schema.maxLength}`);
      }
      if (typeof schema.pattern === "string" && !(new RegExp(schema.pattern).test(value))) {
        errors.push(`${path}: string does not match pattern '${schema.pattern}'`);
      }
    }

    if (typeof value === "number") {
      if (typeof schema.minimum === "number" && value < schema.minimum) {
        errors.push(`${path}: ${value} < minimum ${schema.minimum}`);
      }
      if (typeof schema.maximum === "number" && value > schema.maximum) {
        errors.push(`${path}: ${value} > maximum ${schema.maximum}`);
      }
    }

    if (Array.isArray(value) && schema.items) {
      value.forEach((elem, idx) => {
        errors.push(...this.validate(elem, schema.items, `${path}[${idx}]`));
      });
    }

    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      const props = schema.properties ?? {};
      if (Array.isArray(schema.required)) {
        for (const reqKey of schema.required) {
          if (!(reqKey in value) || value[reqKey] === undefined) {
            errors.push(`${path}.${reqKey}: missing required property`);
          }
        }
      }
      if (schema.additionalProperties === false) {
        for (const key of Object.keys(value)) {
          if (!(key in props)) {
            errors.push(`${path}.${key}: additional property not permitted`);
          }
        }
      }
      for (const [key, subSchema] of Object.entries(props)) {
        if (key in value && value[key] !== undefined) {
          errors.push(...this.validate(value[key], subSchema, `${path}.${key}`));
        }
      }
    }

    return errors;
  }

  validateAndRedact(value, schema) {
    const { sanitized, redactionCount } = this.redact(value);
    const errors = this.validate(sanitized, schema);
    return {
      valid: errors.length === 0,
      errors,
      sanitized,
      redactionCount
    };
  }

  #matchesType(val, expected) {
    switch (expected) {
      case "object":
        return val !== null && typeof val === "object" && !Array.isArray(val);
      case "array":
        return Array.isArray(val);
      case "string":
        return typeof val === "string";
      case "number":
        return typeof val === "number" && Number.isFinite(val);
      case "integer":
        return typeof val === "number" && Number.isInteger(val);
      case "boolean":
        return typeof val === "boolean";
      default:
        return true;
    }
  }
}

function normalizeText(text) {
  return String(text ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function embedNgramCosine(text, dims = 256) {
  const norm = normalizeText(text);
  const vec = new Float64Array(dims);
  if (!norm) {
    return vec;
  }
  const tokens = norm.split(" ");
  const features = [];
  for (const tok of tokens) {
    if (!STOPWORDS.has(tok)) {
      features.push(`w:${tok}`);
      if (tok.length > 4) {
        features.push(`s:${tok.slice(0, 4)}`);
      }
    }
  }
  const compact = norm.replace(/\s+/g, "_");
  for (let i = 0; i + 3 <= compact.length; i += 1) {
    features.push(`c:${compact.slice(i, i + 3)}`);
  }

  for (const feat of features) {
    let h = 2166136261;
    for (let i = 0; i < feat.length; i += 1) {
      h ^= feat.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const idx = (h >>> 0) % dims;
    const sign = (h & 1) === 0 ? 1.0 : -1.0;
    const weight = feat.startsWith("w:") ? 2.0 : 0.65;
    vec[idx] += sign * weight;
  }

  let sumSq = 0.0;
  for (let i = 0; i < dims; i += 1) {
    sumSq += vec[i] * vec[i];
  }
  if (sumSq > 0) {
    const inv = 1.0 / Math.sqrt(sumSq);
    for (let i = 0; i < dims; i += 1) {
      vec[i] *= inv;
    }
  }
  return vec;
}

function dotCosine(a, b) {
  let dot = 0.0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
  }
  return dot;
}

/**
 * Cosine character/word n-gram semantic LRU cache with TTL expiry and hit-rate telemetry.
 */
export class SemanticLruCache {
  #entries = new Map();
  #maxEntries;
  #ttlMs;
  #similarityThreshold;
  #dimensions;
  #nowFn;
  #exactHits = 0;
  #semanticHits = 0;
  #misses = 0;
  #evictions = 0;

  constructor(options = {}) {
    this.#maxEntries = options.maxEntries ?? 256;
    this.#ttlMs = options.ttlMs ?? 60_000;
    this.#similarityThreshold = options.similarityThreshold ?? 0.84;
    this.#dimensions = options.dimensions ?? 256;
    this.#nowFn = options.nowFn ?? (() => Date.now());
  }

  set(query, value) {
    const now = this.#nowFn();
    this.#pruneExpired(now);
    const key = normalizeText(query);
    if (this.#entries.has(key)) {
      this.#entries.delete(key);
    } else if (this.#entries.size >= this.#maxEntries) {
      const oldestKey = this.#entries.keys().next().value;
      if (oldestKey !== undefined) {
        this.#entries.delete(oldestKey);
        this.#evictions += 1;
      }
    }
    this.#entries.set(key, {
      key,
      vector: embedNgramCosine(key, this.#dimensions),
      value,
      expiresAt: now + this.#ttlMs
    });
  }

  get(query) {
    const now = this.#nowFn();
    this.#pruneExpired(now);
    const key = normalizeText(query);
    const exact = this.#entries.get(key);
    if (exact) {
      this.#entries.delete(key);
      this.#entries.set(key, exact);
      this.#exactHits += 1;
      return { hit: true, matchType: "exact", similarity: 1.0, value: exact.value };
    }

    const qVec = embedNgramCosine(key, this.#dimensions);
    let bestEntry = null;
    let bestSim = -1.0;

    for (const entry of this.#entries.values()) {
      const sim = dotCosine(qVec, entry.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestEntry = entry;
      }
    }

    if (bestEntry && bestSim >= this.#similarityThreshold) {
      this.#entries.delete(bestEntry.key);
      this.#entries.set(bestEntry.key, bestEntry);
      this.#semanticHits += 1;
      return {
        hit: true,
        matchType: "semantic",
        similarity: Number(bestSim.toFixed(4)),
        value: bestEntry.value
      };
    }

    this.#misses += 1;
    return { hit: false, matchType: "miss", similarity: Math.max(0, Number(bestSim.toFixed(4))) };
  }

  stats() {
    this.#pruneExpired(this.#nowFn());
    const total = this.#exactHits + this.#semanticHits + this.#misses;
    return {
      size: this.#entries.size,
      maxEntries: this.#maxEntries,
      exactHits: this.#exactHits,
      semanticHits: this.#semanticHits,
      misses: this.#misses,
      evictions: this.#evictions,
      hitRate: total === 0 ? 0 : Number(((this.#exactHits + this.#semanticHits) / total).toFixed(4))
    };
  }

  #pruneExpired(now) {
    for (const [key, entry] of this.#entries.entries()) {
      if (entry.expiresAt <= now) {
        this.#entries.delete(key);
      }
    }
  }
}

/**
 * Sliding-window token & USD cost governor with automatic fallback model routing.
 */
export class TokenBudgetCircuitBreaker {
  #windowMs;
  #maxTokensPerWindow;
  #maxCostUsdPerWindow;
  #softFallbackRatio;
  #failureThreshold;
  #cooldownMs;
  #primaryModel;
  #fallbackModel;
  #nowFn;
  #ledger = [];
  #state = "CLOSED";
  #consecutiveFailures = 0;
  #openedAt = 0;

  constructor(options = {}) {
    this.#windowMs = options.windowMs ?? 60_000;
    this.#maxTokensPerWindow = options.maxTokensPerWindow ?? 50_000;
    this.#maxCostUsdPerWindow = options.maxCostUsdPerWindow ?? 2.50;
    this.#softFallbackRatio = options.softFallbackRatio ?? 0.75;
    this.#failureThreshold = options.failureThreshold ?? 3;
    this.#cooldownMs = options.cooldownMs ?? 15_000;
    this.#primaryModel = options.primaryModel ?? "frontier-dense-70b";
    this.#fallbackModel = options.fallbackModel ?? "local-sparse-moe-q4";
    this.#nowFn = options.nowFn ?? (() => Date.now());
  }

  route(request = {}) {
    const now = this.#nowFn();
    this.#pruneWindow(now);
    const estTokens = request.estimatedTokens ?? 256;
    const estCost = request.estimatedCostUsd ?? 0.002;

    if (this.#state === "OPEN" && now - this.#openedAt >= this.#cooldownMs) {
      this.#state = "HALF_OPEN";
    }

    const { windowTokens, windowCostUsd } = this.#totals();
    const projectedTokens = windowTokens + estTokens;
    const projectedCost = windowCostUsd + estCost;

    if (projectedTokens > this.#maxTokensPerWindow || projectedCost > this.#maxCostUsdPerWindow) {
      return {
        allowed: false,
        selectedModel: "none",
        tier: "blocked",
        state: this.#state,
        windowTokens,
        windowCostUsd: Number(windowCostUsd.toFixed(6)),
        reason: "sliding_window_budget_exceeded"
      };
    }

    if (this.#state === "OPEN") {
      return {
        allowed: true,
        selectedModel: this.#fallbackModel,
        tier: "fallback",
        state: this.#state,
        windowTokens,
        windowCostUsd: Number(windowCostUsd.toFixed(6)),
        reason: "primary_circuit_open_fallback"
      };
    }

    const softTokens = this.#maxTokensPerWindow * this.#softFallbackRatio;
    const softCost = this.#maxCostUsdPerWindow * this.#softFallbackRatio;
    if (projectedTokens >= softTokens || projectedCost >= softCost) {
      return {
        allowed: true,
        selectedModel: this.#fallbackModel,
        tier: "fallback",
        state: this.#state,
        windowTokens,
        windowCostUsd: Number(windowCostUsd.toFixed(6)),
        reason: "soft_budget_fallback_routed"
      };
    }

    return {
      allowed: true,
      selectedModel: this.#primaryModel,
      tier: "primary",
      state: this.#state,
      windowTokens,
      windowCostUsd: Number(windowCostUsd.toFixed(6)),
      reason: this.#state === "HALF_OPEN" ? "half_open_probe" : "nominal_primary"
    };
  }

  record({ tokens = 0, costUsd = 0, ok = true } = {}) {
    const now = this.#nowFn();
    this.#pruneWindow(now);
    this.#ledger.push({ ts: now, tokens, costUsd });

    if (ok) {
      this.#consecutiveFailures = 0;
      this.#state = "CLOSED";
    } else {
      this.#consecutiveFailures += 1;
      if (this.#state === "HALF_OPEN" || this.#consecutiveFailures >= this.#failureThreshold) {
        this.#state = "OPEN";
        this.#openedAt = now;
      }
    }
  }

  snapshot() {
    this.#pruneWindow(this.#nowFn());
    const { windowTokens, windowCostUsd } = this.#totals();
    return {
      state: this.#state,
      consecutiveFailures: this.#consecutiveFailures,
      windowTokens,
      windowCostUsd: Number(windowCostUsd.toFixed(6)),
      primaryModel: this.#primaryModel,
      fallbackModel: this.#fallbackModel
    };
  }

  #pruneWindow(now) {
    const cutoff = now - this.#windowMs;
    while (this.#ledger.length > 0 && this.#ledger[0].ts <= cutoff) {
      this.#ledger.shift();
    }
  }

  #totals() {
    let windowTokens = 0;
    let windowCostUsd = 0;
    for (const item of this.#ledger) {
      windowTokens += item.tokens;
      windowCostUsd += item.costUsd;
    }
    return { windowTokens, windowCostUsd };
  }
}

/**
 * Deterministic per-client sliding-window request & token-budget rate limiter
 * with bounded client-bucket LRU, non-penalizing rejection, and exact 1-ms Retry-After math.
 */
export class SlidingWindowTokenRateLimiter {
  #windowMs;
  #maxRequestsPerWindow;
  #maxTokensPerWindow;
  #maxTrackedClients;
  #nowFn;
  #clients = new Map();
  #totalAllowed = 0;
  #totalRejected = 0;
  #totalTokensConsumed = 0;

  constructor(options = {}) {
    this.#windowMs = Math.max(1, options.windowMs ?? 60_000);
    this.#maxRequestsPerWindow = Math.max(1, options.maxRequestsPerWindow ?? 60);
    this.#maxTokensPerWindow = Math.max(1, options.maxTokensPerWindow ?? 12_000);
    this.#maxTrackedClients = Math.max(1, options.maxTrackedClients ?? 1_024);
    this.#nowFn = options.nowFn ?? (() => Date.now());
  }

  check(clientId = "anonymous", estimatedTokens = 1) {
    return this.#evaluate(clientId, estimatedTokens, false);
  }

  consume(clientId = "anonymous", tokens = 1) {
    return this.#evaluate(clientId, tokens, true);
  }

  reconcile(clientId = "anonymous", deltaTokens = 0) {
    const key = this.#normalizeClient(clientId);
    const entries = this.#clients.get(key);
    const delta = Math.trunc(Number(deltaTokens) || 0);
    if (!entries || entries.length === 0 || delta === 0) {
      return;
    }
    const last = entries[entries.length - 1];
    const nextTokens = Math.max(0, last.tokens + delta);
    const appliedDelta = nextTokens - last.tokens;
    last.tokens = nextTokens;
    this.#totalTokensConsumed = Math.max(0, this.#totalTokensConsumed + appliedDelta);
  }

  toHeaders(decision) {
    const headers = {
      "X-RateLimit-Limit-Requests": String(decision.limitRequests),
      "X-RateLimit-Remaining-Requests": String(decision.remainingRequests),
      "X-RateLimit-Limit-Tokens": String(decision.limitTokens),
      "X-RateLimit-Remaining-Tokens": String(decision.remainingTokens),
      "X-RateLimit-Reset-Ms": String(decision.resetAfterMs)
    };
    if (!decision.allowed) {
      headers["Retry-After"] = String(Math.max(1, Math.ceil(decision.retryAfterMs / 1000)));
    }
    return headers;
  }

  snapshot() {
    const now = this.#nowFn();
    const cutoff = now - this.#windowMs;
    for (const [key, entries] of this.#clients.entries()) {
      this.#pruneEntries(entries, cutoff);
      if (entries.length === 0) {
        this.#clients.delete(key);
      }
    }
    return {
      activeClients: this.#clients.size,
      windowMs: this.#windowMs,
      maxRequestsPerWindow: this.#maxRequestsPerWindow,
      maxTokensPerWindow: this.#maxTokensPerWindow,
      totalAllowed: this.#totalAllowed,
      totalRejected: this.#totalRejected,
      totalTokensConsumed: this.#totalTokensConsumed
    };
  }

  #normalizeClient(clientId) {
    return String(clientId || "anonymous").trim().slice(0, 128) || "anonymous";
  }

  #pruneEntries(entries, cutoff) {
    while (entries.length > 0 && entries[0].ts <= cutoff) {
      entries.shift();
    }
  }

  #evictIdleOrOldestClient(cutoff) {
    for (const [k, bucket] of this.#clients.entries()) {
      this.#pruneEntries(bucket, cutoff);
      if (bucket.length === 0) {
        this.#clients.delete(k);
        if (this.#clients.size < this.#maxTrackedClients) {
          return;
        }
      }
    }
    if (this.#clients.size >= this.#maxTrackedClients) {
      const oldestKey = this.#clients.keys().next().value;
      if (oldestKey !== undefined) {
        this.#clients.delete(oldestKey);
      }
    }
  }

  #evaluate(rawClientId, rawTokens, mutate) {
    const now = this.#nowFn();
    const cutoff = now - this.#windowMs;
    const key = this.#normalizeClient(rawClientId);
    const costTokens = Math.max(0, Math.ceil(Number(rawTokens ?? 1)));

    let entries = this.#clients.get(key);
    if (entries) {
      this.#pruneEntries(entries, cutoff);
      if (entries.length === 0 && !mutate) {
        this.#clients.delete(key);
        entries = undefined;
      }
    }

    const activeEntries = entries ?? [];
    const usedRequests = activeEntries.length;
    let usedTokens = 0;
    for (const item of activeEntries) {
      usedTokens += item.tokens;
    }

    if (costTokens > this.#maxTokensPerWindow) {
      if (mutate) {
        this.#totalRejected += 1;
      }
      const resetAfterMs =
        activeEntries.length > 0
          ? Math.max(0, activeEntries[activeEntries.length - 1].ts + this.#windowMs - now)
          : this.#windowMs;
      return {
        allowed: false,
        clientId: key,
        limitRequests: this.#maxRequestsPerWindow,
        remainingRequests: Math.max(0, this.#maxRequestsPerWindow - usedRequests),
        limitTokens: this.#maxTokensPerWindow,
        remainingTokens: Math.max(0, this.#maxTokensPerWindow - usedTokens),
        usedRequests,
        usedTokens,
        requestedTokens: costTokens,
        retryAfterMs: this.#windowMs,
        resetAfterMs,
        reason: "request_exceeds_window_capacity"
      };
    }

    const reqExceeded = usedRequests + 1 > this.#maxRequestsPerWindow;
    const tokExceeded = usedTokens + costTokens > this.#maxTokensPerWindow;

    if (reqExceeded || tokExceeded) {
      let reqRetryMs = 0;
      if (reqExceeded && activeEntries.length > 0) {
        const needIdx = Math.min(
          activeEntries.length - 1,
          Math.max(0, usedRequests + 1 - this.#maxRequestsPerWindow - 1)
        );
        reqRetryMs = Math.max(1, activeEntries[needIdx].ts + this.#windowMs - now);
      }

      let tokRetryMs = 0;
      if (tokExceeded && activeEntries.length > 0) {
        const deficit = usedTokens + costTokens - this.#maxTokensPerWindow;
        let freed = 0;
        for (const entry of activeEntries) {
          freed += entry.tokens;
          if (freed >= deficit) {
            tokRetryMs = Math.max(1, entry.ts + this.#windowMs - now);
            break;
          }
        }
      }

      const retryAfterMs = Math.max(1, reqRetryMs, tokRetryMs);
      const resetAfterMs =
        activeEntries.length > 0
          ? Math.max(0, activeEntries[activeEntries.length - 1].ts + this.#windowMs - now)
          : 0;

      if (mutate) {
        this.#totalRejected += 1;
        if (entries && entries.length === 0) {
          this.#clients.delete(key);
        }
      }

      return {
        allowed: false,
        clientId: key,
        limitRequests: this.#maxRequestsPerWindow,
        remainingRequests: Math.max(0, this.#maxRequestsPerWindow - usedRequests),
        limitTokens: this.#maxTokensPerWindow,
        remainingTokens: Math.max(0, this.#maxTokensPerWindow - usedTokens),
        usedRequests,
        usedTokens,
        requestedTokens: costTokens,
        retryAfterMs,
        resetAfterMs,
        reason: tokExceeded ? "sliding_window_tokens_exceeded" : "sliding_window_requests_exceeded"
      };
    }

    if (mutate) {
      if (!entries) {
        if (this.#clients.size >= this.#maxTrackedClients) {
          this.#evictIdleOrOldestClient(cutoff);
        }
        entries = [];
      } else {
        this.#clients.delete(key);
      }
      entries.push({ ts: now, tokens: costTokens });
      this.#clients.set(key, entries);
      this.#totalAllowed += 1;
      this.#totalTokensConsumed += costTokens;
    }

    const nextRequests = mutate ? usedRequests + 1 : usedRequests;
    const nextTokens = mutate ? usedTokens + costTokens : usedTokens;
    const oldestTs = (entries && entries.length > 0) ? entries[0].ts : now;
    const resetAfterMs =
      mutate || activeEntries.length > 0 ? Math.max(0, oldestTs + this.#windowMs - now) : 0;

    return {
      allowed: true,
      clientId: key,
      limitRequests: this.#maxRequestsPerWindow,
      remainingRequests: Math.max(0, this.#maxRequestsPerWindow - nextRequests),
      limitTokens: this.#maxTokensPerWindow,
      remainingTokens: Math.max(0, this.#maxTokensPerWindow - nextTokens),
      usedRequests: nextRequests,
      usedTokens: nextTokens,
      requestedTokens: costTokens,
      retryAfterMs: 0,
      resetAfterMs,
      reason: "allowed"
    };
  }
}

const OTLP_HISTOGRAM_BOUNDS_MS = Object.freeze([0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250]);
const TRACEPARENT_RE = /^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$/i;

/**
 * Zero-dependency OpenTelemetry (OTLP/HTTP JSON v1) trace span & latency histogram exporter
 * with W3C Trace Context (`traceparent`) propagation and automatic PII/secret redaction.
 */
export class OtlpTraceExporter {
  static HISTOGRAM_BOUNDS_MS = OTLP_HISTOGRAM_BOUNDS_MS;

  #serviceName;
  #serviceVersion;
  #scopeName;
  #maxSpans;
  #onSpan;
  #nowNsFn;
  #validator;
  #spans = [];
  #histograms = new Map();
  #totalSpansRecorded = 0;
  #droppedSpans = 0;
  #errorSpans = 0;
  #rateLimitAllowed = 0;
  #rateLimitRejected = 0;

  constructor(options = {}) {
    this.#serviceName = options.serviceName ?? "fullstack-rag-eval-studio";
    this.#serviceVersion = options.serviceVersion ?? "1.2.0";
    this.#scopeName = options.scopeName ?? "fullstack_rag_eval_studio.otlp";
    this.#maxSpans = Math.max(1, options.maxSpans ?? 512);
    this.#onSpan = typeof options.onSpan === "function" ? options.onSpan : null;
    this.#validator = options.validator ?? new SchemaToolValidator();

    const anchorEpochNs = BigInt(Date.now()) * 1_000_000n;
    const anchorPerfMs = performance.now();
    this.#nowNsFn =
      options.nowNsFn ??
      (() => {
        const deltaNs = BigInt(Math.max(0, Math.round((performance.now() - anchorPerfMs) * 1_000_000)));
        return anchorEpochNs + deltaNs;
      });
  }

  parseTraceparent(header) {
    if (typeof header !== "string") {
      return null;
    }
    const match = TRACEPARENT_RE.exec(header.trim());
    if (!match) {
      return null;
    }
    const traceId = match[1].toLowerCase();
    const spanId = match[2].toLowerCase();
    const traceFlags = match[3].toLowerCase();
    if (/^0+$/.test(traceId) || /^0+$/.test(spanId)) {
      return null;
    }
    return { version: "00", traceId, spanId, traceFlags };
  }

  formatTraceparent({ traceId, spanId, traceFlags = "01" }) {
    return `00-${String(traceId).toLowerCase()}-${String(spanId).toLowerCase()}-${String(traceFlags).toLowerCase()}`;
  }

  recordRateLimitDecision(allowed) {
    if (allowed) {
      this.#rateLimitAllowed += 1;
    } else {
      this.#rateLimitRejected += 1;
    }
  }

  startSpan(name, options = {}) {
    const parent =
      options.parentContext ??
      (typeof options.traceparent === "string" ? this.parseTraceparent(options.traceparent) : null);
    const traceId = parent?.traceId ?? randomBytes(16).toString("hex");
    const spanId = randomBytes(8).toString("hex");
    const parentSpanId = parent?.spanId ?? "";
    const traceFlags = parent?.traceFlags ?? "01";
    const traceparent = this.formatTraceparent({ traceId, spanId, traceFlags });
    const kind = options.kind ?? 1;
    const startNs = this.#readNs();
    const attrMap = new Map();
    const events = [];
    let status = { code: 1 };
    let finalized = null;

    if (options.attributes && typeof options.attributes === "object") {
      for (const [k, v] of Object.entries(options.attributes)) {
        if (v !== undefined) {
          attrMap.set(k, v);
        }
      }
    }

    return {
      traceId,
      spanId,
      parentSpanId,
      traceFlags,
      traceparent,
      setAttribute: (key, value) => {
        if (!finalized && value !== undefined) {
          attrMap.set(key, value);
        }
      },
      setAttributes: (attrs = {}) => {
        if (!finalized && attrs && typeof attrs === "object") {
          for (const [k, v] of Object.entries(attrs)) {
            if (v !== undefined) {
              attrMap.set(k, v);
            }
          }
        }
      },
      addEvent: (eventName, eventAttrs = {}) => {
        if (!finalized) {
          events.push({
            timeUnixNano: String(this.#readNs()),
            name: this.#validator.redact(String(eventName)).sanitized,
            attributes: this.#encodeAttributes(eventAttrs)
          });
        }
      },
      setStatus: (code, message) => {
        if (!finalized) {
          status = message
            ? { code, message: this.#validator.redact(String(message)).sanitized }
            : { code };
        }
      },
      end: (endAttributes) => {
        if (finalized) {
          return finalized;
        }
        if (endAttributes && typeof endAttributes === "object") {
          for (const [k, v] of Object.entries(endAttributes)) {
            if (v !== undefined) {
              attrMap.set(k, v);
            }
          }
        }
        const rawEndNs = this.#readNs();
        const endNs = rawEndNs >= startNs ? rawEndNs : startNs;
        const durationMs = Number(endNs - startNs) / 1_000_000.0;
        const metricKey = String(options.metricRoute ?? name);
        this.#recordHistogram(metricKey, durationMs);

        const span = {
          traceId,
          spanId,
          ...(parentSpanId ? { parentSpanId } : {}),
          name: this.#validator.redact(String(name)).sanitized,
          kind,
          startTimeUnixNano: String(startNs),
          endTimeUnixNano: String(endNs),
          attributes: this.#encodeAttributes(Object.fromEntries(attrMap.entries())),
          events,
          status
        };

        this.#appendSpan(span);
        finalized = span;
        return span;
      }
    };
  }

  exportOtlpJson() {
    const nowNs = String(this.#readNs());
    const resource = {
      attributes: [
        { key: "service.name", value: { stringValue: this.#serviceName } },
        { key: "service.version", value: { stringValue: this.#serviceVersion } },
        { key: "telemetry.sdk.language", value: { stringValue: "nodejs" } }
      ]
    };

    const histogramDataPoints = [];
    for (const [routeKey, h] of [...this.#histograms.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
      histogramDataPoints.push({
        attributes: [{ key: "http.route", value: { stringValue: routeKey } }],
        timeUnixNano: nowNs,
        count: String(h.count),
        sum: Number(h.sumMs.toFixed(6)),
        min: Number(h.minMs.toFixed(6)),
        max: Number(h.maxMs.toFixed(6)),
        bucketCounts: h.buckets.map((c) => String(c)),
        explicitBounds: [...OTLP_HISTOGRAM_BOUNDS_MS]
      });
    }

    return {
      resourceSpans: [
        {
          resource,
          scopeSpans: [
            {
              scope: { name: this.#scopeName, version: this.#serviceVersion },
              spans: [...this.#spans]
            }
          ]
        }
      ],
      resourceMetrics: [
        {
          resource,
          scopeMetrics: [
            {
              scope: { name: this.#scopeName, version: this.#serviceVersion },
              metrics: [
                {
                  name: "http.server.duration",
                  unit: "ms",
                  histogram: {
                    aggregationTemporality: 2,
                    dataPoints: histogramDataPoints
                  }
                },
                {
                  name: "rag.ratelimit.decisions",
                  unit: "1",
                  sum: {
                    aggregationTemporality: 2,
                    isMonotonic: true,
                    dataPoints: [
                      {
                        attributes: [{ key: "decision", value: { stringValue: "allowed" } }],
                        timeUnixNano: nowNs,
                        asInt: String(this.#rateLimitAllowed)
                      },
                      {
                        attributes: [{ key: "decision", value: { stringValue: "rejected" } }],
                        timeUnixNano: nowNs,
                        asInt: String(this.#rateLimitRejected)
                      }
                    ]
                  }
                }
              ]
            }
          ]
        }
      ],
      summary: this.stats()
    };
  }

  stats() {
    return {
      serviceName: this.#serviceName,
      serviceVersion: this.#serviceVersion,
      spansBuffered: this.#spans.length,
      maxSpans: this.#maxSpans,
      totalSpansRecorded: this.#totalSpansRecorded,
      droppedSpans: this.#droppedSpans,
      errorSpans: this.#errorSpans,
      rateLimitAllowed: this.#rateLimitAllowed,
      rateLimitRejected: this.#rateLimitRejected
    };
  }

  clear() {
    this.#spans.length = 0;
    this.#histograms.clear();
  }

  #readNs() {
    const v = this.#nowNsFn();
    return typeof v === "bigint" ? v : BigInt(Math.trunc(Number(v)));
  }

  #appendSpan(span) {
    if (this.#spans.length >= this.#maxSpans) {
      this.#spans.shift();
      this.#droppedSpans += 1;
    }
    this.#spans.push(span);
    this.#totalSpansRecorded += 1;
    if (span.status?.code === 2) {
      this.#errorSpans += 1;
    }
    if (this.#onSpan) {
      this.#onSpan(span);
    }
  }

  #recordHistogram(key, durationMs) {
    const ms = Math.max(0, durationMs);
    let hist = this.#histograms.get(key);
    if (!hist) {
      hist = {
        count: 0,
        sumMs: 0,
        minMs: ms,
        maxMs: ms,
        buckets: new Array(OTLP_HISTOGRAM_BOUNDS_MS.length + 1).fill(0)
      };
      this.#histograms.set(key, hist);
    }
    hist.count += 1;
    hist.sumMs += ms;
    hist.minMs = Math.min(hist.minMs, ms);
    hist.maxMs = Math.max(hist.maxMs, ms);
    let bucketIdx = OTLP_HISTOGRAM_BOUNDS_MS.length;
    for (let i = 0; i < OTLP_HISTOGRAM_BOUNDS_MS.length; i += 1) {
      if (ms <= OTLP_HISTOGRAM_BOUNDS_MS[i]) {
        bucketIdx = i;
        break;
      }
    }
    hist.buckets[bucketIdx] += 1;
  }

  #encodeAttributes(obj) {
    const attrs = [];
    for (const [k, rawVal] of Object.entries(obj ?? {})) {
      if (rawVal === undefined || rawVal === null) {
        continue;
      }
      attrs.push({ key: k, value: this.#encodeAnyValue(rawVal) });
    }
    return attrs;
  }

  #encodeAnyValue(val) {
    if (typeof val === "boolean") {
      return { boolValue: val };
    }
    if (typeof val === "number") {
      if (Number.isInteger(val)) {
        return { intValue: String(val) };
      }
      return { doubleValue: Number(val.toFixed(6)) };
    }
    if (Array.isArray(val)) {
      return { arrayValue: { values: val.map((item) => this.#encodeAnyValue(item)) } };
    }
    const clean = this.#validator.redact(String(val)).sanitized;
    return { stringValue: clean };
  }
}

function contentWords(text) {
  return normalizeText(text)
    .split(" ")
    .filter((tok) => tok.length >= 2 && !STOPWORDS.has(tok));
}

function percentile(sortedVals, p) {
  if (sortedVals.length === 0) {
    return 0;
  }
  const idx = Math.min(sortedVals.length - 1, Math.max(0, Math.ceil(p * sortedVals.length) - 1));
  return sortedVals[idx];
}

/**
 * Deterministic RAG evaluation engine computing Faithfulness, Context Precision (AP@K),
 * Recall@K, and high-resolution p50/p95 latency metrics.
 */
export class GoldenEvalRunner {
  #topK;

  constructor(options = {}) {
    this.#topK = options.topK ?? 5;
  }

  evaluateCase(item, k = this.#topK) {
    const topDocs = (item.retrievedDocs ?? []).slice(0, k);
    const relevantSet = new Set(item.relevantDocIds ?? []);

    // 1. Recall@K
    let hits = 0;
    for (const doc of topDocs) {
      if (relevantSet.has(doc.id)) {
        hits += 1;
      }
    }
    const recallAtK = relevantSet.size === 0 ? 1.0 : hits / relevantSet.size;

    // 2. Context Precision (Average Precision @ K)
    let relevantSeen = 0;
    let precisionAccum = 0.0;
    topDocs.forEach((doc, idx) => {
      if (relevantSet.has(doc.id)) {
        relevantSeen += 1;
        precisionAccum += relevantSeen / (idx + 1);
      }
    });
    const denom = Math.min(relevantSet.size, topDocs.length);
    const contextPrecision =
      relevantSeen === 0 || denom === 0 ? 0.0 : precisionAccum / relevantSeen;

    // 3. Claim-level Faithfulness against retrieved context tokens
    const contextTokenSet = new Set(
      contentWords(topDocs.map((d) => d.text).join(" "))
    );
    const sentences = String(item.generatedAnswer ?? "")
      .split(/[.!?]+/)
      .map((s) => s.trim())
      .filter(Boolean);

    let faithfulness = 0.0;
    if (sentences.length > 0) {
      let supportedSentences = 0;
      for (const sentence of sentences) {
        const words = contentWords(sentence);
        if (words.length === 0) {
          supportedSentences += 1;
          continue;
        }
        let grounded = 0;
        for (const w of words) {
          if (contextTokenSet.has(w)) {
            grounded += 1;
          }
        }
        if (grounded / words.length >= 0.6) {
          supportedSentences += 1;
        }
      }
      faithfulness = supportedSentences / sentences.length;
    }

    return {
      id: item.id,
      faithfulness: Number(faithfulness.toFixed(4)),
      contextPrecision: Number(contextPrecision.toFixed(4)),
      recallAtK: Number(recallAtK.toFixed(4))
    };
  }

  async runSuite(cases, pipelineFn) {
    const caseMetrics = [];
    const latencies = [];
    const suiteStart = performance.now();

    for (const rawCase of cases) {
      const t0 = performance.now();
      let enriched = rawCase;
      if (typeof pipelineFn === "function") {
        const patch = await pipelineFn(rawCase);
        enriched = { ...rawCase, ...patch };
      }
      const scored = this.evaluateCase(enriched, this.#topK);
      const latencyMs = Number((performance.now() - t0).toFixed(4));
      latencies.push(latencyMs);
      caseMetrics.push({ ...scored, latencyMs });
    }

    const elapsedSec = Math.max(1e-6, (performance.now() - suiteStart) / 1000.0);
    const n = Math.max(1, caseMetrics.length);
    const sortedLat = [...latencies].sort((a, b) => a - b);

    const mean = (key) =>
      Number((caseMetrics.reduce((acc, c) => acc + c[key], 0) / n).toFixed(4));

    return {
      totalCases: caseMetrics.length,
      topK: this.#topK,
      faithfulness: mean("faithfulness"),
      contextPrecision: mean("contextPrecision"),
      recallAtK: mean("recallAtK"),
      meanLatencyMs: mean("latencyMs"),
      p50LatencyMs: Number(percentile(sortedLat, 0.50).toFixed(4)),
      p95LatencyMs: Number(percentile(sortedLat, 0.95).toFixed(4)),
      casesPerSec: Number((caseMetrics.length / elapsedSec).toFixed(1)),
      cases: caseMetrics
    };
  }
}

function meanChunkWordCount(retrievedDocs, topK) {
  const docs = (retrievedDocs ?? []).slice(0, topK);
  if (docs.length === 0) {
    return 0;
  }
  let words = 0;
  for (const doc of docs) {
    const norm = normalizeText(doc.text);
    words += norm ? norm.split(" ").length : 0;
  }
  return words / docs.length;
}

function chunkIdJaccard(baseDocs, candDocs, topK) {
  const setA = new Set((baseDocs ?? []).slice(0, topK).map((d) => d.id));
  const setB = new Set((candDocs ?? []).slice(0, topK).map((d) => d.id));
  if (setA.size === 0 && setB.size === 0) {
    return 1.0;
  }
  let inter = 0;
  for (const id of setA) {
    if (setB.has(id)) {
      inter += 1;
    }
  }
  const union = setA.size + setB.size - inter;
  return union === 0 ? 1.0 : inter / union;
}

function computePairedStats(deltas) {
  const n = deltas.length;
  if (n === 0) {
    return { meanDelta: 0, stdDevDelta: 0, cohensD: 0, pairedTStat: 0, winRate: 0 };
  }
  const mean = deltas.reduce((acc, v) => acc + v, 0) / n;
  let sqSum = 0;
  let wins = 0;
  let nonTies = 0;
  for (const d of deltas) {
    sqSum += (d - mean) * (d - mean);
    if (Math.abs(d) > 1e-6) {
      nonTies += 1;
      if (d > 1e-6) {
        wins += 1;
      }
    }
  }
  const variance = n > 1 ? sqSum / (n - 1) : 0;
  const stdDev = Math.sqrt(variance);
  const cohensD = stdDev > 1e-9 ? mean / stdDev : (Math.abs(mean) > 1e-9 ? Math.sign(mean) * 99.0 : 0);
  const pairedTStat = stdDev > 1e-9 && n > 1 ? mean / (stdDev / Math.sqrt(n)) : 0;
  const winRate = nonTies === 0 ? 0.5 : wins / nonTies;
  return {
    meanDelta: Number(mean.toFixed(4)),
    stdDevDelta: Number(stdDev.toFixed(4)),
    cohensD: Number(cohensD.toFixed(4)),
    pairedTStat: Number(pairedTStat.toFixed(4)),
    winRate: Number(winRate.toFixed(4))
  };
}

/**
 * Pairwise A/B prompt & chunking regression diff analyzer comparing baseline vs. candidate
 * RAG evaluation suites or pre-scored GoldenEvalReports with configurable CI quality gates.
 */
export class RegressionDiffAnalyzer {
  #topK;
  #maxFaithfulnessDrop;
  #maxContextPrecisionDrop;
  #maxRecallDrop;
  #maxCaseRegressions;
  #caseEpsilon;
  #maxP95LatencyRatio;

  constructor(options = {}) {
    this.#topK = options.topK ?? 5;
    this.#maxFaithfulnessDrop = options.maxFaithfulnessDrop ?? 0.01;
    this.#maxContextPrecisionDrop = options.maxContextPrecisionDrop ?? 0.02;
    this.#maxRecallDrop = options.maxRecallDrop ?? 0.01;
    this.#maxCaseRegressions = options.maxCaseRegressions ?? 0;
    this.#caseEpsilon = options.caseEpsilon ?? 0.005;
    this.#maxP95LatencyRatio = options.maxP95LatencyRatio ?? null;
  }

  async #normalizeInput(input) {
    if (Array.isArray(input)) {
      const runner = new GoldenEvalRunner({ topK: this.#topK });
      const report = await runner.runSuite(input);
      return { report, rawById: new Map(input.map((c) => [c.id, c])) };
    }
    if (input && typeof input === "object") {
      if (
        typeof input.faithfulness === "number" &&
        typeof input.contextPrecision === "number" &&
        typeof input.recallAtK === "number" &&
        Array.isArray(input.cases) &&
        (input.cases.length === 0 || typeof input.cases[0].faithfulness === "number")
      ) {
        return { report: input, rawById: new Map() };
      }
      if (Array.isArray(input.cases)) {
        const k = typeof input.topK === "number" ? input.topK : this.#topK;
        const runner = new GoldenEvalRunner({ topK: k });
        const report = await runner.runSuite(input.cases);
        return { report, rawById: new Map(input.cases.map((c) => [c.id, c])) };
      }
    }
    throw new Error("invalid_suite_or_report_input");
  }

  async compare(baselineInput, candidateInput) {
    const baseNorm = await this.#normalizeInput(baselineInput);
    const candNorm = await this.#normalizeInput(candidateInput);
    const baseReport = baseNorm.report;
    const candReport = candNorm.report;

    const baseMap = new Map(baseReport.cases.map((c) => [c.id, c]));
    const candMap = new Map(candReport.cases.map((c) => [c.id, c]));

    const addedCaseIds = [];
    const removedCaseIds = [];
    for (const id of candMap.keys()) {
      if (!baseMap.has(id)) {
        addedCaseIds.push(id);
      }
    }
    for (const id of baseMap.keys()) {
      if (!candMap.has(id)) {
        removedCaseIds.push(id);
      }
    }

    const caseDiffs = [];
    const compositeDeltas = [];
    const jaccardScores = [];
    const chunkLenDeltas = [];
    const answerCosines = [];

    for (const [id, bCase] of baseMap.entries()) {
      const cCase = candMap.get(id);
      if (!cCase) {
        continue;
      }

      const dFaith = Number((cCase.faithfulness - bCase.faithfulness).toFixed(4));
      const dPrec = Number((cCase.contextPrecision - bCase.contextPrecision).toFixed(4));
      const dRec = Number((cCase.recallAtK - bCase.recallAtK).toFixed(4));
      const baseComposite = (bCase.faithfulness + bCase.contextPrecision + bCase.recallAtK) / 3.0;
      const candComposite = (cCase.faithfulness + cCase.contextPrecision + cCase.recallAtK) / 3.0;
      const compositeDelta = Number((candComposite - baseComposite).toFixed(4));
      compositeDeltas.push(compositeDelta);

      const reasons = [];
      if (dFaith < -this.#caseEpsilon) {
        reasons.push("faithfulness_drop");
      }
      if (dPrec < -this.#caseEpsilon) {
        reasons.push("precision_rank_drop");
      }
      if (dRec < -this.#caseEpsilon) {
        reasons.push("recall_chunk_miss");
      }

      let status = "unchanged";
      if (compositeDelta < -this.#caseEpsilon || reasons.length > 0) {
        status = "regressed";
      } else if (compositeDelta > this.#caseEpsilon) {
        status = "improved";
      }

      let chunkOverlapJaccard = null;
      let chunkWordsDelta = null;
      let answerCosineSimilarity = null;
      const rawB = baseNorm.rawById.get(id);
      const rawC = candNorm.rawById.get(id);
      if (rawB && rawC) {
        const jacc = Number(chunkIdJaccard(rawB.retrievedDocs, rawC.retrievedDocs, this.#topK).toFixed(4));
        const wDelta = Number(
          (
            meanChunkWordCount(rawC.retrievedDocs, this.#topK) -
            meanChunkWordCount(rawB.retrievedDocs, this.#topK)
          ).toFixed(2)
        );
        const vecB = embedNgramCosine(rawB.generatedAnswer ?? "", 256);
        const vecC = embedNgramCosine(rawC.generatedAnswer ?? "", 256);
        const ansSim = Number(Math.max(0, dotCosine(vecB, vecC)).toFixed(4));
        chunkOverlapJaccard = jacc;
        chunkWordsDelta = wDelta;
        answerCosineSimilarity = ansSim;
        jaccardScores.push(jacc);
        chunkLenDeltas.push(wDelta);
        answerCosines.push(ansSim);
      }

      caseDiffs.push({
        id,
        status,
        compositeDelta,
        faithfulnessDelta: dFaith,
        contextPrecisionDelta: dPrec,
        recallAtKDelta: dRec,
        reasons,
        chunkOverlapJaccard,
        chunkWordsDelta,
        answerCosineSimilarity
      });
    }

    const regressions = caseDiffs
      .filter((c) => c.status === "regressed")
      .sort((a, b) => a.compositeDelta - b.compositeDelta);
    const improvements = caseDiffs
      .filter((c) => c.status === "improved")
      .sort((a, b) => b.compositeDelta - a.compositeDelta);
    const unchangedCount = caseDiffs.length - regressions.length - improvements.length;

    const deltas = {
      faithfulness: Number((candReport.faithfulness - baseReport.faithfulness).toFixed(4)),
      contextPrecision: Number((candReport.contextPrecision - baseReport.contextPrecision).toFixed(4)),
      recallAtK: Number((candReport.recallAtK - baseReport.recallAtK).toFixed(4)),
      p50LatencyMs: Number(((candReport.p50LatencyMs ?? 0) - (baseReport.p50LatencyMs ?? 0)).toFixed(4)),
      p95LatencyMs: Number(((candReport.p95LatencyMs ?? 0) - (baseReport.p95LatencyMs ?? 0)).toFixed(4))
    };

    const avgOrNull = (arr, digits = 4) =>
      arr.length === 0 ? null : Number((arr.reduce((s, v) => s + v, 0) / arr.length).toFixed(digits));

    const drift = {
      meanChunkOverlapJaccard: avgOrNull(jaccardScores, 4),
      meanChunkWordsDelta: avgOrNull(chunkLenDeltas, 2),
      meanAnswerCosineSimilarity: avgOrNull(answerCosines, 4),
      pairedStats: computePairedStats(compositeDeltas)
    };

    const gateFailures = [];
    if (deltas.faithfulness < -this.#maxFaithfulnessDrop) {
      gateFailures.push(
        `faithfulness dropped ${deltas.faithfulness.toFixed(4)} (allowed <= -${this.#maxFaithfulnessDrop.toFixed(4)})`
      );
    }
    if (deltas.contextPrecision < -this.#maxContextPrecisionDrop) {
      gateFailures.push(
        `contextPrecision dropped ${deltas.contextPrecision.toFixed(4)} (allowed <= -${this.#maxContextPrecisionDrop.toFixed(4)})`
      );
    }
    if (deltas.recallAtK < -this.#maxRecallDrop) {
      gateFailures.push(
        `recallAtK dropped ${deltas.recallAtK.toFixed(4)} (allowed <= -${this.#maxRecallDrop.toFixed(4)})`
      );
    }
    if (regressions.length > this.#maxCaseRegressions) {
      gateFailures.push(
        `regressed cases ${regressions.length} exceeded maxCaseRegressions ${this.#maxCaseRegressions}`
      );
    }
    if (
      typeof this.#maxP95LatencyRatio === "number" &&
      (baseReport.p95LatencyMs ?? 0) > 0 &&
      (candReport.p95LatencyMs ?? 0) / baseReport.p95LatencyMs > this.#maxP95LatencyRatio
    ) {
      const ratio = (candReport.p95LatencyMs / baseReport.p95LatencyMs).toFixed(2);
      gateFailures.push(
        `p95LatencyMs ratio ${ratio}x exceeded maxP95LatencyRatio ${this.#maxP95LatencyRatio.toFixed(2)}x`
      );
    }

    return {
      passed: gateFailures.length === 0,
      verdict: gateFailures.length === 0 ? "PASS" : "REGRESSION_DETECTED",
      gateFailures,
      matchedCases: caseDiffs.length,
      addedCaseIds,
      removedCaseIds,
      baseline: {
        totalCases: baseReport.totalCases,
        faithfulness: baseReport.faithfulness,
        contextPrecision: baseReport.contextPrecision,
        recallAtK: baseReport.recallAtK,
        p50LatencyMs: baseReport.p50LatencyMs ?? 0,
        p95LatencyMs: baseReport.p95LatencyMs ?? 0
      },
      candidate: {
        totalCases: candReport.totalCases,
        faithfulness: candReport.faithfulness,
        contextPrecision: candReport.contextPrecision,
        recallAtK: candReport.recallAtK,
        p50LatencyMs: candReport.p50LatencyMs ?? 0,
        p95LatencyMs: candReport.p95LatencyMs ?? 0
      },
      deltas,
      drift,
      caseBreakdown: {
        improved: improvements.length,
        regressed: regressions.length,
        unchanged: unchangedCount
      },
      regressions,
      improvements
    };
  }
}

export function formatDiffReport(diff, format = "table") {
  if (format === "json") {
    return JSON.stringify(diff, null, 2);
  }
  const fmtSigned = (n) => (n >= 0 ? `+${n.toFixed(4)}` : n.toFixed(4));
  if (format === "markdown") {
    const lines = [
      `### Pairwise RAG A/B Regression Diff (${diff.verdict})`,
      "",
      "| Metric | Baseline | Candidate | Delta |",
      "| :--- | ---: | ---: | ---: |",
      `| Faithfulness | ${diff.baseline.faithfulness.toFixed(4)} | ${diff.candidate.faithfulness.toFixed(4)} | ${fmtSigned(diff.deltas.faithfulness)} |`,
      `| Context Precision (AP@K) | ${diff.baseline.contextPrecision.toFixed(4)} | ${diff.candidate.contextPrecision.toFixed(4)} | ${fmtSigned(diff.deltas.contextPrecision)} |`,
      `| Recall@K | ${diff.baseline.recallAtK.toFixed(4)} | ${diff.candidate.recallAtK.toFixed(4)} | ${fmtSigned(diff.deltas.recallAtK)} |`,
      "",
      `- **Matched Cases**: \`${diff.matchedCases}\` (\`+${diff.caseBreakdown.improved}\` improved, \`-${diff.caseBreakdown.regressed}\` regressed, \`${diff.caseBreakdown.unchanged}\` unchanged)`,
      `- **Paired Effect**: mean $\\Delta Q = ${fmtSigned(diff.drift.pairedStats.meanDelta)}$, $t = ${diff.drift.pairedStats.pairedTStat.toFixed(2)}$, win rate \`${(diff.drift.pairedStats.winRate * 100).toFixed(1)}%\``
    ];
    if (diff.drift.meanChunkOverlapJaccard !== null) {
      lines.push(
        `- **Chunking & Prompt Drift**: chunk Jaccard \`${diff.drift.meanChunkOverlapJaccard.toFixed(4)}\`, chunk words $\\Delta = ${diff.drift.meanChunkWordsDelta >= 0 ? "+" : ""}${diff.drift.meanChunkWordsDelta}$, answer cosine \`${diff.drift.meanAnswerCosineSimilarity.toFixed(4)}\``
      );
    }
    if (diff.gateFailures.length > 0) {
      lines.push("", "**Gate Failures:**", ...diff.gateFailures.map((g) => `- ${g}`));
    }
    return lines.join("\n");
  }

  const pad = (s, w) => String(s).padEnd(w, " ");
  const padL = (s, w) => String(s).padStart(w, " ");
  const rows = [
    ["Faithfulness", diff.baseline.faithfulness.toFixed(4), diff.candidate.faithfulness.toFixed(4), fmtSigned(diff.deltas.faithfulness)],
    ["Context Precision", diff.baseline.contextPrecision.toFixed(4), diff.candidate.contextPrecision.toFixed(4), fmtSigned(diff.deltas.contextPrecision)],
    ["Recall@K", diff.baseline.recallAtK.toFixed(4), diff.candidate.recallAtK.toFixed(4), fmtSigned(diff.deltas.recallAtK)]
  ];
  const out = [
    `Pairwise A/B Prompt & Chunking Diff [${diff.verdict}] (matched=${diff.matchedCases}, +${diff.caseBreakdown.improved}/-${diff.caseBreakdown.regressed}/=${diff.caseBreakdown.unchanged})`,
    `${pad("Metric", 20)} ${padL("Baseline", 10)} ${padL("Candidate", 10)} ${padL("Delta", 10)}`,
    "-".repeat(53),
    ...rows.map(([m, b, c, d]) => `${pad(m, 20)} ${padL(b, 10)} ${padL(c, 10)} ${padL(d, 10)}`)
  ];
  if (diff.drift.meanChunkOverlapJaccard !== null) {
    out.push(
      `Drift: chunkJaccard=${diff.drift.meanChunkOverlapJaccard.toFixed(4)} chunkWordsDelta=${diff.drift.meanChunkWordsDelta} answerCosine=${diff.drift.meanAnswerCosineSimilarity.toFixed(4)} pairedT=${diff.drift.pairedStats.pairedTStat.toFixed(2)}`
    );
  }
  if (diff.gateFailures.length > 0) {
    out.push(`Gate Failures: ${diff.gateFailures.join("; ")}`);
  }
  return out.join("\n");
}

const CASE_ARRAY_SCHEMA = {
  type: "array",
  items: {
    type: "object",
    required: ["id", "query", "relevantDocIds", "retrievedDocs", "generatedAnswer"],
    additionalProperties: false,
    properties: {
      id: { type: "string", minLength: 1 },
      query: { type: "string", minLength: 1 },
      relevantDocIds: { type: "array", items: { type: "string" } },
      retrievedDocs: {
        type: "array",
        items: {
          type: "object",
          required: ["id", "text"],
          additionalProperties: false,
          properties: {
            id: { type: "string", minLength: 1 },
            text: { type: "string" }
          }
        }
      },
      generatedAnswer: { type: "string" }
    }
  }
};

const EVALUATE_REQUEST_SCHEMA = {
  type: "object",
  required: ["cases"],
  additionalProperties: false,
  properties: {
    topK: { type: "integer", minimum: 1, maximum: 50 },
    cases: CASE_ARRAY_SCHEMA
  }
};

const DIFF_REQUEST_SCHEMA = {
  type: "object",
  required: ["baseline", "candidate"],
  additionalProperties: false,
  properties: {
    topK: { type: "integer", minimum: 1, maximum: 50 },
    maxFaithfulnessDrop: { type: "number", minimum: 0, maximum: 1 },
    maxContextPrecisionDrop: { type: "number", minimum: 0, maximum: 1 },
    maxRecallDrop: { type: "number", minimum: 0, maximum: 1 },
    maxCaseRegressions: { type: "integer", minimum: 0, maximum: 10000 },
    baseline: CASE_ARRAY_SCHEMA,
    candidate: CASE_ARRAY_SCHEMA
  }
};

const QUERY_REQUEST_SCHEMA = {
  type: "object",
  required: ["query"],
  additionalProperties: false,
  properties: {
    query: { type: "string", minLength: 1, maxLength: 8192 },
    clientId: { type: "string", minLength: 1, maxLength: 128 },
    estimatedTokens: { type: "integer", minimum: 1, maximum: 1_000_000 },
    topK: { type: "integer", minimum: 1, maximum: 20 },
    documents: {
      type: "array",
      items: {
        type: "object",
        required: ["id", "text"],
        additionalProperties: false,
        properties: {
          id: { type: "string", minLength: 1 },
          text: { type: "string", minLength: 1 }
        }
      }
    }
  }
};

const DEFAULT_KNOWLEDGE_CORPUS = Object.freeze([
  {
    id: "doc-crc32c-ring",
    text: "Castagnoli CRC32C checksums protect 64 byte cache line aligned C11 and Rust FFI ring slot headers against false sharing and silent corruption."
  },
  {
    id: "doc-hybrid-rrf",
    text: "Reciprocal rank fusion merges SQLite WAL FTS5 BM25 lexical rankings with 256 dimensional L2 normalized cosine vectors using constant k equals 60."
  },
  {
    id: "doc-sparse-moe",
    text: "Sparse Top 2 Mixture of Experts routing filters quantized models by VRAM cap and latency SLA before computing softmax expert gating weights."
  },
  {
    id: "doc-p2-quantiles",
    text: "Jain Chlamtac piecewise parabolic P2 estimators track streaming p50, p95, and p99 tail latencies in constant O(1) memory without sample arrays."
  },
  {
    id: "doc-sliding-ratelimit",
    text: "Deterministic token budget sliding window rate limiters enforce per client request and token quotas with exact millisecond Retry After headers."
  },
  {
    id: "doc-otlp-tracing",
    text: "OpenTelemetry OTLP JSON span exporters propagate W3C traceparent headers and aggregate latency histograms with automatic PII redaction."
  }
]);

function rankCorpusRrf(query, docs, topK = 3, rrfK = 60) {
  const qWords = contentWords(query);
  const qVec = embedNgramCosine(query, 256);
  const entries = (docs ?? []).map((doc) => {
    const dWords = new Set(contentWords(doc.text));
    let lexicalHits = 0;
    for (const w of qWords) {
      if (dWords.has(w)) {
        lexicalHits += 1;
      }
    }
    const cosine = Math.max(0, dotCosine(qVec, embedNgramCosine(doc.text, 256)));
    return { id: doc.id, text: doc.text, lexicalHits, cosine };
  });

  const byLex = [...entries]
    .filter((e) => e.lexicalHits > 0)
    .sort((a, b) => b.lexicalHits - a.lexicalHits || b.cosine - a.cosine);
  const lexRank = new Map(byLex.map((e, idx) => [e.id, idx + 1]));

  const byVec = [...entries].sort((a, b) => b.cosine - a.cosine);
  const vecRank = new Map(byVec.map((e, idx) => [e.id, idx + 1]));

  return entries
    .map((e) => {
      const vr = vecRank.get(e.id) ?? entries.length;
      const lr = lexRank.get(e.id);
      const rrfScore = 1.0 / (rrfK + vr) + (lr !== undefined ? 1.25 / (rrfK + lr) : 0.0);
      return {
        id: e.id,
        text: e.text,
        rrfScore: Number(rrfScore.toFixed(6)),
        cosine: Number(e.cosine.toFixed(4)),
        lexicalHits: e.lexicalHits
      };
    })
    .sort((a, b) => b.rrfScore - a.rrfScore || b.cosine - a.cosine)
    .slice(0, Math.max(1, topK));
}

function readJsonBody(req, maxBytes = 1_048_576) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    req.on("data", (chunk) => {
      total += chunk.length;
      if (total > maxBytes) {
        reject(new Error("payload_too_large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        const text = Buffer.concat(chunks).toString("utf8");
        resolve(text ? JSON.parse(text) : {});
      } catch {
        reject(new Error("invalid_json"));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, statusCode, payload, extraHeaders = {}) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "X-Content-Type-Options": "nosniff",
    ...extraHeaders
  });
  res.end(body);
}

function resolveClientId(req, explicitClientId, validator) {
  const raw =
    explicitClientId ||
    (typeof req.headers["x-client-id"] === "string" ? req.headers["x-client-id"] : "") ||
    req.socket?.remoteAddress ||
    "anonymous";
  return validator.redact(String(raw).trim()).sanitized || "anonymous";
}

function estimateQueryTokens(queryText, explicitEstimate) {
  if (typeof explicitEstimate === "number" && Number.isFinite(explicitEstimate) && explicitEstimate >= 1) {
    return Math.ceil(explicitEstimate);
  }
  return Math.max(32, Math.ceil(String(queryText).length / 4) + 96);
}

/**
 * Creates a zero-dependency Node.js 22 HTTP + SSE streaming studio gateway (`v1.2.0`).
 */
export function createStudioServer(options = {}) {
  const validator = new SchemaToolValidator();
  const cache = new SemanticLruCache(options.cacheOptions);
  const breaker = new TokenBudgetCircuitBreaker(options.breakerOptions);
  const rateLimiter = new SlidingWindowTokenRateLimiter(options.rateLimiterOptions);
  const telemetry = new OtlpTraceExporter({
    serviceVersion: "1.2.0",
    validator,
    ...(options.telemetryOptions ?? {})
  });
  const evaluator = new GoldenEvalRunner({ topK: options.topK ?? 5 });
  const differ = new RegressionDiffAnalyzer({ topK: options.topK ?? 5, ...(options.diffOptions ?? {}) });
  const defaultCorpus = Array.isArray(options.corpus) && options.corpus.length > 0 ? options.corpus : DEFAULT_KNOWLEDGE_CORPUS;

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    const incomingTraceparent = typeof req.headers.traceparent === "string" ? req.headers.traceparent : undefined;

    if (req.method === "GET" && url.pathname === "/health") {
      return sendJson(res, 200, {
        status: "ok",
        service: "fullstack-rag-eval-studio",
        version: "1.2.0",
        cache: cache.stats(),
        breaker: breaker.snapshot(),
        rateLimiter: rateLimiter.snapshot(),
        telemetry: telemetry.stats()
      });
    }

    if (req.method === "GET" && url.pathname === "/v1/traces") {
      const payload = telemetry.exportOtlpJson();
      if (url.searchParams.get("clear") === "1") {
        telemetry.clear();
      }
      return sendJson(res, 200, payload);
    }

    if (req.method === "POST" && url.pathname === "/v1/evaluate") {
      const span = telemetry.startSpan("HTTP POST /v1/evaluate", {
        kind: 2,
        traceparent: incomingTraceparent,
        metricRoute: "/v1/evaluate",
        attributes: { "http.method": "POST", "http.route": "/v1/evaluate" }
      });
      const traceHeaders = { traceparent: span.traceparent, "X-Trace-Id": span.traceId };
      try {
        const body = await readJsonBody(req);
        const checked = validator.validateAndRedact(body, EVALUATE_REQUEST_SCHEMA);
        if (!checked.valid) {
          span.setStatus(2, "validation_failed");
          span.end({ "http.status_code": 400 });
          return sendJson(res, 400, { ok: false, errors: checked.errors, traceId: span.traceId }, traceHeaders);
        }
        const report = await evaluator.runSuite(checked.sanitized.cases);
        span.end({
          "http.status_code": 200,
          "rag.eval.cases": report.totalCases,
          "rag.eval.faithfulness": report.faithfulness,
          "rag.redaction.count": checked.redactionCount
        });
        return sendJson(
          res,
          200,
          {
            ok: true,
            redactionCount: checked.redactionCount,
            traceId: span.traceId,
            report
          },
          traceHeaders
        );
      } catch (err) {
        span.setStatus(2, err.message);
        span.end({ "http.status_code": 400 });
        return sendJson(res, 400, { ok: false, errors: [err.message], traceId: span.traceId }, traceHeaders);
      }
    }

    if (req.method === "POST" && url.pathname === "/v1/diff") {
      const span = telemetry.startSpan("HTTP POST /v1/diff", {
        kind: 2,
        traceparent: incomingTraceparent,
        metricRoute: "/v1/diff",
        attributes: { "http.method": "POST", "http.route": "/v1/diff" }
      });
      const traceHeaders = { traceparent: span.traceparent, "X-Trace-Id": span.traceId };
      try {
        const body = await readJsonBody(req);
        const checked = validator.validateAndRedact(body, DIFF_REQUEST_SCHEMA);
        if (!checked.valid) {
          span.setStatus(2, "validation_failed");
          span.end({ "http.status_code": 400 });
          return sendJson(res, 400, { ok: false, errors: checked.errors, traceId: span.traceId }, traceHeaders);
        }
        const reqAnalyzer = new RegressionDiffAnalyzer({
          topK: checked.sanitized.topK ?? options.topK ?? 5,
          maxFaithfulnessDrop: checked.sanitized.maxFaithfulnessDrop,
          maxContextPrecisionDrop: checked.sanitized.maxContextPrecisionDrop,
          maxRecallDrop: checked.sanitized.maxRecallDrop,
          maxCaseRegressions: checked.sanitized.maxCaseRegressions
        });
        const diff = await reqAnalyzer.compare(checked.sanitized.baseline, checked.sanitized.candidate);
        span.end({
          "http.status_code": 200,
          "rag.diff.matched_cases": diff.matchedCases,
          "rag.diff.passed": diff.passed,
          "rag.redaction.count": checked.redactionCount
        });
        return sendJson(
          res,
          200,
          {
            ok: true,
            redactionCount: checked.redactionCount,
            traceId: span.traceId,
            diff
          },
          traceHeaders
        );
      } catch (err) {
        span.setStatus(2, err.message);
        span.end({ "http.status_code": 400 });
        return sendJson(res, 400, { ok: false, errors: [err.message], traceId: span.traceId }, traceHeaders);
      }
    }

    if ((req.method === "GET" || req.method === "POST") && url.pathname === "/v1/query") {
      const reqStartMs = performance.now();
      const span = telemetry.startSpan(`HTTP ${req.method} /v1/query`, {
        kind: 2,
        traceparent: incomingTraceparent,
        metricRoute: "/v1/query",
        attributes: { "http.method": req.method, "http.route": "/v1/query" }
      });
      const traceHeaders = { traceparent: span.traceparent, "X-Trace-Id": span.traceId };

      let sanitizedReq;
      let redactionCount = 0;
      if (req.method === "POST") {
        try {
          const body = await readJsonBody(req);
          const checked = validator.validateAndRedact(body, QUERY_REQUEST_SCHEMA);
          if (!checked.valid) {
            span.setStatus(2, "validation_failed");
            span.end({ "http.status_code": 400 });
            return sendJson(res, 400, { ok: false, errors: checked.errors, traceId: span.traceId }, traceHeaders);
          }
          sanitizedReq = checked.sanitized;
          redactionCount = checked.redactionCount;
        } catch (err) {
          span.setStatus(2, err.message);
          span.end({ "http.status_code": 400 });
          return sendJson(res, 400, { ok: false, errors: [err.message], traceId: span.traceId }, traceHeaders);
        }
      } else {
        const rawQ = url.searchParams.get("query") ?? "";
        if (!rawQ.trim()) {
          span.setStatus(2, "missing_query");
          span.end({ "http.status_code": 400 });
          return sendJson(res, 400, { ok: false, errors: ["$.query: missing required query"], traceId: span.traceId }, traceHeaders);
        }
        const redactedQ = validator.redact(rawQ);
        redactionCount = redactedQ.redactionCount;
        const rawEst = url.searchParams.get("estimatedTokens");
        const rawTopK = url.searchParams.get("topK");
        sanitizedReq = {
          query: redactedQ.sanitized,
          clientId: url.searchParams.get("clientId") ?? undefined,
          estimatedTokens: rawEst ? Number.parseInt(rawEst, 10) : undefined,
          topK: rawTopK ? Number.parseInt(rawTopK, 10) : undefined
        };
      }

      const clientId = resolveClientId(req, sanitizedReq.clientId, validator);
      const estTokens = estimateQueryTokens(sanitizedReq.query, sanitizedReq.estimatedTokens);
      const rlDecision = rateLimiter.consume(clientId, estTokens);
      telemetry.recordRateLimitDecision(rlDecision.allowed);
      const outHeaders = { ...rateLimiter.toHeaders(rlDecision), ...traceHeaders };

      span.addEvent("rate_limit.checked", {
        "rag.client.id": clientId,
        "rag.rate_limit.allowed": rlDecision.allowed,
        "rag.rate_limit.requested_tokens": estTokens,
        "rag.rate_limit.remaining_tokens": rlDecision.remainingTokens
      });

      if (!rlDecision.allowed) {
        span.setStatus(2, rlDecision.reason);
        span.end({
          "http.status_code": 429,
          "rag.client.id": clientId,
          "rag.rate_limit.allowed": false,
          "rag.rate_limit.retry_after_ms": rlDecision.retryAfterMs
        });
        return sendJson(
          res,
          429,
          {
            ok: false,
            error: "rate_limit_exceeded",
            rateLimit: rlDecision,
            traceId: span.traceId
          },
          outHeaders
        );
      }

      const estCostUsd = Number((estTokens * 0.000006).toFixed(6));
      const routeInfo = breaker.route({ estimatedTokens: estTokens, estimatedCostUsd: estCostUsd });
      if (!routeInfo.allowed) {
        span.setStatus(2, routeInfo.reason);
        span.end({
          "http.status_code": 429,
          "rag.client.id": clientId,
          "rag.breaker.allowed": false
        });
        return sendJson(
          res,
          429,
          {
            ok: false,
            error: "circuit_budget_exceeded",
            route: routeInfo,
            rateLimit: rlDecision,
            traceId: span.traceId
          },
          outHeaders
        );
      }

      const topK = Math.max(1, Math.min(20, sanitizedReq.topK ?? options.topK ?? 3));
      const customDocs = Array.isArray(sanitizedReq.documents) && sanitizedReq.documents.length > 0;
      const cacheKey = `${topK}:${sanitizedReq.query}`;
      const lookup = customDocs ? { hit: false, matchType: "miss", similarity: 0 } : cache.get(cacheKey);

      let answer;
      let citations;
      if (lookup.hit && lookup.value && typeof lookup.value === "object" && typeof lookup.value.answer === "string") {
        answer = lookup.value.answer;
        citations = lookup.value.citations;
      } else {
        citations = rankCorpusRrf(sanitizedReq.query, customDocs ? sanitizedReq.documents : defaultCorpus, topK);
        answer = citations.slice(0, 2).map((c) => c.text).join(" ");
        if (!customDocs) {
          cache.set(cacheKey, { answer, citations });
        }
      }

      const actualTokens = Math.max(16, Math.ceil((sanitizedReq.query.length + answer.length) / 4));
      breaker.record({
        model: routeInfo.selectedModel,
        tokens: actualTokens,
        costUsd: Number((actualTokens * 0.000006).toFixed(6)),
        ok: true
      });

      const latencyMs = Number((performance.now() - reqStartMs).toFixed(4));
      span.end({
        "http.status_code": 200,
        "rag.client.id": clientId,
        "rag.cache.hit": lookup.hit,
        "rag.cache.match_type": lookup.matchType,
        "rag.model.selected": routeInfo.selectedModel,
        "rag.model.tier": routeInfo.tier,
        "rag.tokens.estimated": estTokens,
        "rag.tokens.actual": actualTokens,
        "rag.redaction.count": redactionCount,
        "rag.citations.count": citations.length
      });

      return sendJson(
        res,
        200,
        {
          ok: true,
          answer,
          citations,
          route: routeInfo,
          cache: {
            hit: lookup.hit,
            matchType: lookup.matchType,
            similarity: lookup.similarity
          },
          rateLimit: rlDecision,
          redactionCount,
          traceId: span.traceId,
          spanId: span.spanId,
          latencyMs
        },
        outHeaders
      );
    }

    if ((req.method === "GET" || req.method === "POST") && url.pathname === "/v1/stream") {
      const span = telemetry.startSpan(`HTTP ${req.method} /v1/stream`, {
        kind: 2,
        traceparent: incomingTraceparent,
        metricRoute: "/v1/stream",
        attributes: { "http.method": req.method, "http.route": "/v1/stream" }
      });
      const traceHeaders = { traceparent: span.traceparent, "X-Trace-Id": span.traceId };

      let rawQuery = url.searchParams.get("query") ?? "hybrid rag bm25 cosine rrf architecture";
      let bodyClientId = url.searchParams.get("clientId") ?? undefined;
      let explicitEstTokens = url.searchParams.get("estimatedTokens")
        ? Number.parseInt(url.searchParams.get("estimatedTokens") ?? "", 10)
        : undefined;

      if (req.method === "POST") {
        try {
          const body = await readJsonBody(req);
          if (typeof body.query === "string" && body.query.trim()) {
            rawQuery = body.query;
          }
          if (typeof body.clientId === "string" && body.clientId.trim()) {
            bodyClientId = body.clientId;
          }
          if (typeof body.estimatedTokens === "number") {
            explicitEstTokens = body.estimatedTokens;
          }
        } catch (err) {
          span.setStatus(2, err.message);
          span.end({ "http.status_code": 400 });
          return sendJson(res, 400, { ok: false, errors: [err.message], traceId: span.traceId }, traceHeaders);
        }
      }

      const { sanitized: cleanQuery, redactionCount } = validator.redact(rawQuery);
      const clientId = resolveClientId(req, bodyClientId, validator);
      const estTokens = explicitEstTokens ?? 160;
      const rlDecision = rateLimiter.consume(clientId, estTokens);
      telemetry.recordRateLimitDecision(rlDecision.allowed);
      const outHeaders = { ...rateLimiter.toHeaders(rlDecision), ...traceHeaders };

      if (!rlDecision.allowed) {
        span.setStatus(2, rlDecision.reason);
        span.end({
          "http.status_code": 429,
          "rag.client.id": clientId,
          "rag.rate_limit.allowed": false,
          "rag.rate_limit.retry_after_ms": rlDecision.retryAfterMs
        });
        return sendJson(
          res,
          429,
          {
            ok: false,
            error: "rate_limit_exceeded",
            rateLimit: rlDecision,
            traceId: span.traceId
          },
          outHeaders
        );
      }

      const routeInfo = breaker.route({ estimatedTokens: estTokens, estimatedCostUsd: 0.001 });
      if (!routeInfo.allowed) {
        span.setStatus(2, routeInfo.reason);
        span.end({
          "http.status_code": 429,
          "rag.client.id": clientId,
          "rag.breaker.allowed": false
        });
        return sendJson(
          res,
          429,
          {
            ok: false,
            error: "circuit_budget_exceeded",
            route: routeInfo,
            rateLimit: rlDecision,
            traceId: span.traceId
          },
          outHeaders
        );
      }

      res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
        ...outHeaders
      });

      const writeSse = (event, data) => {
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      };

      const lookup = cache.get(cleanQuery);
      writeSse("route", {
        selectedModel: routeInfo.selectedModel,
        tier: routeInfo.tier,
        cacheHit: lookup.hit,
        matchType: lookup.matchType,
        redactionCount,
        rateLimit: rlDecision,
        traceId: span.traceId,
        spanId: span.spanId
      });

      const cachedVal = lookup.hit ? lookup.value : null;
      const answerText =
        typeof cachedVal === "string"
          ? cachedVal
          : typeof cachedVal?.answer === "string"
            ? cachedVal.answer
            : `Hybrid retrieval fused BM25 lexical recall with cosine vector similarity for "${cleanQuery}".`;
      if (!lookup.hit) {
        cache.set(cleanQuery, answerText);
      }

      const words = answerText.split(" ");
      let chunksEmitted = 0;
      for (let i = 0; i < words.length; i += 3) {
        writeSse("token", {
          index: i / 3,
          delta: words.slice(i, i + 3).join(" ") + (i + 3 < words.length ? " " : "")
        });
        chunksEmitted += 1;
      }

      breaker.record({ model: routeInfo.selectedModel, tokens: 96, costUsd: 0.0008, ok: true });
      span.end({
        "http.status_code": 200,
        "rag.client.id": clientId,
        "rag.cache.hit": lookup.hit,
        "rag.cache.match_type": lookup.matchType,
        "rag.model.selected": routeInfo.selectedModel,
        "rag.model.tier": routeInfo.tier,
        "rag.stream.chunks": chunksEmitted,
        "rag.tokens.estimated": estTokens,
        "rag.redaction.count": redactionCount
      });
      writeSse("metrics", {
        cache: cache.stats(),
        breaker: breaker.snapshot(),
        rateLimiter: rateLimiter.snapshot(),
        telemetry: telemetry.stats()
      });
      writeSse("done", { done: true, traceId: span.traceId });
      res.end();
      return;
    }

    return sendJson(res, 404, { ok: false, errors: ["route_not_found"] });
  });

  return {
    server,
    validator,
    cache,
    breaker,
    rateLimiter,
    telemetry,
    evaluator,
    differ,
    listen(port = 0, host = "127.0.0.1") {
      return new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(port, host, () => {
          const addr = server.address();
          const boundPort = typeof addr === "object" && addr ? addr.port : port;
          resolve({ port: boundPort, host, url: `http://${host}:${boundPort}` });
        });
      });
    },
    close() {
      return new Promise((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  };
}
